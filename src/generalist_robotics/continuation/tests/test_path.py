"""Unit tests for the morphology continuation loop."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from generalist_robotics.continuation import path
from generalist_robotics.evaluation.rollout import RolloutStats
from generalist_robotics.morphology.scaling import MorphParams
from generalist_robotics.training import ppo

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

ROBOT = "berkeley_humanoid"

# Body scale of the fake robot, in metres. With it, the Froude floor of 0.01 asks for
# 0.22 m/s, so WALKING_SPEED is comfortably viable and FALLEN_SPEED is not.
LEG_LENGTH = 0.5
WALKING_SPEED = 0.5
FALLEN_SPEED = 0.02

# Environment steps a fake fine-tune round reports, small enough to check by hand.
FAKE_FINETUNE_STEPS = 1_000

# The target of every fake path: a robot twice as long, so alpha and size_scale are tied
# by size = 2**alpha and a capability expressed as a size is easy to read.
DOUBLE_SIZE = MorphParams(size_scale=2.0)

# Budget of the real end-to-end run, kept to one fine-tune round on a smoke-sized PPO
# config; it exercises the whole stack rather than trying to learn anything.
INTEGRATION_EPISODE_LENGTH = 40


def make_stats(speed: float, survived: float = 1.0) -> RolloutStats:
    """Rollout statistics of a fake robot walking at the given speed."""
    return RolloutStats(
        survived_fraction=survived,
        mean_forward_velocity=speed,
        mean_forward_speed=speed,
        distance_travelled=speed * survived,
        net_displacement=speed * survived,
        episode_return=100.0 * survived,
        nominal_leg_length=LEG_LENGTH,
        num_episodes=1,
        num_steps=100,
    )


class FakeWorld:
    """A simulator stand-in in which a policy walks on every body up to a size it knows.

    The fake policy parameters are that size, so viability, the effect of fine-tuning and
    the cost of it are all one number and a whole walk can be predicted by hand. One
    fine-tune round multiplies the known size by gain, capped at the body being trained
    on, so gain sets how far a single step may reach: gain = 1 never learns anything and
    forces backtracking, a huge gain always succeeds in one round.
    """

    def __init__(self, gain: float = 1.0, log_path: pathlib.Path | None = None) -> None:
        self.gain = gain
        self.log_path = log_path
        self.evaluated: list[tuple[float, float]] = []
        self.finetuned: list[tuple[float, int]] = []
        self.log_lengths: list[int] = []

    def evaluate(
        self, config: path.ContinuationConfig, params: MorphParams, policy_params: object
    ) -> RolloutStats:
        """Stand in for evaluate_morphology."""
        del config
        self.evaluated.append((params.size_scale, float(policy_params)))
        if self.log_path is not None and self.log_path.exists():
            self.log_lengths.append(len(self.log_path.read_text().splitlines()))
        walks = params.size_scale <= float(policy_params) * (1.0 + 1e-9)
        if walks:
            return make_stats(WALKING_SPEED)
        return make_stats(FALLEN_SPEED, survived=0.2)

    def finetune(
        self,
        config: path.ContinuationConfig,
        params: MorphParams,
        policy_params: object,
        index: int,
        round_index: int,
    ) -> ppo.TrainingResult:
        """Stand in for finetune_on_morphology."""
        del config, index
        self.finetuned.append((params.size_scale, round_index))
        learned = min(params.size_scale, float(policy_params) * self.gain)
        return ppo.TrainingResult(
            params=learned,
            metrics={},
            num_timesteps=FAKE_FINETUNE_STEPS,
            wall_clock_seconds=0.1,
            steps_per_second=FAKE_FINETUNE_STEPS / 0.1,
        )

    def walk(self, **kwargs: object) -> path.ContinuationResult:
        """Run walk_morphology_path against this fake world."""
        arguments: dict[str, object] = {
            "robot": ROBOT,
            "start": MorphParams(),
            "end": DOUBLE_SIZE,
            "init_policy_params": 1.0,
            "finetune_timesteps": FAKE_FINETUNE_STEPS,
        }
        arguments.update(kwargs)
        with (
            mock.patch.object(path, "evaluate_morphology", self.evaluate),
            mock.patch.object(path, "finetune_on_morphology", self.finetune),
        ):
            return path.walk_morphology_path(**arguments)


def alphas_of(result: path.ContinuationResult) -> list[float]:
    """Path coordinates of every waypoint, rounded to a comparable precision."""
    return [round(waypoint.alpha, 6) for waypoint in result.waypoints]


class FreePathTest(unittest.TestCase):
    """A policy that already covers the target walks the path without any training."""

    def setUp(self):
        self.world = FakeWorld()
        self.result = self.world.walk(init_policy_params=2.0)

    def test_reaches_the_target(self):
        self.assertTrue(self.result.reached_target)
        self.assertEqual(alphas_of(self.result)[-1], 1.0)

    def test_costs_nothing(self):
        self.assertEqual(self.result.total_finetune_steps, 0)
        self.assertEqual([w.finetune_steps for w in self.result.waypoints], [0] * 6)
        self.assertEqual(self.world.finetuned, [])

    def test_no_waypoint_reports_a_correction(self):
        for waypoint in self.result.waypoints:
            self.assertTrue(waypoint.viable_before)
            self.assertIsNone(waypoint.stats_after)

    def test_stride_grows_while_transfer_is_free(self):
        # 0.1 grown by 1.5 each time, capped at 0.5, and clipped to the target.
        self.assertEqual(alphas_of(self.result), [0.0, 0.1, 0.25, 0.475, 0.8125, 1.0])

    def test_final_policy_is_the_untouched_one(self):
        self.assertEqual(self.result.final_policy_params, 2.0)


class FineTuningPathTest(unittest.TestCase):
    """A policy that starts on its own body must be trained onto every later one."""

    def setUp(self):
        self.world = FakeWorld(gain=1e6)
        self.result = self.world.walk(init_policy_params=1.0)

    def test_reaches_the_target_by_training(self):
        self.assertTrue(self.result.reached_target)
        self.assertAlmostEqual(self.result.final_policy_params, DOUBLE_SIZE.size_scale)

    def test_every_step_beyond_the_anchor_needed_one_round(self):
        corrected = [w for w in self.result.waypoints if w.alpha > 0.0]
        self.assertTrue(corrected)
        for waypoint in corrected:
            self.assertFalse(waypoint.viable_before)
            self.assertEqual(waypoint.finetune_steps, FAKE_FINETUNE_STEPS)
            self.assertIsNotNone(waypoint.stats_after)
            self.assertTrue(path.waypoint_accepted(waypoint))

    def test_stride_never_grows_when_training_is_needed(self):
        self.assertEqual(
            alphas_of(self.result), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

    def test_cumulative_steps_are_the_running_sum(self):
        running = 0
        for waypoint in self.result.waypoints:
            running += waypoint.finetune_steps
            self.assertEqual(waypoint.cumulative_steps, running)
        self.assertEqual(self.result.total_finetune_steps, running)
        self.assertEqual(running, 10 * FAKE_FINETUNE_STEPS)


class BacktrackingTest(unittest.TestCase):
    """A step too far is retreated from, and a smaller one is tried in its place."""

    def setUp(self):
        # One round buys 2%, three rounds 6.1%: not enough for a stride of 0.1, which
        # asks for 7.2% more size, and enough for the halved stride, which asks for 3.5%.
        self.world = FakeWorld(gain=1.02)
        self.result = self.world.walk(init_policy_params=1.0, step_alpha=0.1, min_step_alpha=0.01)

    def test_the_first_stride_is_rejected_and_halved(self):
        self.assertEqual(alphas_of(self.result)[:3], [0.0, 0.1, 0.05])
        rejected = self.result.waypoints[1]
        self.assertFalse(path.waypoint_accepted(rejected))
        self.assertEqual(rejected.finetune_steps, 3 * FAKE_FINETUNE_STEPS)

    def test_the_rejected_waypoint_does_not_move_the_policy_on(self):
        # The walk retries from the last accepted morphology, so the body at 0.05 is
        # evaluated with the policy the anchor ended with, not the one trained at 0.1.
        first_attempt_at_005 = next(
            evaluation for evaluation in self.world.evaluated if evaluation[0] == 2.0**0.05
        )
        self.assertEqual(first_attempt_at_005[1], 1.0)

    def test_the_smaller_stride_is_accepted(self):
        self.assertTrue(path.waypoint_accepted(self.result.waypoints[2]))

    def test_the_walk_still_arrives(self):
        self.assertTrue(self.result.reached_target)
        self.assertEqual(alphas_of(self.result)[-1], 1.0)

    def test_rejected_waypoints_are_charged_to_the_run(self):
        self.assertEqual(
            self.result.total_finetune_steps,
            sum(w.finetune_steps for w in self.result.waypoints),
        )
        self.assertEqual(
            self.result.total_finetune_steps, len(self.world.finetuned) * FAKE_FINETUNE_STEPS
        )


class GivingUpTest(unittest.TestCase):
    """A policy that cannot learn anything is reported as not having arrived."""

    def setUp(self):
        self.world = FakeWorld(gain=1.0)
        self.result = self.world.walk(init_policy_params=1.0, step_alpha=0.1, min_step_alpha=0.01)

    def test_reports_failure_honestly(self):
        self.assertFalse(self.result.reached_target)

    def test_halves_the_stride_until_it_hits_the_floor(self):
        self.assertEqual(alphas_of(self.result), [0.0, 0.1, 0.05, 0.025, 0.0125])

    def test_keeps_the_policy_it_started_with(self):
        self.assertEqual(self.result.final_policy_params, 1.0)

    def test_charges_every_failed_round(self):
        self.assertEqual(self.result.total_finetune_steps, 4 * 3 * FAKE_FINETUNE_STEPS)


class AnchorTest(unittest.TestCase):
    """The waypoint at alpha = 0 measures the policy on its own body and nothing else."""

    def test_is_recorded_first_at_the_start_morphology(self):
        result = FakeWorld().walk(init_policy_params=2.0)
        anchor = result.waypoints[0]
        self.assertEqual(anchor.alpha, 0.0)
        self.assertEqual(anchor.params, MorphParams())

    def test_is_never_fine_tuned_even_when_it_fails(self):
        world = FakeWorld(gain=1e6)
        result = world.walk(init_policy_params=0.5, step_alpha=0.1, min_step_alpha=0.1)
        anchor = result.waypoints[0]
        self.assertFalse(anchor.viable_before)
        self.assertEqual(anchor.finetune_steps, 0)
        self.assertIsNone(anchor.stats_after)
        trained_bodies = [body for body, _ in world.finetuned]
        self.assertEqual(trained_bodies[0], 2.0**0.1)
        self.assertNotIn(MorphParams().size_scale, trained_bodies)


class ValidationTest(unittest.TestCase):
    """Unusable step control is refused rather than silently repaired."""

    def test_rejects_a_non_positive_stride(self):
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=2.0, step_alpha=0.0)

    def test_rejects_a_stride_above_the_whole_path(self):
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=2.0, step_alpha=1.5)

    def test_rejects_a_floor_above_the_stride(self):
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=2.0, step_alpha=0.1, min_step_alpha=0.2)

    def test_rejects_an_empty_fine_tune_budget(self):
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=2.0, finetune_timesteps=0)

    def test_rejects_a_shrinking_stride_growth(self):
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=2.0, step_growth=0.5)


class RunLogTest(unittest.TestCase):
    """The JSONL log is written as the walk happens, so a dead run is still analysable."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log_path = pathlib.Path(self.directory.name) / "nested" / "run.jsonl"
        self.world = FakeWorld(gain=1e6, log_path=self.log_path)
        self.result = self.world.walk(
            init_policy_params=1.0, log_path=self.log_path, scale_time=True, scale_task=True
        )
        self.records = [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def tearDown(self):
        self.directory.cleanup()

    def test_opens_with_the_run_configuration(self):
        header = self.records[0]
        self.assertEqual(header["record"], path.CONFIG_RECORD)
        self.assertEqual(header["robot"], ROBOT)
        self.assertEqual(header["end"], {"size_scale": 2.0, "mass_scale": 1.0, "torque_scale": 1.0})
        self.assertEqual(header["env_kwargs"], {"scale_time": True, "scale_task": True})

    def test_holds_one_record_per_waypoint(self):
        waypoints = [row for row in self.records if row["record"] == path.WAYPOINT_RECORD]
        self.assertEqual(len(waypoints), len(self.result.waypoints))
        self.assertEqual([row["index"] for row in waypoints], list(range(len(waypoints))))

    def test_a_waypoint_record_carries_its_body_cost_and_verdict(self):
        row = self.records[2]
        self.assertAlmostEqual(row["alpha"], 0.1)
        self.assertAlmostEqual(row["params"]["size_scale"], 2.0**0.1)
        self.assertEqual(row["finetune_steps"], FAKE_FINETUNE_STEPS)
        self.assertEqual(row["cumulative_steps"], FAKE_FINETUNE_STEPS)
        self.assertFalse(row["viable_before"])
        self.assertTrue(row["accepted"])
        self.assertAlmostEqual(row["stats_after"]["mean_forward_speed"], WALKING_SPEED)

    def test_records_appear_before_the_run_ends(self):
        # The fake evaluates after each record is written, so the log grows under it.
        self.assertEqual(self.world.log_lengths[0], 1)
        self.assertGreater(self.world.log_lengths[-1], self.world.log_lengths[0])

    def test_save_run_log_summarises_the_finished_walk(self):
        summary_path = path.save_run_log(
            self.result, pathlib.Path(self.directory.name) / "run.json"
        )
        summary = json.loads(summary_path.read_text())
        self.assertTrue(summary["reached_target"])
        self.assertEqual(summary["total_finetune_steps"], self.result.total_finetune_steps)
        self.assertEqual(summary["num_waypoints"], len(self.result.waypoints))
        self.assertEqual(summary["num_accepted_waypoints"], len(self.result.waypoints))
        self.assertEqual(len(summary["waypoints"]), len(self.result.waypoints))


class WaypointAcceptedTest(unittest.TestCase):
    """A waypoint counts as reached if the policy walked there before or after training."""

    def make_waypoint(self, viable_before: bool, after: RolloutStats | None) -> path.Waypoint:
        """Build a waypoint with the given verdict before and statistics after."""
        return path.Waypoint(
            alpha=0.5,
            params=MorphParams(),
            stats_before=make_stats(WALKING_SPEED if viable_before else FALLEN_SPEED),
            viable_before=viable_before,
            stats_after=after,
            finetune_steps=0,
            cumulative_steps=0,
        )

    def test_free_transfer_is_accepted(self):
        self.assertTrue(path.waypoint_accepted(self.make_waypoint(True, None)))

    def test_failure_without_a_correction_is_rejected(self):
        self.assertFalse(path.waypoint_accepted(self.make_waypoint(False, None)))

    def test_a_correction_that_walks_is_accepted(self):
        waypoint = self.make_waypoint(False, make_stats(WALKING_SPEED))
        self.assertTrue(path.waypoint_accepted(waypoint))

    def test_a_correction_that_still_falls_is_rejected(self):
        waypoint = self.make_waypoint(False, make_stats(FALLEN_SPEED, survived=0.2))
        self.assertFalse(path.waypoint_accepted(waypoint))


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the real walk")
class PlaygroundIntegrationTest(unittest.TestCase):
    """One real waypoint on a real robot: env, PPO, evaluation, checkpoint and log.

    A smoke-sized PPO round cannot make a randomly initialised policy walk, so the walk is
    expected to reject its single step and report reached_target False; what is under test
    is that every seam between the continuation loop and the rest of the stack holds.
    """

    def test_walks_one_real_step_and_leaves_replayable_artifacts(self):
        from generalist_robotics.envs.locomotion import make_locomotion_env

        env = make_locomotion_env(ROBOT)
        cold = ppo.train_policy(
            ROBOT,
            num_timesteps=None,
            ppo_overrides=ppo.SMOKE_PPO_OVERRIDES,
            config_overrides={"episode_length": INTEGRATION_EPISODE_LENGTH},
        )
        del env

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            result = path.walk_morphology_path(
                robot=ROBOT,
                start=MorphParams(),
                end=MorphParams(size_scale=1.2),
                init_policy_params=cold.params,
                step_alpha=1.0,
                min_step_alpha=1.0,
                finetune_timesteps=ppo.SMOKE_PPO_OVERRIDES["num_timesteps"],
                max_finetune_rounds=1,
                num_eval_episodes=1,
                log_path=root / "run.jsonl",
                checkpoint_dir=root,
                ppo_overrides=ppo.SMOKE_PPO_OVERRIDES,
                scale_time=True,
                scale_task=True,
                config_overrides={"episode_length": INTEGRATION_EPISODE_LENGTH},
            )
            records = [json.loads(line) for line in (root / "run.jsonl").read_text().splitlines()]
            checkpoints = sorted(root.glob(f"{path.CHECKPOINT_PREFIX}_*/params"))
            progress = sorted((root / path.PROGRESS_DIRNAME).glob("*.jsonl"))

        self.assertEqual(len(result.waypoints), 2)
        self.assertEqual(len(records), 1 + len(result.waypoints))
        self.assertEqual(len(checkpoints), len(result.waypoints))
        self.assertEqual(len(progress), 1)
        self.assertGreater(result.waypoints[1].finetune_steps, 0)
        self.assertEqual(result.total_finetune_steps, result.waypoints[-1].cumulative_steps)
        self.assertGreater(result.waypoints[0].stats_before.nominal_leg_length, 0.0)


if __name__ == "__main__":
    unittest.main()
