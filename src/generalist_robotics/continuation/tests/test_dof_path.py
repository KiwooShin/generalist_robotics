"""Unit tests for the continuation loop that grows degrees of freedom along the path."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

from generalist_robotics.continuation import dof_path
from generalist_robotics.evaluation.rollout import RolloutStats
from generalist_robotics.morphology.scaling import MorphParams
from generalist_robotics.morphology.topology import DofLock
from generalist_robotics.training import ppo

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

ROBOT = "g1"

# Body scale of the fake robot, in metres: with it the Froude floor of 0.01 asks for
# 0.22 m/s, so WALKING_SPEED is comfortably viable and FALLEN_SPEED is not.
LEG_LENGTH = 0.5
WALKING_SPEED = 0.5
FALLEN_SPEED = 0.02

FAKE_FINETUNE_STEPS = 1_000

# The fake path: a robot twice as long whose locked joints come free on the way, so the
# hardest state is the far end and a capability expressed as one number reads directly.
DOUBLE_SIZE = MorphParams(size_scale=2.0)
LOCKED_JOINTS = ("waist_yaw_joint", "left_elbow_joint")
START_LOCKS = (DofLock(LOCKED_JOINTS, 1.0),)
END_LOCKS = (DofLock(LOCKED_JOINTS, 0.0),)

# Degrees of freedom the fake robot reports: twelve while locked, fourteen once free.
LOCKED_DOF = 12
FREE_DOF = 14

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


def difficulty(params: MorphParams, locks: tuple[DofLock, ...]) -> float:
    """How hard a body-and-lock state is, in the units the fake policy's capability uses.

    Both axes count: the body gets harder as it grows, and unlocking a joint the policy
    has never had to drive gets harder as the lock falls away.
    """
    lock = locks[0].lock if locks else 0.0
    return params.size_scale * (2.0 - lock)


def fake_active_dof(robot: str, locks: tuple[DofLock, ...]) -> int:
    """Stand in for waypoint_active_dof: the joints count once the lock is half gone."""
    del robot
    lock = locks[0].lock if locks else 0.0
    return LOCKED_DOF if lock > 0.5 else FREE_DOF


class FakeWorld:
    """A simulator stand-in in which a policy walks on every state up to a difficulty.

    The fake policy parameters are that difficulty, so viability, the effect of a
    fine-tune round and its cost are all one number and a whole walk can be predicted by
    hand. One round multiplies what the policy knows by gain, capped at the state being
    trained on, so gain sets how far one step may reach.
    """

    def __init__(self, gain: float = 1.0, log_path: pathlib.Path | None = None) -> None:
        self.gain = gain
        self.log_path = log_path
        self.evaluated: list[tuple[float, float]] = []
        self.finetuned: list[tuple[float, int]] = []
        self.log_lengths: list[int] = []

    def evaluate(
        self,
        config: dof_path.DofContinuationConfig,
        params: MorphParams,
        locks: tuple[DofLock, ...],
        policy_params: object,
    ) -> RolloutStats:
        """Stand in for evaluate_dof_morphology."""
        del config
        hardness = difficulty(params, locks)
        self.evaluated.append((hardness, float(policy_params)))
        if self.log_path is not None and self.log_path.exists():
            self.log_lengths.append(len(self.log_path.read_text().splitlines()))
        if hardness <= float(policy_params) * (1.0 + 1e-9):
            return make_stats(WALKING_SPEED)
        return make_stats(FALLEN_SPEED, survived=0.2)

    def finetune(
        self,
        config: dof_path.DofContinuationConfig,
        params: MorphParams,
        locks: tuple[DofLock, ...],
        policy_params: object,
        index: int,
        round_index: int,
    ) -> ppo.TrainingResult:
        """Stand in for finetune_on_dof_morphology."""
        del config, index
        hardness = difficulty(params, locks)
        self.finetuned.append((hardness, round_index))
        learned = min(hardness, float(policy_params) * self.gain)
        return ppo.TrainingResult(
            params=learned,
            metrics={},
            num_timesteps=FAKE_FINETUNE_STEPS,
            wall_clock_seconds=0.1,
            steps_per_second=FAKE_FINETUNE_STEPS / 0.1,
        )

    def walk(self, **kwargs: object) -> dof_path.DofContinuationResult:
        """Run walk_dof_path against this fake world."""
        arguments: dict[str, object] = {
            "robot": ROBOT,
            "start_params": MorphParams(),
            "end_params": DOUBLE_SIZE,
            "start_locks": START_LOCKS,
            "end_locks": END_LOCKS,
            "init_policy_params": 1.0,
            "finetune_timesteps": FAKE_FINETUNE_STEPS,
        }
        arguments.update(kwargs)
        with (
            mock.patch.object(dof_path, "evaluate_dof_morphology", self.evaluate),
            mock.patch.object(dof_path, "finetune_on_dof_morphology", self.finetune),
            mock.patch.object(dof_path, "waypoint_active_dof", fake_active_dof),
        ):
            return dof_path.walk_dof_path(**arguments)


def alphas_of(result: dof_path.DofContinuationResult) -> list[float]:
    """Path coordinates of every waypoint, rounded to a comparable precision."""
    return [round(waypoint.alpha, 6) for waypoint in result.waypoints]


class FreePathTest(unittest.TestCase):
    """A policy that already covers the target walks the whole path untouched."""

    def setUp(self):
        self.world = FakeWorld()
        self.result = self.world.walk(init_policy_params=4.0)

    def test_reaches_the_target(self):
        self.assertTrue(self.result.reached_target)
        self.assertEqual(alphas_of(self.result)[-1], 1.0)

    def test_costs_nothing(self):
        self.assertEqual(self.result.total_finetune_steps, 0)
        self.assertEqual(self.world.finetuned, [])

    def test_stride_grows_while_transfer_is_free(self):
        self.assertEqual(alphas_of(self.result), [0.0, 0.1, 0.25, 0.475, 0.8125, 1.0])

    def test_the_locks_are_interpolated_along_with_the_body(self):
        for waypoint in self.result.waypoints:
            self.assertEqual(len(waypoint.locks), 1)
            self.assertEqual(waypoint.locks[0].joint_names, LOCKED_JOINTS)
            self.assertAlmostEqual(waypoint.locks[0].lock, 1.0 - waypoint.alpha)

    def test_degrees_of_freedom_appear_along_the_path(self):
        counts = [waypoint.active_dof for waypoint in self.result.waypoints]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual((counts[0], counts[-1]), (LOCKED_DOF, FREE_DOF))


class FineTuningPathTest(unittest.TestCase):
    """A policy that starts on its own state must be trained onto every later one."""

    def setUp(self):
        self.world = FakeWorld(gain=1e6)
        self.result = self.world.walk(init_policy_params=1.0)

    def test_reaches_the_target_by_training(self):
        self.assertTrue(self.result.reached_target)
        self.assertAlmostEqual(self.result.final_policy_params, difficulty(DOUBLE_SIZE, END_LOCKS))

    def test_every_step_beyond_the_anchor_needed_one_round(self):
        corrected = [w for w in self.result.waypoints if w.alpha > 0.0]
        self.assertTrue(corrected)
        for waypoint in corrected:
            self.assertFalse(waypoint.viable_before)
            self.assertEqual(waypoint.finetune_steps, FAKE_FINETUNE_STEPS)
            self.assertTrue(dof_path.dof_waypoint_accepted(waypoint))

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

    def test_the_anchor_is_measured_but_never_corrected(self):
        anchor = self.result.waypoints[0]
        self.assertEqual(anchor.alpha, 0.0)
        self.assertEqual(anchor.finetune_steps, 0)
        self.assertIsNone(anchor.stats_after)


class BacktrackingTest(unittest.TestCase):
    """A step too far is retreated from, and a smaller one is tried in its place."""

    def setUp(self):
        self.world = FakeWorld(gain=1.05)
        self.result = self.world.walk(init_policy_params=1.0, step_alpha=0.1, min_step_alpha=0.01)

    def test_the_first_stride_is_rejected_and_halved(self):
        self.assertEqual(alphas_of(self.result)[:3], [0.0, 0.1, 0.05])
        rejected = self.result.waypoints[1]
        self.assertFalse(dof_path.dof_waypoint_accepted(rejected))
        self.assertEqual(rejected.finetune_steps, 3 * FAKE_FINETUNE_STEPS)

    def test_a_rejected_waypoint_is_kept_in_the_trajectory(self):
        rejected = [w for w in self.result.waypoints if not dof_path.dof_waypoint_accepted(w)]
        self.assertTrue(rejected)

    def test_the_walk_retreats_to_the_last_accepted_policy(self):
        # The evaluation right after a rejection is run with the policy from before it.
        for index, waypoint in enumerate(self.result.waypoints[:-1]):
            if dof_path.dof_waypoint_accepted(waypoint):
                continue
            before = self.result.waypoints[index - 1]
            self.assertLess(self.result.waypoints[index + 1].alpha, waypoint.alpha)
            self.assertGreaterEqual(self.result.waypoints[index + 1].alpha, before.alpha)

    def test_a_stride_floor_that_cannot_be_met_stops_the_walk(self):
        stalled = FakeWorld(gain=1.0).walk(
            init_policy_params=1.0, step_alpha=0.1, min_step_alpha=0.05
        )
        self.assertFalse(stalled.reached_target)


class ValidationTest(unittest.TestCase):
    """The walk refuses settings it cannot honour."""

    def bad(self, **kwargs: object) -> None:
        with self.assertRaises(ValueError):
            FakeWorld().walk(init_policy_params=4.0, **kwargs)

    def test_step_alpha_must_be_a_fraction_of_the_path(self):
        self.bad(step_alpha=0.0)
        self.bad(step_alpha=1.5)

    def test_the_stride_floor_must_not_exceed_the_stride(self):
        self.bad(step_alpha=0.1, min_step_alpha=0.2)

    def test_the_budgets_must_be_usable(self):
        self.bad(max_finetune_rounds=-1)
        self.bad(finetune_timesteps=0)


class LoggingTest(unittest.TestCase):
    """The run log is written as the walk goes, and carries enough to rebuild every body."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.log = self.root / "run.jsonl"
        self.world = FakeWorld(log_path=self.log)
        self.result = self.world.walk(init_policy_params=4.0, log_path=self.log)
        self.records = [json.loads(line) for line in self.log.read_text().splitlines()]

    def tearDown(self):
        self.directory.cleanup()

    def test_the_log_opens_with_the_run_configuration(self):
        head = self.records[0]
        self.assertEqual(head["record"], dof_path.CONFIG_RECORD)
        self.assertEqual(head["robot"], ROBOT)
        self.assertEqual(head["start_locks"], [{"joint_names": list(LOCKED_JOINTS), "lock": 1.0}])
        self.assertEqual(head["end_locks"], [{"joint_names": list(LOCKED_JOINTS), "lock": 0.0}])

    def test_every_waypoint_is_logged_with_its_body_and_its_locks(self):
        waypoints = [r for r in self.records if r["record"] == dof_path.WAYPOINT_RECORD]
        self.assertEqual(len(waypoints), len(self.result.waypoints))
        for record, waypoint in zip(waypoints, self.result.waypoints, strict=True):
            self.assertAlmostEqual(record["alpha"], waypoint.alpha)
            self.assertEqual(record["active_dof"], waypoint.active_dof)
            self.assertEqual(record["params"]["size_scale"], waypoint.params.size_scale)
            self.assertAlmostEqual(record["locks"][0]["lock"], waypoint.locks[0].lock)

    def test_records_are_flushed_as_the_walk_runs(self):
        self.assertEqual(self.world.log_lengths, list(range(1, len(self.world.log_lengths) + 1)))

    def test_the_summary_document_round_trips(self):
        path = dof_path.save_dof_run_log(self.result, self.root / "run.json")
        document = json.loads(path.read_text())
        self.assertTrue(document["reached_target"])
        self.assertEqual(document["num_waypoints"], len(self.result.waypoints))
        self.assertEqual(len(document["waypoints"]), len(self.result.waypoints))
        self.assertEqual(document["waypoints"][0]["locks"][0]["lock"], 1.0)


class CheckpointTest(unittest.TestCase):
    """Each waypoint leaves a checkpoint that knows which body it belongs to."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.result = FakeWorld().walk(init_policy_params=4.0, checkpoint_dir=self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_every_waypoint_records_its_own_body_and_locks(self):
        files = sorted(self.root.glob(f"*/{dof_path.WAYPOINT_MORPHOLOGY_FILENAME}"))
        self.assertEqual(len(files), len(self.result.waypoints))
        for file, waypoint in zip(files, self.result.waypoints, strict=True):
            document = json.loads(file.read_text())
            self.assertAlmostEqual(document["alpha"], waypoint.alpha)
            self.assertEqual(document["active_dof"], waypoint.active_dof)
            self.assertAlmostEqual(document["locks"][0]["lock"], waypoint.locks[0].lock)
            self.assertEqual(document["locks"][0]["joint_names"], list(LOCKED_JOINTS))

    def test_the_checkpoint_metadata_carries_the_scalar_summary(self):
        metadata = json.loads((self.root / "waypoint_000" / "metadata.json").read_text())
        self.assertEqual(metadata["metrics"]["active_dof"], float(LOCKED_DOF))
        self.assertEqual(metadata["metrics"]["size_scale"], 1.0)


class AcceptanceTest(unittest.TestCase):
    """What counts as having stayed on the solution branch."""

    def make_waypoint(self, viable_before: bool, stats_after: RolloutStats | None) -> "object":
        return dof_path.DofWaypoint(
            alpha=0.5,
            params=MorphParams(),
            locks=START_LOCKS,
            active_dof=LOCKED_DOF,
            stats_before=make_stats(WALKING_SPEED if viable_before else FALLEN_SPEED),
            viable_before=viable_before,
            stats_after=stats_after,
            finetune_steps=0,
            cumulative_steps=0,
        )

    def test_free_transfer_is_accepted(self):
        self.assertTrue(dof_path.dof_waypoint_accepted(self.make_waypoint(True, None)))

    def test_failure_without_a_correction_is_rejected(self):
        self.assertFalse(dof_path.dof_waypoint_accepted(self.make_waypoint(False, None)))

    def test_a_correction_that_walks_is_accepted(self):
        waypoint = self.make_waypoint(False, make_stats(WALKING_SPEED))
        self.assertTrue(dof_path.dof_waypoint_accepted(waypoint))

    def test_a_correction_that_still_falls_is_rejected(self):
        waypoint = self.make_waypoint(False, make_stats(FALLEN_SPEED, survived=0.2))
        self.assertFalse(dof_path.dof_waypoint_accepted(waypoint))


# A stand-in for the two-frame sensor layout G1 and T1 use.
FRAMED_XML = """
<mujoco model="framed">
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint/>
      <geom name="body" type="sphere" size="0.1" mass="1"/>
      <site name="pelvis_site"/>
    </body>
  </worldbody>
  <sensor>
    <framelinvel name="local_linvel_pelvis" objtype="site" objname="pelvis_site"/>
  </sensor>
</mujoco>
"""


class FakeFramedEnv:
    """An environment whose local-velocity accessor names the frame, as G1's does."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.mj_model = model
        self.calls: list[str] = []

    def get_local_linvel(self, data: object, frame: str) -> object:
        """Record which frame was asked for and hand the data straight back."""
        del data
        self.calls.append(frame)
        return frame


class LocalVelocityFrameTest(unittest.TestCase):
    """The rollout harness must be able to call the accessor with the data alone."""

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(FRAMED_XML)
        self.env = FakeFramedEnv(self.model)

    def test_the_base_frame_is_found_on_the_model(self):
        self.assertEqual(dof_path.local_velocity_frame(self.model), "pelvis")

    def test_binding_makes_the_frame_optional_without_removing_it(self):
        env = dof_path.bind_local_velocity_frame(self.env)
        self.assertEqual(env.get_local_linvel(None), "pelvis")
        self.assertEqual(env.get_local_linvel(None, "torso"), "torso")
        self.assertEqual(env.calls, ["pelvis", "torso"])

    def test_an_env_without_the_accessor_is_untouched(self):
        class Bare:
            """An env that exposes no local-velocity accessor at all."""

        bare = Bare()
        self.assertIs(dof_path.bind_local_velocity_frame(bare), bare)

    def test_a_model_without_the_sensor_is_reported(self):
        bare = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/>"
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        with self.assertRaises(RuntimeError):
            dof_path.local_velocity_frame(bare)


class FakeLockedEnv:
    """An env whose CPU and MJX models can be made to disagree, to test the lock check."""

    def __init__(self, model: mujoco.MjModel, simulated: np.ndarray | None = None) -> None:
        self.mj_model = model
        self.mjx_model = mock.Mock(
            jnt_stiffness=(
                np.asarray(model.jnt_stiffness) if simulated is None else np.asarray(simulated)
            )
        )


LOCKABLE_XML = """
<mujoco model="lockable">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="elbow_joint" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="elbow_joint" joint="elbow_joint" kp="10"/>
  </actuator>
</mujoco>
"""


class LockReachedEnvTest(unittest.TestCase):
    """A lock that stopped at the CPU model would be silently inert."""

    def setUp(self):
        from generalist_robotics.morphology.topology import lock_joints

        self.base = mujoco.MjModel.from_xml_string(LOCKABLE_XML)
        self.locks = (DofLock(("elbow_joint",), 1.0),)
        self.locked = lock_joints(self.base, self.locks)

    def test_a_locked_model_passes(self):
        dof_path.check_locks_reached_env(FakeLockedEnv(self.locked), self.locks)

    def test_an_mjx_model_that_did_not_inherit_the_lock_is_caught(self):
        env = FakeLockedEnv(self.locked, simulated=np.zeros_like(self.locked.jnt_stiffness))
        with self.assertRaises(RuntimeError):
            dof_path.check_locks_reached_env(env, self.locks)

    def test_a_model_that_was_never_locked_is_caught(self):
        with self.assertRaises(RuntimeError):
            dof_path.check_locks_reached_env(FakeLockedEnv(self.base), self.locks)


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the real walk")
class PlaygroundIntegrationTest(unittest.TestCase):
    """One real waypoint on the real superset robot, locks and all.

    A smoke-sized PPO round cannot make a random policy walk, so the single step is
    expected to be rejected; what is under test is that every seam holds - the lock
    reaches the simulated model, the morph composes with it, the rollout harness reads the
    right sensor, and the artifacts name the body they belong to.
    """

    def test_walks_one_real_step_and_leaves_replayable_artifacts(self):
        from generalist_robotics.morphology.topology import joint_group

        locked_joints = joint_group(ROBOT, "waist") + joint_group(ROBOT, "arms")
        start_locks = (DofLock(locked_joints, 1.0),)
        end_locks = (DofLock(locked_joints, 0.0),)
        overrides = dict(ppo.SMOKE_PPO_OVERRIDES)

        with dof_path.locked_compilation(start_locks, dof_path.base_sim_dt(ROBOT)):
            cold = ppo.train_policy(
                ROBOT,
                num_timesteps=None,
                ppo_overrides=overrides,
                config_overrides={"episode_length": INTEGRATION_EPISODE_LENGTH},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            result = dof_path.walk_dof_path(
                robot=ROBOT,
                start_params=MorphParams(size_scale=0.8, mass_scale=0.512, torque_scale=0.4096),
                end_params=MorphParams(),
                start_locks=start_locks,
                end_locks=end_locks,
                init_policy_params=cold.params,
                step_alpha=1.0,
                min_step_alpha=1.0,
                finetune_timesteps=overrides["num_timesteps"],
                max_finetune_rounds=1,
                num_eval_episodes=1,
                log_path=root / "run.jsonl",
                checkpoint_dir=root,
                ppo_overrides=overrides,
                scale_time=True,
                scale_task=True,
                config_overrides={"episode_length": INTEGRATION_EPISODE_LENGTH},
            )
            records = [json.loads(line) for line in (root / "run.jsonl").read_text().splitlines()]
            morphologies = sorted(root.glob(f"*/{dof_path.WAYPOINT_MORPHOLOGY_FILENAME}"))

        self.assertEqual(len(result.waypoints), 2)
        self.assertEqual([w.active_dof for w in result.waypoints], [12, 29])
        self.assertEqual(len(records), 3)
        self.assertEqual(len(morphologies), 2)
