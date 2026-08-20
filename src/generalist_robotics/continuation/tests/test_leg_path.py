"""Unit tests for the leg-growth continuation path and its multiped environment."""

import dataclasses
import json
import os
import pathlib
import tempfile
import unittest

import numpy as np

from generalist_robotics.analysis.gait import GaitSignature, gait_signature
from generalist_robotics.continuation import leg_path
from generalist_robotics.evaluation.rollout import RolloutStats
from generalist_robotics.morphology.multiped import SIM_TIMESTEP, LegGrowth, MultipedSpec
from generalist_robotics.training import ppo

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

SPEC = MultipedSpec(n_legs=4)
BIPED = (LegGrowth(2, 0.0), LegGrowth(3, 0.0))
TRIPOD = (LegGrowth(2, 1.0), LegGrowth(3, 0.0))
# What a walk actually records: every leg named, which is what keeps the lock's spring
# normalisation from being renormalised each time a leg finishes growing.
WALKED_BIPED = (LegGrowth(0, 1.0), LegGrowth(1, 1.0), LegGrowth(2, 0.0), LegGrowth(3, 0.0))


def stats(survived: float, speed: float) -> RolloutStats:
    """A RolloutStats carrying only the two fields viability is judged on."""
    return RolloutStats(
        survived_fraction=survived,
        mean_forward_velocity=speed,
        mean_forward_speed=speed,
        distance_travelled=speed * 10.0,
        net_displacement=speed * 10.0,
        episode_return=1.0,
        nominal_leg_length=0.5127,
        num_episodes=8,
        num_steps=4000,
    )


def waypoint(
    alpha: float,
    growth: tuple[LegGrowth, ...] = WALKED_BIPED,
    viable_before: bool = True,
    signature: GaitSignature | None = None,
) -> leg_path.LegWaypoint:
    """A LegWaypoint with plausible contents, for the bookkeeping tests."""
    return leg_path.LegWaypoint(
        alpha=alpha,
        growth=growth,
        total_mass=11.0,
        stats_before=stats(1.0 if viable_before else 0.1, 0.5),
        viable_before=viable_before,
        stats_after=None,
        finetune_steps=0,
        cumulative_steps=0,
        signature=signature,
        contacts=np.zeros((4, 4)),
    )


def alternating_signature() -> GaitSignature:
    """The gait signature of a clean two-legged alternation."""
    time = np.arange(160)[:, None] / 20.0
    contacts = np.mod(time + np.array([[0.0, 0.5, 0.0, 0.0]]), 1.0) < 0.5
    contacts[:, 2:] = False
    return gait_signature(contacts.astype(float), leg_path.CTRL_TIMESTEP)


class ConfigTest(unittest.TestCase):
    """The task and PPO configurations are the ones the body and brax require."""

    def test_simulation_timestep_matches_the_generated_model(self):
        self.assertEqual(float(leg_path.default_multiped_config().sim_dt), SIM_TIMESTEP)

    def test_observation_width_counts_the_supersets_legs(self):
        self.assertEqual(leg_path.observation_width(MultipedSpec(n_legs=4)), 46)
        self.assertEqual(leg_path.observation_width(MultipedSpec(n_legs=2)), 28)

    def test_batch_shape_satisfies_brax(self):
        config = leg_path.multiped_ppo_config(SPEC)
        self.assertEqual((config["batch_size"] * config["num_minibatches"]) % config["num_envs"], 0)
        self.assertEqual(config["episode_length"], 500)

    def test_overrides_win(self):
        config = leg_path.multiped_ppo_config(SPEC, 1234, {"num_envs": 8})
        self.assertEqual(config["num_timesteps"], 1234)
        self.assertEqual(config["num_envs"], 8)

    def test_smoke_overrides_are_usable_on_this_body(self):
        config = leg_path.multiped_ppo_config(SPEC, overrides=ppo.SMOKE_PPO_OVERRIDES)
        self.assertEqual((config["batch_size"] * config["num_minibatches"]) % config["num_envs"], 0)


class InterpolateGrowthTest(unittest.TestCase):
    """The path coordinate moves every leg linearly between its two endpoints."""

    def test_endpoints_are_reproduced(self):
        start = leg_path.interpolate_growth(BIPED, TRIPOD, 0.0, SPEC)
        end = leg_path.interpolate_growth(BIPED, TRIPOD, 1.0, SPEC)
        self.assertEqual([entry.growth for entry in start], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual([entry.growth for entry in end], [1.0, 1.0, 1.0, 0.0])

    def test_midpoint_grows_only_the_leg_that_moves(self):
        middle = leg_path.interpolate_growth(BIPED, TRIPOD, 0.25, SPEC)
        self.assertEqual([entry.growth for entry in middle], [1.0, 1.0, 0.25, 0.0])

    def test_every_leg_is_named_so_the_lock_scale_never_jumps(self):
        middle = leg_path.interpolate_growth(BIPED, TRIPOD, 0.5, SPEC)
        self.assertEqual([entry.leg_index for entry in middle], [0, 1, 2, 3])

    def test_rejects_alphas_off_the_path(self):
        for alpha in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                leg_path.interpolate_growth(BIPED, TRIPOD, alpha, SPEC)


class WaypointBookkeepingTest(unittest.TestCase):
    """Waypoints are accepted, recorded and reduced to a gait path the way M2's are."""

    def test_a_waypoint_is_accepted_when_it_walks_before_or_after_training(self):
        self.assertTrue(leg_path.leg_waypoint_accepted(waypoint(0.5)))
        self.assertFalse(leg_path.leg_waypoint_accepted(waypoint(0.5, viable_before=False)))
        trained = dataclasses.replace(
            waypoint(0.5, viable_before=False), stats_after=stats(1.0, 0.5)
        )
        self.assertTrue(leg_path.leg_waypoint_accepted(trained))

    def test_records_are_json_serialisable(self):
        record = leg_path.leg_waypoint_record(SPEC, waypoint(0.3, signature=None), 2)
        json.dumps(record)
        self.assertEqual(record["index"], 2)
        self.assertEqual(record["alpha"], 0.3)
        self.assertEqual(record["spec"]["n_legs"], 4)
        self.assertEqual(len(record["growth"]), 4)
        self.assertEqual(record["params"]["size_scale"], 1.0)

    def test_a_gait_is_recorded_when_there_is_one(self):
        point = waypoint(0.3, signature=alternating_signature())
        record = leg_path.leg_waypoint_record(SPEC, point, 0)
        json.dumps(record)
        self.assertAlmostEqual(record["gait"]["phase_offsets"][1], 0.5, places=6)

    def test_the_gait_path_drops_waypoints_without_a_gait(self):
        signature = alternating_signature()
        result = leg_path.LegContinuationResult(
            waypoints=[
                waypoint(0.0, signature=signature),
                waypoint(0.5, signature=None),
                waypoint(0.5, viable_before=False, signature=signature),
                waypoint(1.0, signature=signature),
            ],
            reached_target=True,
            total_finetune_steps=0,
            wall_clock_seconds=1.0,
            final_policy_params=None,
        )
        signatures, alphas = leg_path.accepted_gait_path(result)
        self.assertEqual(alphas, [0.0, 1.0])
        self.assertEqual(len(signatures), 2)

    def test_run_log_round_trips(self):
        result = leg_path.LegContinuationResult(
            waypoints=[waypoint(0.0), waypoint(1.0)],
            reached_target=True,
            total_finetune_steps=42,
            wall_clock_seconds=1.5,
            final_policy_params=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = leg_path.save_leg_run_log(result, SPEC, pathlib.Path(directory) / "run.json")
            document = json.loads(path.read_text())
        self.assertEqual(document["num_waypoints"], 2)
        self.assertEqual(document["total_finetune_steps"], 42)
        self.assertEqual(document["spec"]["n_legs"], 4)


class WalkArgumentTest(unittest.TestCase):
    """The walk refuses a step schedule it cannot follow, before touching the GPU."""

    def test_rejects_unusable_schedules(self):
        for kwargs in (
            {"step_alpha": 0.0},
            {"step_alpha": 1.5},
            {"min_step_alpha": 0.5, "step_alpha": 0.1},
            {"step_growth": 0.5},
            {"max_finetune_rounds": -1},
            {"finetune_timesteps": 0},
        ):
            with self.assertRaises(ValueError):
                leg_path.walk_leg_path(SPEC, BIPED, TRIPOD, None, **kwargs)


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables building MJX bodies")
class MultipedEnvironmentTest(unittest.TestCase):
    """The generated environment simulates, and an ungrown leg is inert inside it."""

    @classmethod
    def setUpClass(cls):
        cls.env = leg_path.make_multiped_env(SPEC, BIPED)

    def test_observation_and_action_widths_are_the_supersets(self):
        self.assertEqual(self.env.observation_size[leg_path.OBSERVATION_KEY][-1], 46)
        self.assertEqual(self.env.action_size, 12)

    def test_reset_stands_the_grown_feet_on_the_floor_and_the_stubs_off_it(self):
        import jax

        state = self.env.reset(jax.random.PRNGKey(0))
        contacts = np.asarray(self.env.foot_contacts(state.data))
        np.testing.assert_array_equal(contacts, [True, True, False, False])

    def test_a_step_stays_finite(self):
        import jax
        import jax.numpy as jnp

        state = self.env.step(self.env.reset(jax.random.PRNGKey(0)), jnp.zeros(12))
        self.assertTrue(bool(np.all(np.isfinite(np.asarray(state.data.qpos)))))
        self.assertEqual(float(state.done), 0.0)

    def test_the_growth_state_reaches_the_simulated_model(self):
        np.testing.assert_allclose(
            np.asarray(self.env.mjx_model.body_mass), np.asarray(self.env.mj_model.body_mass)
        )
        np.testing.assert_allclose(
            np.asarray(self.env.mjx_model.jnt_stiffness),
            np.asarray(self.env.mj_model.jnt_stiffness),
        )

    def test_a_fixed_simulation_step_is_required(self):
        with self.assertRaises(ValueError):
            leg_path.make_multiped_env(SPEC, BIPED, config_overrides={"sim_dt": 0.002})


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the PPO smoke run")
class MultipedTrainingTest(unittest.TestCase):
    """A test-sized PPO run trains, checkpoints and hands a policy back to the harness."""

    def test_smoke_run_produces_a_usable_policy(self):
        result = leg_path.train_multiped_policy(
            SPEC, BIPED, num_timesteps=8192, seed=0, ppo_overrides=ppo.SMOKE_PPO_OVERRIDES
        )
        self.assertGreater(result.num_timesteps, 0)
        with tempfile.TemporaryDirectory() as directory:
            saved = ppo.save_checkpoint(result, pathlib.Path(directory) / "checkpoint")
            params, _ = ppo.load_checkpoint(saved)
        policy = leg_path.make_multiped_policy(SPEC, params)
        env = leg_path.make_multiped_env(SPEC, BIPED)
        signature, trace = leg_path.waypoint_gait(env, policy, seed=0)
        self.assertEqual(trace.ndim, 2)
        self.assertEqual(trace.shape[1], SPEC.n_legs)
        if signature is not None:
            self.assertEqual(signature.num_legs, SPEC.n_legs)


if __name__ == "__main__":
    unittest.main()
