"""Unit tests for PPO training, checkpointing and warm-started fine-tuning."""

import json
import os
import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from generalist_robotics.envs.locomotion import make_locomotion_env
from generalist_robotics.evaluation import rollout
from generalist_robotics.morphology.scaling import MorphParams
from generalist_robotics.training import ppo

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

ROBOT = "berkeley_humanoid"

# Rollout budget for the seam test: enough steps to exercise reset, step and termination
# without paying for Playground's 1000-step episode.
TEST_EPISODE_LENGTH = 50
TEST_EPISODES = 2


def largest_difference(left: object, right: object) -> float:
    """Largest absolute elementwise difference between two parameter trees."""
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
        )
    )


class FloatMetricsTest(unittest.TestCase):
    """float_metrics keeps the scalars a TrainingResult can carry."""

    def test_keeps_scalars_and_drops_the_rest(self):
        metrics = {
            "eval/episode_reward": jnp.array(1.5),
            "training/sps": 2.0,
            "eval/curve": jnp.arange(3.0),
        }
        self.assertEqual(
            ppo.float_metrics(metrics), {"eval/episode_reward": 1.5, "training/sps": 2.0}
        )

    def test_values_are_python_floats(self):
        values = ppo.float_metrics({"a": jnp.array(3.0)})
        self.assertIsInstance(values["a"], float)


class ProgressLogTest(unittest.TestCase):
    """ProgressLog is brax's progress_fn and the record of what a run really cost."""

    def test_records_in_memory_without_a_path(self):
        log = ppo.ProgressLog()
        log.record(0, {ppo.EVAL_REWARD_KEY: jnp.array(0.5)})
        log.record(1024, {ppo.EVAL_REWARD_KEY: jnp.array(2.5)})
        self.assertEqual(log.last_step(), 1024)
        self.assertAlmostEqual(log.initial_reward(), 0.5)

    def test_streams_one_json_line_per_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "nested" / "progress.jsonl"
            log = ppo.ProgressLog(path)
            log.record(0, {ppo.EVAL_REWARD_KEY: 0.5})
            log.record(512, {ppo.EVAL_REWARD_KEY: 1.5})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["num_steps"] for row in rows], [0.0, 512.0])
        self.assertEqual(rows[1][ppo.EVAL_REWARD_KEY], 1.5)

    def test_empty_log_reports_no_steps_and_no_initial_reward(self):
        log = ppo.ProgressLog()
        self.assertEqual(log.last_step(), 0)
        self.assertIsNone(log.initial_reward())


class PpoConfigTest(unittest.TestCase):
    """ppo_config hands over Playground's tuned hyperparameters, not guesses."""

    def test_uses_the_tuned_values(self):
        config = ppo.ppo_config(ROBOT)
        self.assertEqual(config["num_timesteps"], 150_000_000)
        self.assertEqual(config["num_envs"], 8192)
        self.assertEqual(config["batch_size"], 256)

    def test_network_factory_becomes_a_callable(self):
        factory = ppo.ppo_config(ROBOT)["network_factory"]
        self.assertTrue(callable(factory))
        # The Berkeley Humanoid critic reads the privileged observation, not the actor's.
        self.assertEqual(factory.keywords["value_obs_key"], "privileged_state")
        self.assertEqual(factory.keywords["policy_obs_key"], "state")

    def test_timesteps_and_overrides_are_applied(self):
        config = ppo.ppo_config(ROBOT, num_timesteps=1234, overrides={"num_envs": 8})
        self.assertEqual(config["num_timesteps"], 1234)
        self.assertEqual(config["num_envs"], 8)

    def test_unknown_robot_is_rejected(self):
        with self.assertRaises(ValueError):
            ppo.ppo_config("not_a_robot")


class DevicePutReplicatedTest(unittest.TestCase):
    """The compatibility shim brax needs on a JAX that dropped device_put_replicated."""

    def test_adds_a_leading_device_axis(self):
        devices = jax.local_devices()[:1]
        replicated = ppo.device_put_replicated({"a": jnp.arange(3.0)}, devices)
        self.assertEqual(replicated["a"].shape, (1, 3))

    def test_result_is_consumable_by_pmap(self):
        devices = jax.local_devices()[:1]
        replicated = ppo.device_put_replicated(jnp.ones((2,)), devices)
        summed = jax.pmap(lambda x: jax.lax.psum(jnp.sum(x), "i"), axis_name="i")(replicated)
        self.assertAlmostEqual(float(summed[0]), 2.0)

    def test_restoring_is_idempotent_and_leaves_the_name_callable(self):
        ppo.restore_replicated_device_put()
        first = getattr(jax, ppo.REPLICATED_DEVICE_PUT_NAME)
        ppo.restore_replicated_device_put()
        self.assertIs(getattr(jax, ppo.REPLICATED_DEVICE_PUT_NAME), first)


class StochasticActionKeyTest(unittest.TestCase):
    """A sampling policy gets its randomness from the observation it is handed."""

    def test_distinct_observations_give_distinct_keys(self):
        base = jax.random.PRNGKey(0)
        left = ppo.stochastic_action_key(base, {"state": jnp.arange(4.0)})
        right = ppo.stochastic_action_key(base, {"state": jnp.arange(4.0) + 1.0})
        self.assertFalse(bool(jnp.array_equal(left, right)))

    def test_equal_observations_give_equal_keys(self):
        base = jax.random.PRNGKey(0)
        left = ppo.stochastic_action_key(base, {"state": jnp.ones(4)})
        right = ppo.stochastic_action_key(base, {"state": jnp.ones(4)})
        self.assertTrue(bool(jnp.array_equal(left, right)))


class MissingCheckpointTest(unittest.TestCase):
    """Loading a checkpoint that is not there fails loudly."""

    def test_absent_directory_raises(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            ppo.load_checkpoint(pathlib.Path(directory) / "nothing_here")


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the PPO smoke runs")
class WarmStartTest(unittest.TestCase):
    """init_params must reach brax's training state, not be silently dropped.

    A zero-step run is the sharpest possible probe: brax builds its training state,
    restores the given parameters into it and returns without a single gradient step, so
    the restored parameters must come back bit for bit while a cold run of the same seed
    comes back with its random initialisation. The source parameters travel through a
    checkpoint on the way, which is the continuation path's actual mechanism.
    """

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        overrides = dict(ppo.SMOKE_PPO_OVERRIDES, num_timesteps=0)
        source = ppo.train_policy(ROBOT, seed=7, ppo_overrides=overrides)
        path = ppo.save_checkpoint(source, pathlib.Path(cls.directory.name) / "source")
        cls.source_params, _ = ppo.load_checkpoint(path)
        cls.warm = ppo.train_policy(
            ROBOT, seed=11, init_params=cls.source_params, ppo_overrides=overrides
        )
        cls.cold = ppo.train_policy(ROBOT, seed=11, ppo_overrides=overrides)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_warm_started_parameters_are_the_restored_ones(self):
        self.assertEqual(largest_difference(self.source_params, self.warm.params), 0.0)

    def test_cold_start_of_the_same_seed_differs_from_the_restored_policy(self):
        self.assertGreater(largest_difference(self.source_params, self.cold.params), 0.0)


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the PPO smoke runs")
class TrainedPolicyTest(unittest.TestCase):
    """One tiny real run, then everything the rest of the project asks of its output."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.directory.name)
        cls.progress_path = root / "progress.jsonl"
        cls.result = ppo.train_policy(
            ROBOT,
            params=MorphParams(size_scale=1.1, mass_scale=1.2, torque_scale=1.3),
            ppo_overrides=ppo.SMOKE_PPO_OVERRIDES,
            progress_path=cls.progress_path,
        )
        cls.checkpoint = ppo.save_checkpoint(cls.result, root / "checkpoint")
        cls.loaded, cls.loaded_metrics = ppo.load_checkpoint(cls.checkpoint)
        cls.env = make_locomotion_env(ROBOT)
        cls.observation = cls.env.reset(jax.random.PRNGKey(0)).obs

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_run_reports_its_own_cost(self):
        self.assertGreaterEqual(self.result.num_timesteps, ppo.SMOKE_PPO_OVERRIDES["num_timesteps"])
        self.assertGreater(self.result.wall_clock_seconds, 0.0)
        self.assertAlmostEqual(
            self.result.steps_per_second,
            self.result.num_timesteps / self.result.wall_clock_seconds,
            places=3,
        )

    def test_metrics_carry_both_evaluations(self):
        self.assertIn(ppo.EVAL_REWARD_KEY, self.result.metrics)
        self.assertIn(ppo.INITIAL_EVAL_REWARD_KEY, self.result.metrics)

    def test_progress_file_holds_one_row_per_evaluation(self):
        rows = [json.loads(line) for line in self.progress_path.read_text().splitlines()]
        self.assertEqual(len(rows), ppo.SMOKE_PPO_OVERRIDES["num_evals"])
        self.assertEqual(rows[0]["num_steps"], 0.0)

    def test_checkpoint_round_trips_to_identical_actions(self):
        before = ppo.make_policy(ROBOT, self.result.params)(self.observation)
        after = ppo.make_policy(ROBOT, self.loaded)(self.observation)
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_checkpoint_carries_the_metrics(self):
        self.assertEqual(self.loaded_metrics, self.result.metrics)

    def test_policy_is_consumed_by_the_rollout_harness(self):
        stats = rollout.evaluate_policy(
            self.env,
            ppo.make_policy(ROBOT, self.loaded),
            num_episodes=TEST_EPISODES,
            seed=0,
            episode_length=TEST_EPISODE_LENGTH,
        )
        self.assertIsInstance(stats, rollout.RolloutStats)
        self.assertEqual(stats.num_steps, TEST_EPISODES * TEST_EPISODE_LENGTH)
        self.assertGreater(stats.nominal_leg_length, 0.0)

    def test_deterministic_policy_repeats_itself(self):
        policy = ppo.make_policy(ROBOT, self.loaded)
        np.testing.assert_array_equal(
            np.asarray(policy(self.observation)), np.asarray(policy(self.observation))
        )

    def test_stochastic_policy_differs_from_the_mode(self):
        mode = np.asarray(ppo.make_policy(ROBOT, self.loaded)(self.observation))
        sample = np.asarray(
            ppo.make_policy(ROBOT, self.loaded, deterministic=False)(self.observation)
        )
        self.assertEqual(sample.shape, (self.env.action_size,))
        self.assertTrue(np.all(np.isfinite(sample)))
        self.assertGreater(float(np.max(np.abs(sample - mode))), 0.0)


if __name__ == "__main__":
    unittest.main()
