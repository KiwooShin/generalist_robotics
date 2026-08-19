"""Unit tests for the policy rollout and viability harness."""

import math
import os
import unittest
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from generalist_robotics.evaluation import rollout

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"


class FakeData(NamedTuple):
    """MJX-data stand-in: free-joint base pose, world velocity and episode bookkeeping."""

    qpos: jax.Array
    qvel: jax.Array
    speed: jax.Array
    step_index: jax.Array
    fall_step: jax.Array


class FakeState(NamedTuple):
    """Brax-style state carrying the fields the harness reads."""

    data: FakeData
    obs: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array


class FakeLocomotionEnv:
    """Deterministic locomotion stand-in with a controllable termination step.

    The robot walks forward at forward_speed until step fall_step, which is the
    last live step and the one reporting done. Every later step reports garbage
    (post_fall_speed, post_fall_reward) so that any statistic which fails to mask
    post-termination data is visibly wrong.

    Attributes:
        heading_quat: base orientation, used to place the forward axis away from world x.
    """

    def __init__(
        self,
        action_size: int = 4,
        dt: float = 0.1,
        forward_speed: float = 0.5,
        reward_per_step: float = 1.0,
        fall_step: int | None = None,
        random_fall_horizon: int | None = None,
        done_persists: bool = True,
        post_fall_speed: float = 100.0,
        post_fall_reward: float = 100.0,
        provide_local_linvel: bool = True,
        heading_quat: Any = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.dt = dt
        self.action_size = action_size
        self.observation_size = {"state": (3,)}
        self.forward_speed = forward_speed
        self.reward_per_step = reward_per_step
        self.fall_step = fall_step
        self.random_fall_horizon = random_fall_horizon
        self.done_persists = done_persists
        self.post_fall_speed = post_fall_speed
        self.post_fall_reward = post_fall_reward
        self.heading_quat = jnp.array(heading_quat, dtype=jnp.float32)
        if provide_local_linvel:
            self.get_local_linvel = self.local_linvel

    def local_linvel(self, data: FakeData) -> jax.Array:
        """Base velocity in the body frame, as the playground envs expose it."""
        return jnp.array([data.speed, 0.0, 0.0])

    def world_velocity(self, speed: jax.Array) -> jax.Array:
        """Body-forward speed expressed in the world frame."""
        return rollout.rotate_by_quat(self.heading_quat, jnp.array([speed, 0.0, 0.0]))

    def draw_fall_step(self, rng: jax.Array) -> jax.Array:
        """Step at which this episode terminates, possibly drawn from the rng."""
        if self.random_fall_horizon is not None:
            return jax.random.randint(rng, (), 1, self.random_fall_horizon + 1).astype(jnp.float32)
        if self.fall_step is None:
            return jnp.array(jnp.inf, dtype=jnp.float32)
        return jnp.array(float(self.fall_step), dtype=jnp.float32)

    def make_state(
        self,
        position: jax.Array,
        speed: jax.Array,
        step_index: jax.Array,
        fall_step: jax.Array,
        reward: jax.Array,
        done: jax.Array,
    ) -> FakeState:
        """Assemble a state with the fixed pytree structure the scan carry needs."""
        qpos = jnp.concatenate([position, self.heading_quat])
        qvel = jnp.concatenate([self.world_velocity(speed), jnp.zeros((3,))])
        data = FakeData(
            qpos=qpos,
            qvel=qvel,
            speed=speed,
            step_index=step_index,
            fall_step=fall_step,
        )
        obs = {"state": jnp.array([qpos[0], speed, step_index])}
        return FakeState(data=data, obs=obs, reward=reward, done=done)

    def reset(self, rng: jax.Array) -> FakeState:
        """Reset to the origin, standing still."""
        return self.make_state(
            position=jnp.zeros((3,)),
            speed=jnp.array(0.0, dtype=jnp.float32),
            step_index=jnp.array(0.0, dtype=jnp.float32),
            fall_step=self.draw_fall_step(rng),
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
        )

    def step(self, state: FakeState, action: jax.Array) -> FakeState:
        """Advance one control step; reward is penalised by the action magnitude."""
        step_index = state.data.step_index + 1.0
        fall_step = state.data.fall_step
        fallen = step_index > fall_step
        speed = jnp.where(fallen, self.post_fall_speed, self.forward_speed)
        reward = jnp.where(
            fallen,
            self.post_fall_reward,
            self.reward_per_step - jnp.sum(action**2),
        ).astype(jnp.float32)
        done = step_index >= fall_step if self.done_persists else step_index == fall_step
        position = state.data.qpos[:3] + self.world_velocity(speed) * self.dt
        return self.make_state(
            position=position,
            speed=speed.astype(jnp.float32),
            step_index=step_index,
            fall_step=fall_step,
            reward=reward,
            done=done.astype(jnp.float32),
        )


class ConfiguredEnv(FakeLocomotionEnv):
    """Fake env that carries a playground-style locked config with an episode length."""

    def __init__(self, episode_length: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = type("Config", (), {"episode_length": episode_length})()


def constant_policy(action_size: int, value: float) -> rollout.Policy:
    """Policy emitting the same constant action every step."""
    action = jnp.full((action_size,), value)

    def policy(obs: dict[str, jax.Array]) -> jax.Array:
        del obs
        return action

    return policy


class ZeroPolicyTest(unittest.TestCase):
    """The zero-action baseline policy."""

    def test_emits_zeros_of_the_requested_size(self):
        action = rollout.zero_policy(12)({"state": jnp.zeros((3,))})
        self.assertEqual(action.shape, (12,))
        self.assertTrue(bool(jnp.all(action == 0.0)))


class ForwardVelocityTest(unittest.TestCase):
    """Reading the forward axis velocity out of a state."""

    def test_prefers_the_env_local_velocity_sensor(self):
        env = FakeLocomotionEnv(forward_speed=0.75)
        state = env.step(env.reset(jax.random.PRNGKey(0)), jnp.zeros((4,)))
        self.assertAlmostEqual(float(rollout.forward_velocity(env, state.data)), 0.75, places=5)

    def test_falls_back_to_the_free_joint_when_no_sensor(self):
        # 90 degrees about z: the body forward axis points along world +y.
        quat = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        env = FakeLocomotionEnv(forward_speed=0.75, provide_local_linvel=False, heading_quat=quat)
        state = env.step(env.reset(jax.random.PRNGKey(0)), jnp.zeros((4,)))
        self.assertFalse(hasattr(env, "get_local_linvel"))
        self.assertAlmostEqual(float(state.data.qvel[1]), 0.75, places=5)
        self.assertAlmostEqual(float(rollout.forward_velocity(env, state.data)), 0.75, places=5)


class EpisodeLengthTest(unittest.TestCase):
    """Choosing the rollout horizon."""

    def test_reads_the_env_config_when_present(self):
        self.assertEqual(rollout.episode_length_for(ConfiguredEnv(37)), 37)

    def test_falls_back_to_the_locomotion_default(self):
        self.assertEqual(
            rollout.episode_length_for(FakeLocomotionEnv()),
            rollout.DEFAULT_EPISODE_LENGTH,
        )

    def test_explicit_length_overrides_the_env_config(self):
        env = ConfiguredEnv(1000, forward_speed=1.0)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=1, episode_length=5
        )
        self.assertEqual(stats.num_steps, 5)


class EvaluatePolicyTest(unittest.TestCase):
    """Statistics aggregated over episodes of the fake environment."""

    def test_full_episode_without_termination(self):
        env = FakeLocomotionEnv(forward_speed=0.5, reward_per_step=1.0)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=3, episode_length=20
        )
        self.assertAlmostEqual(stats.survived_fraction, 1.0, places=6)
        self.assertAlmostEqual(stats.mean_forward_velocity, 0.5, places=5)
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 20, places=5)
        self.assertAlmostEqual(stats.episode_return, 20.0, places=4)
        self.assertEqual(stats.num_episodes, 3)
        self.assertEqual(stats.num_steps, 60)

    def test_statistics_stop_at_termination(self):
        env = FakeLocomotionEnv(forward_speed=0.5, reward_per_step=1.0, fall_step=5)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=2, episode_length=20
        )
        self.assertAlmostEqual(stats.survived_fraction, 0.25, places=6)
        self.assertAlmostEqual(stats.mean_forward_velocity, 0.5, places=5)
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 5, places=5)
        self.assertAlmostEqual(stats.episode_return, 5.0, places=4)

    def test_flickering_done_is_latched(self):
        # The real playground env recomputes done from the current state, so it can
        # drop back to zero after a fall; survival must still latch at the fall.
        env = FakeLocomotionEnv(
            forward_speed=0.5, reward_per_step=1.0, fall_step=5, done_persists=False
        )
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=2, episode_length=20
        )
        self.assertAlmostEqual(stats.survived_fraction, 0.25, places=6)
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 5, places=5)
        self.assertAlmostEqual(stats.episode_return, 5.0, places=4)

    def test_non_finite_values_after_the_fall_do_not_pollute(self):
        env = FakeLocomotionEnv(
            forward_speed=0.5,
            reward_per_step=1.0,
            fall_step=5,
            post_fall_speed=float("nan"),
            post_fall_reward=float("inf"),
        )
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=2, episode_length=20
        )
        self.assertTrue(math.isfinite(stats.distance_travelled))
        self.assertTrue(math.isfinite(stats.episode_return))
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 5, places=5)
        self.assertAlmostEqual(stats.episode_return, 5.0, places=4)

    def test_policy_actions_reach_the_environment(self):
        env = FakeLocomotionEnv(action_size=4, reward_per_step=1.0)
        zero_stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=1, episode_length=10
        )
        # Reward loses sum(action**2) = 4 * 0.5**2 = 1 per step.
        half_stats = rollout.evaluate_policy(
            env, constant_policy(4, 0.5), num_episodes=1, episode_length=10
        )
        self.assertAlmostEqual(zero_stats.episode_return, 10.0, places=4)
        self.assertAlmostEqual(half_stats.episode_return, 0.0, places=4)

    def test_survival_averages_over_episodes_with_different_fall_steps(self):
        horizon = 20
        env = FakeLocomotionEnv(forward_speed=0.5, random_fall_horizon=horizon)
        num_episodes = 6
        stats = rollout.evaluate_policy(
            env,
            rollout.zero_policy(4),
            num_episodes=num_episodes,
            seed=3,
            episode_length=horizon,
        )
        rngs = jax.random.split(jax.random.PRNGKey(3), num_episodes)
        fall_steps = jnp.stack([env.draw_fall_step(rng) for rng in rngs])
        expected = float(jnp.mean(jnp.minimum(fall_steps, horizon)) / horizon)
        self.assertAlmostEqual(stats.survived_fraction, expected, places=6)
        self.assertAlmostEqual(
            stats.distance_travelled,
            float(jnp.mean(jnp.minimum(fall_steps, horizon))) * 0.5 * 0.1,
            places=5,
        )
        self.assertAlmostEqual(stats.mean_forward_velocity, 0.5, places=5)

    def test_rejects_invalid_arguments(self):
        env = FakeLocomotionEnv()
        with self.assertRaises(ValueError):
            rollout.evaluate_policy(env, rollout.zero_policy(4), num_episodes=0)
        with self.assertRaises(ValueError):
            rollout.evaluate_policy(env, rollout.zero_policy(4), num_episodes=1, episode_length=0)


class DeterminismTest(unittest.TestCase):
    """Seeding through jax.random."""

    def test_same_seed_reproduces_stats(self):
        env = FakeLocomotionEnv(forward_speed=0.5, random_fall_horizon=20)
        first = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=4, seed=7, episode_length=20
        )
        second = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=4, seed=7, episode_length=20
        )
        self.assertEqual(first, second)

    def test_different_seeds_change_stats(self):
        env = FakeLocomotionEnv(forward_speed=0.5, random_fall_horizon=20)
        first = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=4, seed=0, episode_length=20
        )
        second = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=4, seed=1, episode_length=20
        )
        self.assertNotEqual(first.survived_fraction, second.survived_fraction)


class ViabilityTest(unittest.TestCase):
    """The viability predicate over aggregate stats."""

    def make_stats(self, survived: float, velocity: float) -> rollout.RolloutStats:
        """Stats with only the two fields the predicate reads set meaningfully."""
        return rollout.RolloutStats(
            survived_fraction=survived,
            mean_forward_velocity=velocity,
            distance_travelled=velocity * survived,
            episode_return=0.0,
            num_episodes=1,
            num_steps=1,
        )

    def test_viable_when_both_thresholds_are_met(self):
        viable = rollout.is_viable(self.make_stats(0.95, 0.4))
        self.assertTrue(viable)
        self.assertIsInstance(viable, bool)

    def test_not_viable_when_it_falls_over(self):
        self.assertFalse(rollout.is_viable(self.make_stats(0.5, 0.4)))

    def test_not_viable_when_it_stands_still(self):
        self.assertFalse(rollout.is_viable(self.make_stats(1.0, 0.05), min_forward_velocity=0.2))

    def test_thresholds_are_inclusive(self):
        self.assertTrue(
            rollout.is_viable(
                self.make_stats(0.8, 0.2),
                min_survived_fraction=0.8,
                min_forward_velocity=0.2,
            )
        )


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the playground rollout")
class PlaygroundIntegrationTest(unittest.TestCase):
    """One short rollout against a real MuJoCo Playground locomotion env."""

    def test_zero_policy_rollout_produces_finite_stats(self):
        from mujoco_playground import registry

        env = registry.load("BerkeleyHumanoidJoystickFlatTerrain")
        stats = rollout.evaluate_policy(
            env,
            rollout.zero_policy(env.action_size),
            num_episodes=2,
            seed=0,
            episode_length=15,
        )
        self.assertEqual(stats.num_episodes, 2)
        self.assertEqual(stats.num_steps, 30)
        self.assertGreater(stats.survived_fraction, 0.0)
        self.assertLessEqual(stats.survived_fraction, 1.0)
        self.assertTrue(math.isfinite(stats.mean_forward_velocity))
        self.assertTrue(math.isfinite(stats.distance_travelled))
        self.assertTrue(math.isfinite(stats.episode_return))
        self.assertIsInstance(rollout.is_viable(stats), bool)


if __name__ == "__main__":
    unittest.main()
