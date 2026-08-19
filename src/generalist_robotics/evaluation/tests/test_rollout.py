"""Unit tests for the policy rollout and viability harness."""

import math
import os
import unittest
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco

from generalist_robotics.evaluation import rollout

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

# A free-joint body standing at z = 0.4 whose home keyframe crouches to 0.35, with a
# decoy keyframe first so that preferring the one named "home" is actually tested.
KEYFRAME_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.4">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="crouch" qpos="0 0 0.2 1 0 0 0"/>
    <key name="home" qpos="0 0 0.35 1 0 0 0"/>
  </keyframe>
</mujoco>
"""

HINGE_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 0.4">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    """Hamilton product of two wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


class ModelEnv:
    """Env stand-in exposing nothing but a compiled MuJoCo model."""

    def __init__(self, xml: str) -> None:
        self.mj_model = mujoco.MjModel.from_xml_string(xml)


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

    The robot walks along its own forward axis at forward_speed until step fall_step,
    which is the last live step and the one reporting done. Every later step reports
    garbage (post_fall_speed, post_fall_reward) so that any statistic which fails to mask
    post-termination data is visibly wrong. A non-zero turn_rate yaws the heading as it
    goes, which separates distance along the heading from net displacement.

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
        turn_rate: float = 0.0,
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
        self.turn_rate = turn_rate
        if provide_local_linvel:
            self.get_local_linvel = self.local_linvel

    def local_linvel(self, data: FakeData) -> jax.Array:
        """Base velocity in the body frame, as the playground envs expose it."""
        return jnp.array([data.speed, 0.0, 0.0])

    def heading_at(self, step_index: jax.Array) -> jax.Array:
        """Orientation after yawing at turn_rate for step_index control steps."""
        half = 0.5 * self.turn_rate * step_index * self.dt
        spin = jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)])
        return quat_multiply(self.heading_quat, spin)

    def world_velocity(self, quat: jax.Array, speed: jax.Array) -> jax.Array:
        """Body-forward speed expressed in the world frame."""
        return rollout.rotate_by_quat(quat, jnp.array([speed, 0.0, 0.0]))

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
        quat: jax.Array,
        speed: jax.Array,
        step_index: jax.Array,
        fall_step: jax.Array,
        reward: jax.Array,
        done: jax.Array,
    ) -> FakeState:
        """Assemble a state with the fixed pytree structure the scan carry needs."""
        qpos = jnp.concatenate([position, quat])
        qvel = jnp.concatenate([self.world_velocity(quat, speed), jnp.zeros((3,))])
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
        zero = jnp.array(0.0, dtype=jnp.float32)
        return self.make_state(
            position=jnp.zeros((3,)),
            quat=self.heading_at(zero),
            speed=zero,
            step_index=zero,
            fall_step=self.draw_fall_step(rng),
            reward=zero,
            done=zero,
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
        quat = self.heading_at(step_index)
        position = state.data.qpos[:3] + self.world_velocity(quat, speed) * self.dt
        return self.make_state(
            position=position,
            quat=quat,
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


class NominalLegLengthTest(unittest.TestCase):
    """Reading the robot's standing base height off its MuJoCo model."""

    def test_prefers_the_keyframe_named_home(self):
        self.assertAlmostEqual(
            rollout.nominal_leg_length(ModelEnv(KEYFRAME_MODEL_XML)), 0.35, places=6
        )

    def test_falls_back_to_qpos0_without_keyframes(self):
        xml = KEYFRAME_MODEL_XML[: KEYFRAME_MODEL_XML.index("<keyframe>")] + "</mujoco>"
        self.assertAlmostEqual(rollout.nominal_leg_length(ModelEnv(xml)), 0.4, places=6)

    def test_reports_zero_without_a_free_joint(self):
        self.assertEqual(rollout.nominal_leg_length(ModelEnv(HINGE_MODEL_XML)), 0.0)

    def test_reports_zero_for_an_env_with_no_model(self):
        self.assertEqual(rollout.nominal_leg_length(FakeLocomotionEnv()), 0.0)


class FroudeNumberTest(unittest.TestCase):
    """The dimensionless gait speed the viability threshold is expressed in."""

    def test_matches_the_definition(self):
        self.assertAlmostEqual(rollout.froude_number(1.0, 0.5), 1.0 / (9.81 * 0.5), places=9)

    def test_is_invariant_along_the_similarity_family(self):
        # Under dynamic similarity length scales as k and speed as sqrt(k).
        base = rollout.froude_number(0.5, 0.515)
        for scale in (0.25, 4.0):
            scaled = rollout.froude_number(0.5 * math.sqrt(scale), 0.515 * scale)
            self.assertAlmostEqual(scaled, base, places=9)

    def test_rejects_an_unknown_body_scale(self):
        with self.assertRaises(ValueError):
            rollout.froude_number(0.5, 0.0)


class EvaluatePolicyTest(unittest.TestCase):
    """Statistics aggregated over episodes of the fake environment."""

    def test_full_episode_without_termination(self):
        env = FakeLocomotionEnv(forward_speed=0.5, reward_per_step=1.0)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=3, episode_length=20
        )
        self.assertAlmostEqual(stats.survived_fraction, 1.0, places=6)
        self.assertAlmostEqual(stats.mean_forward_velocity, 0.5, places=5)
        self.assertAlmostEqual(stats.mean_forward_speed, 0.5, places=5)
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 20, places=5)
        self.assertAlmostEqual(stats.net_displacement, 0.5 * 0.1 * 20, places=5)
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
        self.assertAlmostEqual(stats.net_displacement, 0.5 * 0.1 * 5, places=5)
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
        self.assertAlmostEqual(stats.net_displacement, 0.5 * 0.1 * 5, places=5)
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
        self.assertTrue(math.isfinite(stats.net_displacement))
        self.assertTrue(math.isfinite(stats.episode_return))
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * 5, places=5)
        self.assertAlmostEqual(stats.net_displacement, 0.5 * 0.1 * 5, places=5)
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


class DistanceSemanticsTest(unittest.TestCase):
    """Distance along the heading and net displacement are different quantities."""

    def test_a_circling_robot_travels_far_and_gets_nowhere(self):
        # A full revolution in 20 steps: 20 equally spaced heading directions sum to zero.
        steps = 20
        env = FakeLocomotionEnv(forward_speed=0.5, dt=0.1, turn_rate=2.0 * math.pi / 2.0)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=1, episode_length=steps
        )
        self.assertAlmostEqual(stats.distance_travelled, 0.5 * 0.1 * steps, places=4)
        self.assertAlmostEqual(stats.mean_forward_speed, 0.5, places=5)
        self.assertLess(stats.net_displacement, 1e-4)

    def test_a_straight_walker_travels_exactly_as_far_as_it_displaces(self):
        env = FakeLocomotionEnv(forward_speed=0.5, dt=0.1)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=1, episode_length=20
        )
        self.assertAlmostEqual(stats.distance_travelled, stats.net_displacement, places=5)

    def test_walking_backwards_signs_velocity_but_not_speed_or_displacement(self):
        env = FakeLocomotionEnv(forward_speed=-0.5, dt=0.1)
        stats = rollout.evaluate_policy(
            env, rollout.zero_policy(4), num_episodes=1, episode_length=20
        )
        self.assertAlmostEqual(stats.mean_forward_velocity, -0.5, places=5)
        self.assertAlmostEqual(stats.mean_forward_speed, 0.5, places=5)
        self.assertAlmostEqual(stats.distance_travelled, -1.0, places=5)
        self.assertAlmostEqual(stats.net_displacement, 1.0, places=5)


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

    # Standing base heights measured off the Playground models, in metres.
    OP3_LEG = 0.2436
    BERKELEY_LEG = 0.5150
    APOLLO_LEG = 1.0800

    def make_stats(self, survived: float, speed: float, leg: float = 0.5150):
        """Stats with only the fields the predicate reads set meaningfully."""
        return rollout.RolloutStats(
            survived_fraction=survived,
            mean_forward_velocity=speed,
            mean_forward_speed=speed,
            distance_travelled=speed * survived,
            net_displacement=speed * survived,
            episode_return=0.0,
            nominal_leg_length=leg,
            num_episodes=1,
            num_steps=1,
        )

    def test_viable_when_both_thresholds_are_met(self):
        viable = rollout.is_viable(self.make_stats(0.95, 0.4))
        self.assertTrue(viable)
        self.assertIsInstance(viable, bool)

    def test_not_viable_when_it_falls_over(self):
        self.assertFalse(rollout.is_viable(self.make_stats(0.5, 0.4)))

    def test_a_motionless_robot_is_not_viable_by_default(self):
        # The defect this default exists to prevent: surviving by standing still.
        self.assertFalse(rollout.is_viable(self.make_stats(1.0, 0.0)))
        self.assertFalse(rollout.is_viable(self.make_stats(1.0, 0.02)))

    def test_the_speed_floor_is_size_normalised(self):
        # 0.2 m/s is a walk for a 0.24 m Op3 and a shuffle for a 1.08 m Apollo.
        self.assertTrue(rollout.is_viable(self.make_stats(1.0, 0.2, leg=self.OP3_LEG)))
        self.assertFalse(rollout.is_viable(self.make_stats(1.0, 0.2, leg=self.APOLLO_LEG)))

    def test_the_verdict_is_invariant_along_the_similarity_family(self):
        # Speed scales as sqrt(k) and length as k, so the same gait keeps its verdict.
        for scale in (0.25, 1.0, 4.0):
            walking = self.make_stats(1.0, 0.3 * math.sqrt(scale), leg=self.BERKELEY_LEG * scale)
            self.assertTrue(rollout.is_viable(walking))
            standing = self.make_stats(1.0, 0.05 * math.sqrt(scale), leg=self.BERKELEY_LEG * scale)
            self.assertFalse(rollout.is_viable(standing))

    def test_thresholds_are_inclusive(self):
        speed = math.sqrt(rollout.MIN_WALKING_FROUDE * rollout.STANDARD_GRAVITY * self.BERKELEY_LEG)
        self.assertTrue(
            rollout.is_viable(
                self.make_stats(0.8, speed),
                min_survived_fraction=0.8,
                min_froude_number=rollout.MIN_WALKING_FROUDE,
            )
        )

    def test_an_explicit_speed_floor_needs_no_body_scale(self):
        stats = self.make_stats(1.0, 0.3, leg=0.0)
        self.assertTrue(rollout.is_viable(stats, min_forward_speed=0.2))
        self.assertFalse(rollout.is_viable(stats, min_forward_speed=0.5))

    def test_refuses_to_guess_without_a_body_scale(self):
        with self.assertRaises(ValueError):
            rollout.is_viable(self.make_stats(1.0, 0.3, leg=0.0))


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables the playground rollout")
class PlaygroundIntegrationTest(unittest.TestCase):
    """Rollouts against a real MuJoCo Playground env, run past a real fall.

    The zero policy drops Berkeley Humanoid well inside HORIZON control steps, so the
    horizon is long enough that termination, and therefore the latching that masks
    everything after it, is exercised on real physics rather than only on the fake env.
    """

    HORIZON = 60

    @classmethod
    def setUpClass(cls):
        from mujoco_playground import registry

        cls.env = registry.load("BerkeleyHumanoidJoystickFlatTerrain")
        cls.stats = rollout.evaluate_policy(
            cls.env,
            rollout.zero_policy(cls.env.action_size),
            num_episodes=1,
            seed=0,
            episode_length=cls.HORIZON,
        )

    def zero_policy(self) -> rollout.Policy:
        """Baseline policy sized for the shared env."""
        return rollout.zero_policy(self.env.action_size)

    def alive_steps(self) -> int:
        """Control steps the shared rollout stayed alive for."""
        return round(self.stats.survived_fraction * self.HORIZON)

    def done_trace(self) -> jax.Array:
        """The env's raw done signal over the shared horizon, from the same reset."""
        policy = self.zero_policy()

        def step(state, _):
            next_state = self.env.step(state, policy(state.obs))
            return next_state, next_state.done

        rng = jax.random.split(jax.random.PRNGKey(0), 1)[0]
        _, done = jax.lax.scan(step, self.env.reset(rng), None, length=self.HORIZON)
        return done

    def test_the_terminating_step_is_counted_inclusively(self):
        # Measure the fall independently of the harness and check the count against it,
        # which pins the inclusive convention without hard coding the physics.
        done = self.done_trace()
        first_done = int(jnp.argmax(done > 0.5))
        self.assertGreater(float(done[first_done]), 0.5)
        self.assertEqual(self.alive_steps(), first_done + 1)

    def test_the_robot_really_terminates_inside_the_horizon(self):
        self.assertLess(self.stats.survived_fraction, 1.0)
        self.assertGreater(self.stats.survived_fraction, 0.0)
        self.assertAlmostEqual(
            self.stats.survived_fraction * self.HORIZON, self.alive_steps(), places=4
        )

    def test_statistics_are_finite_and_size_aware(self):
        self.assertEqual(self.stats.num_steps, self.HORIZON)
        self.assertTrue(math.isfinite(self.stats.mean_forward_velocity))
        self.assertTrue(math.isfinite(self.stats.mean_forward_speed))
        self.assertTrue(math.isfinite(self.stats.distance_travelled))
        self.assertTrue(math.isfinite(self.stats.net_displacement))
        self.assertTrue(math.isfinite(self.stats.episode_return))
        self.assertAlmostEqual(self.stats.nominal_leg_length, 0.515, places=3)
        # A robot that fell over inside 60 steps is not walking.
        self.assertFalse(rollout.is_viable(self.stats))

    def test_termination_latches_so_the_horizon_cannot_inflate_the_statistics(self):
        # Re-run the identical episode with the horizon cut to the terminating step, so
        # no post-termination step exists to be masked. Latching means the two rollouts
        # must agree; without it the longer one keeps integrating the fallen robot
        # sliding on the floor and reports several times the distance.
        alive_steps = self.alive_steps()
        self.assertLess(alive_steps, self.HORIZON)
        truncated = rollout.evaluate_policy(
            self.env, self.zero_policy(), num_episodes=1, seed=0, episode_length=alive_steps
        )
        # Every step of the shorter horizon was alive, which pins the terminating step as
        # counted inclusively, with no off-by-one.
        self.assertAlmostEqual(truncated.survived_fraction, 1.0, places=6)
        self.assertAlmostEqual(
            self.stats.distance_travelled, truncated.distance_travelled, places=6
        )
        self.assertAlmostEqual(self.stats.net_displacement, truncated.net_displacement, places=6)
        self.assertAlmostEqual(self.stats.episode_return, truncated.episode_return, places=5)
        self.assertAlmostEqual(
            self.stats.mean_forward_velocity, truncated.mean_forward_velocity, places=5
        )


if __name__ == "__main__":
    unittest.main()
