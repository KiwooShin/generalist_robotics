"""Tests for the morphed MuJoCo Playground locomotion environment factory."""

import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco_playground import registry

from generalist_robotics.envs.locomotion import (
    available_robots,
    check_morph_reached_env,
    environment_id,
    make_locomotion_env,
    morphed_compilation,
    similarity_time_overrides,
    simulated_total_mass,
)
from generalist_robotics.morphology.scaling import (
    MorphParams,
    dynamic_similarity_params,
    similar_time_scale,
)

ROBOT = "berkeley_humanoid"
SIZE_SCALE = 2.0
MASS_SCALE = 5.0

EMPTY_MODEL_XML = (
    "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"
)


def rollout_qvel(env, rng: jax.Array, steps: int) -> np.ndarray:
    """Reset an environment and return qvel after a fixed number of zero-action steps."""
    state = env.reset(rng)
    action = jp.zeros(env.action_size)
    for _ in range(steps):
        state = env.step(state, action)
    return np.asarray(state.data.qvel)


class RobotRegistryTest(unittest.TestCase):
    """The short-name to Playground-id mapping."""

    def test_required_robots_are_registered(self):
        expected = {
            "berkeley_humanoid": "BerkeleyHumanoidJoystickFlatTerrain",
            "g1": "G1JoystickFlatTerrain",
            "t1": "T1JoystickFlatTerrain",
            "h1": "H1JoystickGaitTracking",
            "op3": "Op3Joystick",
            "apollo": "ApolloJoystickFlatTerrain",
            "go1": "Go1JoystickFlatTerrain",
        }
        robots = available_robots()
        for name, env_id in expected.items():
            self.assertEqual(robots[name], env_id)

    def test_every_environment_id_exists_in_playground(self):
        for name, env_id in available_robots().items():
            self.assertIn(env_id, registry.ALL_ENVS, msg=name)

    def test_available_robots_returns_a_copy(self):
        robots = available_robots()
        robots.pop(ROBOT)
        self.assertIn(ROBOT, available_robots())

    def test_unknown_robot_raises_with_valid_names(self):
        with self.assertRaises(ValueError) as caught:
            environment_id("humanoid_deluxe")
        message = str(caught.exception)
        self.assertIn("humanoid_deluxe", message)
        self.assertIn(ROBOT, message)

    def test_make_locomotion_env_rejects_unknown_robot(self):
        with self.assertRaises(ValueError):
            make_locomotion_env("humanoid_deluxe")


class MorphedCompilationTest(unittest.TestCase):
    """The compile-time hook that injects the morphology."""

    def test_models_compiled_inside_the_block_are_morphed(self):
        with morphed_compilation(MorphParams(mass_scale=3.0)) as produced:
            model = mujoco.MjModel.from_xml_string(EMPTY_MODEL_XML)
        self.assertEqual(len(produced), 1)
        self.assertIs(produced[0], model)
        plain = mujoco.MjModel.from_xml_string(EMPTY_MODEL_XML)
        np.testing.assert_allclose(model.body_mass, plain.body_mass * 3.0, rtol=1e-6)

    def test_compilation_is_restored_after_the_block(self):
        with morphed_compilation(MorphParams(mass_scale=3.0)):
            pass
        before = mujoco.MjModel.from_xml_string(EMPTY_MODEL_XML).body_mass.copy()
        after = mujoco.MjModel.from_xml_string(EMPTY_MODEL_XML).body_mass
        np.testing.assert_allclose(before, after)

    def test_compilation_is_restored_after_an_error(self):
        with (
            self.assertRaises(RuntimeError),
            morphed_compilation(MorphParams(mass_scale=3.0)),
        ):
            raise RuntimeError("boom")
        model = mujoco.MjModel.from_xml_string(EMPTY_MODEL_XML)
        self.assertAlmostEqual(float(model.body_mass.sum()), 4.1887902, places=5)


class MorphedModelTest(unittest.TestCase):
    """A dynamically similar morph as seen by the MJX model that step integrates."""

    @classmethod
    def setUpClass(cls):
        cls.params = dynamic_similarity_params(SIZE_SCALE)
        cls.base = make_locomotion_env(ROBOT)
        cls.scaled = make_locomotion_env(ROBOT, cls.params)

    def test_environment_api_is_unchanged_by_the_morph(self):
        self.assertEqual(self.scaled.action_size, self.base.action_size)
        self.assertEqual(self.scaled.observation_size, self.base.observation_size)

    def test_mjx_masses_follow_the_mass_scale(self):
        base_mass = np.asarray(self.base.mjx_model.body_mass)
        scaled_mass = np.asarray(self.scaled.mjx_model.body_mass)
        np.testing.assert_allclose(scaled_mass, base_mass * self.params.mass_scale, rtol=1e-5)
        self.assertAlmostEqual(
            simulated_total_mass(self.scaled),
            simulated_total_mass(self.base) * self.params.mass_scale,
            places=3,
        )

    def test_mjx_geometry_follows_the_size_scale(self):
        base_size = np.asarray(self.base.mjx_model.geom_size)
        scaled_size = np.asarray(self.scaled.mjx_model.geom_size)
        np.testing.assert_allclose(scaled_size, base_size * SIZE_SCALE, rtol=1e-5)

    def test_mjx_torque_limits_follow_the_torque_scale(self):
        base_limits = np.asarray(self.base.mjx_model.jnt_actfrcrange)
        scaled_limits = np.asarray(self.scaled.mjx_model.jnt_actfrcrange)
        np.testing.assert_allclose(scaled_limits, base_limits * self.params.torque_scale, rtol=1e-5)

    def test_mj_model_and_mjx_model_agree(self):
        np.testing.assert_allclose(
            np.asarray(self.scaled.mjx_model.body_mass),
            self.scaled.mj_model.body_mass,
            rtol=1e-5,
        )

    def test_cached_start_pose_follows_the_size_scale(self):
        base_start = self.base.reset(jax.random.PRNGKey(0))
        scaled_start = self.scaled.reset(jax.random.PRNGKey(0))
        self.assertAlmostEqual(
            float(scaled_start.data.qpos[2]),
            float(base_start.data.qpos[2]) * SIZE_SCALE,
            places=4,
        )

    def test_identity_morph_leaves_the_model_alone(self):
        plain = make_locomotion_env(ROBOT, MorphParams())
        np.testing.assert_allclose(
            np.asarray(plain.mjx_model.body_mass),
            np.asarray(self.base.mjx_model.body_mass),
        )


class MorphedPhysicsTest(unittest.TestCase):
    """The morph must reach the physics that env.step actually integrates."""

    STEPS = 5

    @classmethod
    def setUpClass(cls):
        cls.rng = jax.random.PRNGKey(0)
        cls.base = make_locomotion_env(ROBOT)
        cls.heavy = make_locomotion_env(ROBOT, MorphParams(mass_scale=MASS_SCALE))
        cls.base_qvel = rollout_qvel(cls.base, cls.rng, cls.STEPS)
        cls.heavy_qvel = rollout_qvel(cls.heavy, cls.rng, cls.STEPS)

    def test_mass_only_morph_leaves_the_reset_state_identical(self):
        base_start = self.base.reset(self.rng)
        heavy_start = self.heavy.reset(self.rng)
        np.testing.assert_allclose(
            np.asarray(heavy_start.data.qpos), np.asarray(base_start.data.qpos)
        )
        np.testing.assert_allclose(
            np.asarray(heavy_start.data.qvel), np.asarray(base_start.data.qvel)
        )

    def test_rollout_is_deterministic(self):
        repeat = rollout_qvel(self.base, self.rng, self.STEPS)
        np.testing.assert_allclose(repeat, self.base_qvel)

    def test_mass_only_morph_changes_the_stepped_dynamics(self):
        # Same start state and same zero action, so any divergence comes from mjx_model.
        difference = np.linalg.norm(self.heavy_qvel - self.base_qvel)
        self.assertGreater(difference, 1e-2)


class TimeScalingTest(unittest.TestCase):
    """Time scaling is opt-in and never silently half-applied."""

    def test_time_is_not_scaled_by_default(self):
        env = make_locomotion_env(ROBOT, dynamic_similarity_params(SIZE_SCALE))
        base = registry.get_default_config(environment_id(ROBOT))
        self.assertAlmostEqual(env.dt, float(base.ctrl_dt))
        self.assertAlmostEqual(env.sim_dt, float(base.sim_dt))

    def test_scale_time_stretches_both_clocks(self):
        env = make_locomotion_env(ROBOT, dynamic_similarity_params(SIZE_SCALE), scale_time=True)
        base = registry.get_default_config(environment_id(ROBOT))
        factor = similar_time_scale(SIZE_SCALE)
        self.assertAlmostEqual(env.dt, float(base.ctrl_dt) * factor)
        self.assertAlmostEqual(env.sim_dt, float(base.sim_dt) * factor)
        self.assertAlmostEqual(env.mj_model.opt.timestep, float(base.sim_dt) * factor)

    def test_substep_count_is_preserved_by_time_scaling(self):
        env = make_locomotion_env(ROBOT, dynamic_similarity_params(SIZE_SCALE), scale_time=True)
        self.assertEqual(env.n_substeps, make_locomotion_env(ROBOT).n_substeps)

    def test_explicit_config_overrides_win_over_time_scaling(self):
        env = make_locomotion_env(
            ROBOT,
            dynamic_similarity_params(SIZE_SCALE),
            config_overrides={"ctrl_dt": 0.05},
            scale_time=True,
        )
        self.assertAlmostEqual(env.dt, 0.05)

    def test_similarity_time_overrides_scale_the_base_config(self):
        env_id = environment_id(ROBOT)
        base = registry.get_default_config(env_id)
        overrides = similarity_time_overrides(env_id, SIZE_SCALE)
        factor = similar_time_scale(SIZE_SCALE)
        self.assertAlmostEqual(overrides["ctrl_dt"], float(base.ctrl_dt) * factor)
        self.assertAlmostEqual(overrides["sim_dt"], float(base.sim_dt) * factor)


class ConfigOverrideTest(unittest.TestCase):
    """Playground config overrides reach the constructed environment."""

    def test_flat_override_is_applied(self):
        env = make_locomotion_env(ROBOT, config_overrides={"episode_length": 123})
        self.assertEqual(env._config.episode_length, 123)

    def test_nested_override_is_applied(self):
        env = make_locomotion_env(
            ROBOT,
            MorphParams(size_scale=1.5),
            config_overrides={"noise_config.level": 0.0},
        )
        self.assertEqual(env._config.noise_config.level, 0.0)


class MorphGuardTest(unittest.TestCase):
    """The guard against a morph that never reached the simulated model."""

    def test_guard_rejects_an_environment_built_from_another_model(self):
        env = make_locomotion_env(ROBOT)
        with self.assertRaises(RuntimeError):
            check_morph_reached_env(env, [])

    def test_guard_accepts_a_properly_morphed_environment(self):
        with morphed_compilation(MorphParams(mass_scale=2.0)) as produced:
            env = registry.load(environment_id(ROBOT))
        check_morph_reached_env(env, produced)


if __name__ == "__main__":
    unittest.main()
