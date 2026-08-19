"""Tests for the morphed MuJoCo Playground locomotion environment factory."""

import types
import unittest

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco_playground import registry

from generalist_robotics.envs.locomotion import (
    MORPH_WITNESS_FIELDS,
    TASK_SIZE_EXPONENTS,
    available_robots,
    check_morph_reached_env,
    environment_id,
    flatten_config,
    make_locomotion_env,
    morph_gain_overrides,
    morphed_compilation,
    scaled_task_value,
    similarity_task_overrides,
    similarity_time_overrides,
    simulated_total_mass,
)
from generalist_robotics.morphology.scaling import (
    MorphParams,
    apply_morphology,
    dynamic_similarity_params,
    similar_time_scale,
)

ROBOT = "berkeley_humanoid"
GEARED_ROBOT = "go1"  # Keeps its servo gains in the task config rather than the XML.
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

    def test_identity_morph_through_the_real_morph_code_is_a_no_op(self):
        # make_locomotion_env short circuits identity params without calling
        # apply_morphology, so comparing its output against registry.load would assert
        # nothing. Going through morphed_compilation instead runs the real morph, deep
        # copy and mj_setConst refresh included, on the actual robot.
        with morphed_compilation(MorphParams()) as produced:
            identity = registry.load(environment_id(ROBOT))
        self.assertEqual(len(produced), 1)
        check_morph_reached_env(identity, produced)
        for name in (*MORPH_WITNESS_FIELDS, "jnt_actfrcrange", "body_inertia", "qpos0"):
            np.testing.assert_allclose(
                np.asarray(getattr(identity.mjx_model, name)),
                np.asarray(getattr(self.base.mjx_model, name)),
                rtol=1e-12,
                atol=0.0,
                err_msg=name,
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


class TaskScalingOverridesTest(unittest.TestCase):
    """The override table that restates a task in the scaled robot's units."""

    def test_every_tabled_key_exists_in_at_least_one_robot_config(self):
        defined = set()
        for env_id in available_robots().values():
            defined.update(flatten_config(registry.get_default_config(env_id)))
        for key in TASK_SIZE_EXPONENTS:
            self.assertIn(key, defined, msg=key)

    def test_reference_robot_scales_exactly_the_dimensional_entries(self):
        overrides = similarity_task_overrides(environment_id(ROBOT), SIZE_SCALE)
        self.assertEqual(
            set(overrides),
            {
                "reward_config.max_foot_height",
                "reward_config.base_height_target",
                "lin_vel_x",
                "lin_vel_y",
                "ang_vel_yaw",
                "push_config.magnitude_range",
                "push_config.interval_range",
                "noise_config.scales.linvel",
                "noise_config.scales.gyro",
                "noise_config.scales.joint_vel",
            },
        )

    def test_lengths_speeds_rates_and_durations_use_their_own_exponents(self):
        base = registry.get_default_config(environment_id(ROBOT))
        overrides = similarity_task_overrides(environment_id(ROBOT), SIZE_SCALE)
        root = similar_time_scale(SIZE_SCALE)
        self.assertAlmostEqual(
            overrides["reward_config.max_foot_height"],
            float(base.reward_config.max_foot_height) * SIZE_SCALE,
        )
        self.assertAlmostEqual(
            overrides["reward_config.base_height_target"],
            float(base.reward_config.base_height_target) * SIZE_SCALE,
        )
        np.testing.assert_allclose(overrides["lin_vel_x"], np.array(base.lin_vel_x) * root)
        np.testing.assert_allclose(overrides["ang_vel_yaw"], np.array(base.ang_vel_yaw) / root)
        np.testing.assert_allclose(
            overrides["push_config.interval_range"],
            np.array(base.push_config.interval_range) * root,
        )
        np.testing.assert_allclose(
            overrides["push_config.magnitude_range"],
            np.array(base.push_config.magnitude_range) * root,
        )
        self.assertAlmostEqual(
            overrides["noise_config.scales.joint_vel"],
            float(base.noise_config.scales.joint_vel) / root,
        )

    def test_command_vectors_scale_component_wise(self):
        env_id = environment_id(GEARED_ROBOT)
        base = registry.get_default_config(env_id)
        overrides = similarity_task_overrides(env_id, SIZE_SCALE)
        root = similar_time_scale(SIZE_SCALE)
        forward, lateral, yaw = base.command_config.a
        np.testing.assert_allclose(
            overrides["command_config.a"],
            [forward * root, lateral * root, yaw / root],
            rtol=1e-12,
        )
        self.assertNotIn("command_config.b", overrides)

    def test_missing_keys_are_skipped_per_robot(self):
        quadruped = similarity_task_overrides(environment_id(GEARED_ROBOT), SIZE_SCALE)
        gait_tracker = similarity_task_overrides(environment_id("h1"), SIZE_SCALE)
        self.assertIn("pert_config.velocity_kick", quadruped)
        self.assertNotIn("lin_vel_x", quadruped)
        self.assertIn("gait_frequency", gait_tracker)
        self.assertIn("command_config.ang_vel_yaw", gait_tracker)
        self.assertNotIn("reward_config.max_foot_height", gait_tracker)

    def test_identity_size_leaves_every_entry_where_it_was(self):
        env_id = environment_id(ROBOT)
        base = flatten_config(registry.get_default_config(env_id))
        for key, value in similarity_task_overrides(env_id, 1.0).items():
            np.testing.assert_allclose(value, base[key], rtol=1e-12, err_msg=key)

    def test_component_exponents_reject_a_mismatched_entry(self):
        with self.assertRaises(ValueError):
            scaled_task_value([1.0, 2.0], (0.5, 0.5, -0.5), SIZE_SCALE)


class TaskScalingEnvironmentTest(unittest.TestCase):
    """Task scaling is opt-in and reaches the constructed environment."""

    @classmethod
    def setUpClass(cls):
        cls.params = dynamic_similarity_params(SIZE_SCALE)
        cls.base_config = registry.get_default_config(environment_id(ROBOT))
        cls.unscaled = make_locomotion_env(ROBOT, cls.params)
        cls.scaled = make_locomotion_env(ROBOT, cls.params, scale_task=True)

    def test_task_is_not_scaled_by_default(self):
        self.assertAlmostEqual(
            float(self.unscaled._config.reward_config.max_foot_height),
            float(self.base_config.reward_config.max_foot_height),
        )
        np.testing.assert_allclose(self.unscaled._config.lin_vel_x, self.base_config.lin_vel_x)

    def test_scale_task_reaches_the_environment_config(self):
        root = similar_time_scale(SIZE_SCALE)
        self.assertAlmostEqual(
            float(self.scaled._config.reward_config.max_foot_height),
            float(self.base_config.reward_config.max_foot_height) * SIZE_SCALE,
        )
        np.testing.assert_allclose(
            self.scaled._config.lin_vel_x, np.array(self.base_config.lin_vel_x) * root
        )
        np.testing.assert_allclose(
            self.scaled._config.ang_vel_yaw, np.array(self.base_config.ang_vel_yaw) / root
        )

    def test_reward_weights_and_action_scale_are_untouched(self):
        self.assertAlmostEqual(
            float(self.scaled._config.reward_config.scales.feet_phase),
            float(self.base_config.reward_config.scales.feet_phase),
        )
        self.assertAlmostEqual(
            float(self.scaled._config.reward_config.tracking_sigma),
            float(self.base_config.reward_config.tracking_sigma),
        )
        self.assertAlmostEqual(
            float(self.scaled._config.action_scale), float(self.base_config.action_scale)
        )

    def test_commanded_speeds_actually_widen_with_size(self):
        commands = np.array(
            [self.scaled.sample_command(jax.random.PRNGKey(seed)) for seed in range(16)]
        )
        base_commands = np.array(
            [self.unscaled.sample_command(jax.random.PRNGKey(seed)) for seed in range(16)]
        )
        self.assertGreater(np.abs(commands[:, 0]).max(), np.abs(base_commands[:, 0]).max())

    def test_explicit_config_overrides_win_over_task_scaling(self):
        env = make_locomotion_env(
            ROBOT,
            self.params,
            config_overrides={"reward_config.max_foot_height": 0.42, "lin_vel_x": [-3.0, 3.0]},
            scale_task=True,
        )
        self.assertAlmostEqual(float(env._config.reward_config.max_foot_height), 0.42)
        np.testing.assert_allclose(env._config.lin_vel_x, [-3.0, 3.0])


class MorphGainOverrideTest(unittest.TestCase):
    """Servo gains a robot keeps in its config must follow the morph too."""

    def test_reference_robot_has_no_config_gains(self):
        self.assertEqual(morph_gain_overrides(environment_id(ROBOT), MorphParams()), {})

    def test_config_gains_follow_the_similarity_exponents(self):
        env_id = environment_id(GEARED_ROBOT)
        base = registry.get_default_config(env_id)
        overrides = morph_gain_overrides(env_id, dynamic_similarity_params(SIZE_SCALE))
        self.assertAlmostEqual(overrides["Kp"], float(base.Kp) * SIZE_SCALE**4)
        self.assertAlmostEqual(overrides["Kd"], float(base.Kd) * SIZE_SCALE**4.5)

    def test_config_gains_follow_a_mass_only_morph(self):
        env_id = environment_id(GEARED_ROBOT)
        base = registry.get_default_config(env_id)
        overrides = morph_gain_overrides(env_id, MorphParams(mass_scale=MASS_SCALE))
        self.assertAlmostEqual(overrides["Kp"], float(base.Kp))
        self.assertAlmostEqual(overrides["Kd"], float(base.Kd) * MASS_SCALE)

    def test_config_gains_reach_the_simulated_model(self):
        # Go1 writes config.Kp and config.Kd over the compiled model before mjx.put_model,
        # so without the override the morphed gains never reach the physics.
        env_id = environment_id(GEARED_ROBOT)
        base = registry.get_default_config(env_id)
        env = make_locomotion_env(GEARED_ROBOT, dynamic_similarity_params(SIZE_SCALE))
        gains = np.asarray(env.mjx_model.actuator_gainprm)[:, 0]
        damping = np.asarray(env.mjx_model.dof_damping)[6:]
        np.testing.assert_allclose(gains, float(base.Kp) * SIZE_SCALE**4, rtol=1e-6)
        np.testing.assert_allclose(damping, float(base.Kd) * SIZE_SCALE**4.5, rtol=1e-6)


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

    def test_guard_rejects_a_stale_mjx_model_after_a_size_only_morph(self):
        # A size-only morph leaves every body mass alone, so a mass comparison cannot
        # tell the morphed model from the MJX model of the unmorphed one.
        env = make_locomotion_env(ROBOT)
        morphed = apply_morphology(env.mj_model, MorphParams(size_scale=SIZE_SCALE))
        np.testing.assert_allclose(morphed.body_mass, env.mj_model.body_mass)
        stale = types.SimpleNamespace(mj_model=morphed, mjx_model=env.mjx_model)
        with self.assertRaises(RuntimeError):
            check_morph_reached_env(stale, [morphed])


if __name__ == "__main__":
    unittest.main()
