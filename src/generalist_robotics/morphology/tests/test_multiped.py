"""Unit tests for the procedural multiped and its leg-growth anneal."""

import math
import unittest

import mujoco
import numpy as np

from generalist_robotics.morphology import multiped
from generalist_robotics.morphology.multiped import LegGrowth, MultipedSpec

# Every leg of the four-leg superset held at zero, i.e. the biped configuration of it.
BIPED_GROWTH = (LegGrowth(2, 0.0), LegGrowth(3, 0.0))


def degrees(spec: MultipedSpec) -> list[int]:
    """Hip angles of a spec, in whole degrees, in growth order."""
    return [round(math.degrees(angle)) % 360 for angle in multiped.leg_angles(spec)]


def foot_world_height(model: mujoco.MjModel, spec: MultipedSpec, leg: int) -> float:
    """Height of one foot above the floor in the model's standing pose, in metres."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return float(data.site(multiped.foot_site_name(spec, leg)).xpos[2])


class MultipedSpecTest(unittest.TestCase):
    """The spec validates its own parameters and lays legs out in growth order."""

    def test_rejects_impossible_bodies(self):
        with self.assertRaises(ValueError):
            MultipedSpec(n_legs=0)
        with self.assertRaises(ValueError):
            MultipedSpec(leg_length=0.0)
        with self.assertRaises(ValueError):
            MultipedSpec(leg_mass=-1.0)
        with self.assertRaises(ValueError):
            MultipedSpec(hip_spacing=float("nan"))

    def test_biped_legs_are_the_lateral_pair(self):
        self.assertEqual(degrees(MultipedSpec(n_legs=2)), [90, 270])

    def test_quadruped_is_indexed_left_right_rear_front(self):
        self.assertEqual(degrees(MultipedSpec(n_legs=4)), [90, 270, 180, 0])

    def test_tripod_keeps_the_lateral_leg_first(self):
        self.assertEqual(degrees(MultipedSpec(n_legs=3)), [90, 210, 330])

    def test_hips_sit_on_a_circle_of_the_given_radius(self):
        spec = MultipedSpec(n_legs=4, hip_spacing=0.2)
        for leg in range(spec.n_legs):
            position = np.asarray(multiped.hip_position(spec, leg))
            self.assertAlmostEqual(float(np.linalg.norm(position)), 0.2, places=9)

    def test_leg_index_is_checked(self):
        spec = MultipedSpec(n_legs=2)
        with self.assertRaises(ValueError):
            multiped.leg_joint_names(spec, 2)
        with self.assertRaises(ValueError):
            multiped.hip_position(spec, -1)


class BuildMultipedTest(unittest.TestCase):
    """The generator compiles to a model whose size and naming follow the spec."""

    def test_model_shape_follows_leg_count(self):
        for n_legs in (2, 3, 4, 6):
            spec = MultipedSpec(n_legs=n_legs)
            model = multiped.build_multiped_model(spec)
            self.assertEqual(model.nu, 3 * n_legs)
            self.assertEqual(model.nq, 7 + 3 * n_legs)
            self.assertEqual(model.nv, 6 + 3 * n_legs)

    def test_every_named_joint_actuator_and_site_exists(self):
        spec = MultipedSpec(n_legs=4)
        model = multiped.build_multiped_model(spec)
        for leg in range(spec.n_legs):
            for name in multiped.leg_joint_names(spec, leg):
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0
                )
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name), 0
                )
            for name in multiped.leg_geom_names(spec, leg):
                self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), 0)
            site = multiped.foot_site_name(spec, leg)
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site), 0)

    def test_masses_add_up_to_the_spec(self):
        spec = MultipedSpec(n_legs=4)
        model = multiped.build_multiped_model(spec)
        expected = spec.torso_mass + spec.n_legs * spec.leg_mass
        self.assertAlmostEqual(multiped.total_mass(model), expected, places=6)

    def test_home_keyframe_stands_every_foot_on_the_floor(self):
        spec = MultipedSpec(n_legs=4)
        model = multiped.build_multiped_model(spec)
        radius = multiped.FOOT_RADIUS_FRACTION * spec.leg_length
        for leg in range(spec.n_legs):
            self.assertAlmostEqual(foot_world_height(model, spec, leg), radius, places=6)

    def test_home_pose_matches_the_keyframe(self):
        spec = MultipedSpec(n_legs=3)
        model = multiped.build_multiped_model(spec)
        np.testing.assert_allclose(model.key_qpos[0], multiped.home_qpos(spec), atol=1e-6)
        np.testing.assert_allclose(model.key_ctrl[0], multiped.home_ctrl(spec), atol=1e-6)

    def test_only_feet_and_torso_can_touch_the_floor(self):
        spec = MultipedSpec(n_legs=4)
        model = multiped.build_multiped_model(spec)
        colliding = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in range(model.ngeom)
            if model.geom_conaffinity[geom] != 0
        }
        expected = {"floor", "torso"} | {f"leg{leg}_foot" for leg in range(spec.n_legs)}
        self.assertEqual(colliding, expected)

    def test_damping_is_explicit_so_the_lock_stability_bound_holds(self):
        model = multiped.build_multiped_model(MultipedSpec(n_legs=2))
        self.assertTrue(bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP))
        self.assertAlmostEqual(float(model.opt.timestep), multiped.SIM_TIMESTEP, places=9)


class LegGrowthTest(unittest.TestCase):
    """Growth shrinks a leg toward its hip and holds its joints rigid, both continuously."""

    def setUp(self):
        self.spec = MultipedSpec(n_legs=4)
        self.model = multiped.build_multiped_model(self.spec)

    def test_growth_is_validated(self):
        with self.assertRaises(ValueError):
            LegGrowth(0, 1.5)
        with self.assertRaises(ValueError):
            LegGrowth(-1, 0.5)
        with self.assertRaises(ValueError):
            multiped.growth_by_leg(self.spec, (LegGrowth(1, 0.0), LegGrowth(1, 1.0)))
        with self.assertRaises(ValueError):
            multiped.growth_by_leg(self.spec, (LegGrowth(9, 0.0),))

    def test_unnamed_legs_stay_fully_grown(self):
        self.assertEqual(multiped.growth_by_leg(self.spec, (LegGrowth(3, 0.25),)), (1, 1, 1, 0.25))

    def test_full_growth_leaves_the_model_alone(self):
        grown = multiped.apply_leg_growth(self.model, self.spec, (LegGrowth(2, 1.0),))
        np.testing.assert_allclose(grown.body_mass, self.model.body_mass)
        np.testing.assert_allclose(grown.geom_size, self.model.geom_size)
        np.testing.assert_allclose(grown.jnt_stiffness, self.model.jnt_stiffness)
        np.testing.assert_allclose(grown.actuator_gainprm, self.model.actuator_gainprm)

    def test_the_input_model_is_not_touched(self):
        before = np.array(self.model.body_mass)
        multiped.apply_leg_growth(self.model, self.spec, BIPED_GROWTH)
        np.testing.assert_allclose(self.model.body_mass, before)

    def test_an_ungrown_leg_is_a_stub_at_its_hip(self):
        grown = multiped.apply_leg_growth(self.model, self.spec, BIPED_GROWTH)
        names = multiped.leg_body_names(self.spec, 2)
        mass = sum(float(grown.body(name).mass[0]) for name in names)
        self.assertLess(mass, multiped.GROWTH_MASS_FLOOR * self.spec.leg_mass * 1.001)
        self.assertGreater(mass, 0.0)
        reach = self.spec.leg_length * multiped.GROWTH_LENGTH_FLOOR
        floor = multiped.standing_height(self.spec) - reach - 1e-9
        self.assertGreater(foot_world_height(grown, self.spec, 2), floor)

    def test_an_ungrown_leg_cannot_steer_and_cannot_move(self):
        grown = multiped.apply_leg_growth(self.model, self.spec, BIPED_GROWTH)
        for leg in (2, 3):
            for name in multiped.leg_joint_names(self.spec, leg):
                self.assertGreater(float(grown.jnt_stiffness[grown.joint(name).id]), 0.0)
                self.assertAlmostEqual(float(grown.actuator(name).gainprm[0]), 0.0, places=12)

    def test_a_grown_leg_keeps_its_full_authority(self):
        grown = multiped.apply_leg_growth(self.model, self.spec, BIPED_GROWTH)
        for leg in (0, 1):
            for name in multiped.leg_joint_names(self.spec, leg):
                self.assertEqual(float(grown.jnt_stiffness[grown.joint(name).id]), 0.0)
                self.assertAlmostEqual(
                    float(grown.actuator(name).gainprm[0]),
                    float(self.model.actuator(name).gainprm[0]),
                    places=9,
                )

    def test_mass_and_reach_increase_with_growth(self):
        masses = []
        heights = []
        for growth in (0.0, 0.25, 0.5, 0.75, 1.0):
            grown = multiped.apply_leg_growth(self.model, self.spec, (LegGrowth(2, growth),))
            masses.append(multiped.total_mass(grown))
            heights.append(foot_world_height(grown, self.spec, 2))
        self.assertTrue(all(np.diff(masses) > 0.0))
        self.assertTrue(all(np.diff(heights) < 0.0))

    def test_inertia_follows_mass_times_length_squared(self):
        growth = 0.5
        grown = multiped.apply_leg_growth(self.model, self.spec, (LegGrowth(2, growth),))
        expected = multiped.growth_mass_scale(growth) * multiped.growth_length_scale(growth) ** 2
        name = multiped.leg_body_names(self.spec, 2)[1]
        np.testing.assert_allclose(
            grown.body(name).inertia, self.model.body(name).inertia * expected, rtol=1e-9
        )

    def test_locks_name_every_leg_so_the_spring_scale_never_jumps(self):
        locks = multiped.leg_growth_locks(self.spec, (LegGrowth(2, 0.4),))
        self.assertEqual(len(locks), self.spec.n_legs)
        self.assertEqual({lock.lock for lock in locks}, {0.0, 0.6})

    def test_the_locks_grip_falls_away_smoothly_as_the_leg_grows(self):
        name = multiped.leg_joint_names(self.spec, 2)[1]
        rigidity = []
        for growth in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            grown = multiped.apply_leg_growth(self.model, self.spec, (LegGrowth(2, growth),))
            inertia = multiped.growth_mass_scale(growth) * multiped.growth_length_scale(growth) ** 2
            rigidity.append(float(grown.jnt_stiffness[grown.joint(name).id]) / inertia)
        # What a lock has to beat is the inertia it holds, so that ratio is the grip.
        self.assertTrue(all(np.diff(rigidity) < 0.0))
        self.assertEqual(rigidity[-1], 0.0)


class SettleTest(unittest.TestCase):
    """A model with a static support polygon holds its standing pose under its own servos."""

    def test_three_and_four_legs_stand_up(self):
        for n_legs in (3, 4):
            spec = MultipedSpec(n_legs=n_legs)
            model = multiped.build_multiped_model(spec)
            data = multiped.settle(model, seconds=2.0)
            self.assertTrue(bool(np.all(np.isfinite(data.qpos))))
            self.assertGreater(float(data.qpos[2]), 0.9 * multiped.standing_height(spec))

    def test_an_ungrown_leg_does_not_explode_the_integrator(self):
        spec = MultipedSpec(n_legs=4)
        model = multiped.apply_leg_growth(
            multiped.build_multiped_model(spec), spec, (LegGrowth(3, 0.0),)
        )
        data = multiped.settle(model, seconds=2.0)
        self.assertTrue(bool(np.all(np.isfinite(data.qpos))))
        self.assertLess(float(np.linalg.norm(data.qvel)), 1.0)


if __name__ == "__main__":
    unittest.main()
