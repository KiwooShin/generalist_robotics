"""Unit tests for parametric morphology scaling of MuJoCo models."""

import copy
import unittest

import mujoco
import numpy as np

from generalist_robotics import morphology

# A small robot that exercises every path: a mesh geom, a free joint, hinge and
# slide joints, a joint-level torque limit, a servo and a direct-drive motor.
SYNTHETIC_XML = """
<mujoco model="morphology_test">
  <asset>
    <mesh name="wedge" vertex="0 0 0  0.2 0 0  0 0.3 0  0 0 0.4"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
    <body name="root" pos="0 0 0.5">
      <freejoint/>
      <geom name="torso" type="box" size="0.1 0.05 0.2" mass="2"/>
      <geom name="shell" type="mesh" mesh="wedge" pos="0 0 0.2" mass="0.5"/>
      <site name="imu" pos="0 0 0.1" size="0.01"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="hip" type="hinge" axis="0 1 0" pos="0 0 0.05"
               range="-1 1" actuatorfrcrange="-7 7" armature="0.01"/>
        <geom name="leg" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
      </body>
      <body name="arm" pos="0.1 0 0.1">
        <joint name="shoulder" type="hinge" axis="1 0 0" range="-2 2"/>
        <geom name="forearm" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.4"/>
      </body>
      <body name="slider" pos="-0.1 0 0">
        <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.2" armature="0.5"/>
        <geom name="pad" type="sphere" size="0.04" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_servo" joint="hip" kp="20" ctrlrange="-1 1"/>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-5 5" forcerange="-6 6"/>
    <position name="lift_servo" joint="lift" kp="100" ctrlrange="-0.2 0.2"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0 0.3 -0.4 0.05"/>
  </keyframe>
</mujoco>
"""

PLAYGROUND_ENV = "BerkeleyHumanoidJoystickFlatTerrain"


def saturated_actuator_force(model: mujoco.MjModel) -> np.ndarray:
    """Return the joint forces produced by commanding every actuator far past its limit."""
    unclamped = copy.deepcopy(model)
    unclamped.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    data = mujoco.MjData(unclamped)
    if unclamped.nkey:
        data.qpos[:] = unclamped.key_qpos[0]
    data.ctrl[:] = 1e3
    mujoco.mj_forward(unclamped, data)
    return data.qfrc_actuator.copy()


def torque_limited_dofs(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of degrees of freedom whose actuator torque is capped by the model."""
    limited = np.zeros(model.nv, dtype=bool)
    for joint in range(model.njnt):
        if model.jnt_actfrclimited[joint]:
            limited[model.jnt_dofadr[joint]] = True
    for actuator in range(model.nu):
        if not model.actuator_forcelimited[actuator]:
            continue
        if model.actuator_trntype[actuator] in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        ):
            limited[model.jnt_dofadr[model.actuator_trnid[actuator, 0]]] = True
    return limited


def lowest_contact_height(model: mujoco.MjModel) -> float:
    """Return the world height of the lowest collidable point of the robot at its keyframe."""
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_kinematics(model, data)
    heights = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or model.geom_contype[geom] == 0:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        world = data.geom_xpos[geom] + morphology.geom_bounding_points(model, geom) @ rotation.T
        heights.append(float(world[:, 2].min()))
    return min(heights)


def simulate(model: mujoco.MjModel, steps: int = 300) -> mujoco.MjData:
    """Step a model forward from its keyframe holding the keyframe controls."""
    data = mujoco.MjData(model)
    if model.nkey:
        data.qpos[:] = model.key_qpos[0]
        data.ctrl[:] = model.key_ctrl[0]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


class TestMorphParams(unittest.TestCase):
    """Checks the morphology parameter container."""

    def test_defaults_are_the_identity_morphology(self):
        params = morphology.MorphParams()
        self.assertEqual(params.size_scale, 1.0)
        self.assertEqual(params.mass_scale, 1.0)
        self.assertEqual(params.torque_scale, 1.0)

    def test_rejects_non_positive_factors(self):
        for kwargs in ({"size_scale": 0.0}, {"mass_scale": -1.0}, {"torque_scale": -0.5}):
            with self.assertRaises(ValueError):
                morphology.MorphParams(**kwargs)

    def test_rejects_non_finite_factors(self):
        with self.assertRaises(ValueError):
            morphology.MorphParams(size_scale=float("nan"))
        with self.assertRaises(ValueError):
            morphology.MorphParams(mass_scale=float("inf"))

    def test_is_frozen(self):
        params = morphology.MorphParams()
        with self.assertRaises(Exception):
            params.size_scale = 2.0


class TestSimilarityHelpers(unittest.TestCase):
    """Checks the dynamic-similarity scaling laws."""

    def test_exponents(self):
        self.assertAlmostEqual(morphology.similar_mass_scale(2.0), 8.0)
        self.assertAlmostEqual(morphology.similar_torque_scale(2.0), 16.0)
        self.assertAlmostEqual(morphology.similar_time_scale(4.0), 2.0)

    def test_identity_at_unit_size(self):
        self.assertAlmostEqual(morphology.similar_mass_scale(1.0), 1.0)
        self.assertAlmostEqual(morphology.similar_torque_scale(1.0), 1.0)
        self.assertAlmostEqual(morphology.similar_time_scale(1.0), 1.0)

    def test_dynamic_similarity_params_combines_the_laws(self):
        params = morphology.dynamic_similarity_params(3.0)
        self.assertAlmostEqual(params.size_scale, 3.0)
        self.assertAlmostEqual(params.mass_scale, 27.0)
        self.assertAlmostEqual(params.torque_scale, 81.0)


class TestInterpolate(unittest.TestCase):
    """Checks blending between two morphologies."""

    def test_endpoints_are_reproduced(self):
        start = morphology.MorphParams(1.0, 1.0, 1.0)
        end = morphology.dynamic_similarity_params(2.0)
        self.assertEqual(morphology.interpolate(start, end, 0.0), start)
        for a, b in zip(
            morphology.interpolate(start, end, 1.0).__dict__.values(), end.__dict__.values()
        ):
            self.assertAlmostEqual(a, b)

    def test_midpoint_is_the_geometric_mean(self):
        start = morphology.MorphParams(1.0, 1.0, 1.0)
        end = morphology.MorphParams(4.0, 16.0, 64.0)
        middle = morphology.interpolate(start, end, 0.5)
        self.assertAlmostEqual(middle.size_scale, 2.0)
        self.assertAlmostEqual(middle.mass_scale, 4.0)
        self.assertAlmostEqual(middle.torque_scale, 8.0)

    def test_stays_on_the_dynamic_similarity_manifold(self):
        start = morphology.dynamic_similarity_params(1.0)
        end = morphology.dynamic_similarity_params(4.0)
        middle = morphology.interpolate(start, end, 0.5)
        expected = morphology.dynamic_similarity_params(2.0)
        self.assertAlmostEqual(middle.size_scale, expected.size_scale)
        self.assertAlmostEqual(middle.mass_scale, expected.mass_scale)
        self.assertAlmostEqual(middle.torque_scale, expected.torque_scale)

    def test_rejects_non_finite_alpha(self):
        params = morphology.MorphParams()
        with self.assertRaises(ValueError):
            morphology.interpolate(params, params, float("nan"))


class TestApplyMorphology(unittest.TestCase):
    """Checks field-level scaling on a small model that covers every joint and actuator type."""

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(SYNTHETIC_XML)

    def test_input_model_is_not_mutated(self):
        before = {
            name: getattr(self.model, name).copy()
            for name in ("body_pos", "body_mass", "body_inertia", "mesh_vert", "jnt_actfrcrange")
        }
        morphology.apply_morphology(self.model, morphology.dynamic_similarity_params(2.5))
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(self.model, name), value)

    def test_identity_params_reproduce_the_model(self):
        same = morphology.apply_morphology(self.model, morphology.MorphParams())
        for name in ("body_pos", "body_mass", "body_inertia", "mesh_vert", "geom_size", "qpos0"):
            np.testing.assert_array_equal(getattr(same, name), getattr(self.model, name))

    def test_every_length_field_scales(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=3.0))
        for name in (
            "body_pos",
            "body_ipos",
            "geom_pos",
            "geom_size",
            "geom_rbound",
            "site_pos",
            "site_size",
            "jnt_pos",
            "mesh_vert",
        ):
            np.testing.assert_allclose(
                getattr(scaled, name), getattr(self.model, name) * 3.0, rtol=1e-9, atol=1e-12,
                err_msg=f"{name} did not scale",
            )

    def test_mass_and_inertia_follow_their_own_laws(self):
        params = morphology.MorphParams(size_scale=2.0, mass_scale=5.0)
        scaled = morphology.apply_morphology(self.model, params)
        np.testing.assert_allclose(scaled.body_mass, self.model.body_mass * 5.0, rtol=1e-9)
        np.testing.assert_allclose(
            scaled.body_inertia, self.model.body_inertia * 5.0 * 4.0, rtol=1e-9
        )

    def test_armature_units_depend_on_the_joint_type(self):
        params = morphology.MorphParams(size_scale=2.0, mass_scale=5.0)
        scaled = morphology.apply_morphology(self.model, params)
        hip = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")]
        lift = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift")
        ]
        self.assertAlmostEqual(scaled.dof_armature[hip], self.model.dof_armature[hip] * 5.0 * 4.0)
        self.assertAlmostEqual(scaled.dof_armature[lift], self.model.dof_armature[lift] * 5.0)

    def test_size_scaling_alone_leaves_mass_untouched(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=4.0))
        self.assertAlmostEqual(morphology.total_mass(scaled), morphology.total_mass(self.model))

    def test_mass_scaling_alone_leaves_geometry_untouched(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(mass_scale=7.0))
        np.testing.assert_array_equal(scaled.mesh_vert, self.model.mesh_vert)
        self.assertAlmostEqual(
            morphology.physical_extent(scaled), morphology.physical_extent(self.model)
        )

    def test_torque_scaling_hits_both_limit_fields(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=3.0))
        np.testing.assert_allclose(
            scaled.jnt_actfrcrange, self.model.jnt_actfrcrange * 3.0, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled.actuator_forcerange, self.model.actuator_forcerange * 3.0, rtol=1e-9
        )

    def test_torque_scaling_moves_the_realised_joint_forces(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=3.0))
        base = saturated_actuator_force(self.model)
        grown = saturated_actuator_force(scaled)
        active = np.abs(base) > 1e-9
        limited = torque_limited_dofs(self.model)
        self.assertTrue((active & limited).any())
        np.testing.assert_allclose(grown[active & limited] / base[active & limited], 3.0, rtol=1e-6)

    def test_torque_scaling_leaves_unlimited_actuators_alone(self):
        # The lift servo declares no torque limit, so there is no budget to scale.
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=3.0))
        base = saturated_actuator_force(self.model)
        grown = saturated_actuator_force(scaled)
        free = (np.abs(base) > 1e-9) & ~torque_limited_dofs(self.model)
        self.assertTrue(free.any())
        np.testing.assert_allclose(grown[free], base[free], rtol=1e-9)

    def test_control_ranges_scale_by_the_units_they_carry(self):
        params = morphology.MorphParams(size_scale=2.0, torque_scale=3.0)
        scaled = morphology.apply_morphology(self.model, params)
        hip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_servo")
        shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
        lift = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_servo")
        # A hinge servo commands an angle, which is dimensionless.
        np.testing.assert_allclose(
            scaled.actuator_ctrlrange[hip], self.model.actuator_ctrlrange[hip]
        )
        np.testing.assert_allclose(
            scaled.actuator_ctrlrange[shoulder], self.model.actuator_ctrlrange[shoulder] * 3.0
        )
        np.testing.assert_allclose(
            scaled.actuator_ctrlrange[lift], self.model.actuator_ctrlrange[lift] * 2.0
        )

    def test_joint_ranges_scale_only_for_slide_joints(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=2.0))
        hip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        lift = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift")
        np.testing.assert_allclose(scaled.jnt_range[hip], self.model.jnt_range[hip])
        np.testing.assert_allclose(scaled.jnt_range[lift], self.model.jnt_range[lift] * 2.0)

    def test_reference_configurations_scale_by_coordinate_type(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=2.0))
        # qpos layout: free translation, free quaternion, hip, shoulder, lift.
        np.testing.assert_allclose(scaled.key_qpos[0, :3], self.model.key_qpos[0, :3] * 2.0)
        np.testing.assert_allclose(scaled.key_qpos[0, 3:7], self.model.key_qpos[0, 3:7])
        np.testing.assert_allclose(scaled.key_qpos[0, 7:9], self.model.key_qpos[0, 7:9])
        np.testing.assert_allclose(scaled.key_qpos[0, 9], self.model.key_qpos[0, 9] * 2.0)
        np.testing.assert_allclose(scaled.qpos0[:3], self.model.qpos0[:3] * 2.0)

    def test_scaled_model_simulates_without_warnings(self):
        scaled = morphology.apply_morphology(self.model, morphology.dynamic_similarity_params(2.0))
        data = simulate(scaled)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertEqual(int(data.warning.number.sum()), 0)


class TestPlaygroundHumanoid(unittest.TestCase):
    """Checks scaling on a real mesh-based MuJoCo Playground humanoid."""

    @classmethod
    def setUpClass(cls):
        from mujoco_playground import registry

        cls.model = registry.load(PLAYGROUND_ENV).mj_model
        cls.extent = morphology.physical_extent(cls.model)
        cls.mass = morphology.total_mass(cls.model)

    def test_model_is_mesh_based(self):
        mesh_geoms = (self.model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)).sum()
        self.assertGreater(mesh_geoms, 0)
        self.assertGreater(self.model.nmeshvert, 0)

    def test_scaling_geom_size_alone_does_not_change_the_extent(self):
        faked = copy.deepcopy(self.model)
        faked.geom_size *= 2.0
        self.assertAlmostEqual(morphology.physical_extent(faked), self.extent, places=6)

    def test_size_scaling_changes_the_physical_extent(self):
        for factor in (0.5, 2.0, 3.0):
            scaled = morphology.apply_morphology(
                self.model, morphology.MorphParams(size_scale=factor)
            )
            self.assertAlmostEqual(
                morphology.physical_extent(scaled) / self.extent, factor, places=6
            )
            np.testing.assert_allclose(scaled.mesh_vert, self.model.mesh_vert * factor, rtol=1e-6)

    def test_mass_scaling_changes_the_total_mass_by_the_expected_factor(self):
        for factor in (0.5, 8.0):
            scaled = morphology.apply_morphology(
                self.model, morphology.MorphParams(mass_scale=factor)
            )
            self.assertAlmostEqual(morphology.total_mass(scaled) / self.mass, factor, places=6)

    def test_axes_are_independent(self):
        bigger = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=2.0))
        heavier = morphology.apply_morphology(self.model, morphology.MorphParams(mass_scale=2.0))
        self.assertAlmostEqual(morphology.total_mass(bigger), self.mass, places=6)
        self.assertAlmostEqual(morphology.physical_extent(heavier), self.extent, places=6)

    def test_inertia_tracks_mass_times_length_squared(self):
        params = morphology.MorphParams(size_scale=2.0, mass_scale=3.0)
        scaled = morphology.apply_morphology(self.model, params)
        np.testing.assert_allclose(
            scaled.body_inertia, self.model.body_inertia * 3.0 * 4.0, rtol=1e-9
        )

    def test_torque_scaling_changes_the_realised_joint_torques(self):
        # This humanoid limits torque on the joint, not on the actuator, so a
        # forcerange-only implementation would leave its torque budget untouched.
        self.assertFalse(self.model.actuator_forcelimited.any())
        self.assertTrue(self.model.jnt_actfrclimited.any())
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=4.0))
        base = saturated_actuator_force(self.model)
        grown = saturated_actuator_force(scaled)
        active = np.abs(base) > 1e-9
        self.assertTrue(active.any())
        np.testing.assert_allclose(grown[active] / base[active], 4.0, rtol=1e-6)

    def test_keyframe_stance_scales_with_the_robot(self):
        factor = 2.0
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(factor)
        )
        base = lowest_contact_height(self.model)
        grown = lowest_contact_height(scaled)
        self.assertAlmostEqual(grown / base, factor, places=5)

    def test_scaled_humanoid_simulates_without_warnings(self):
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(2.0)
        )
        data = simulate(scaled)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertEqual(int(data.warning.number.sum()), 0)

    def test_bounding_volumes_follow_the_geometry(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=2.0))
        np.testing.assert_allclose(
            scaled.geom_rbound, self.model.geom_rbound * 2.0, rtol=1e-9
        )
        finite = np.abs(self.model.bvh_aabb).max(axis=1) < morphology.MAX_MODEL_VALUE / 2.0
        np.testing.assert_allclose(
            scaled.bvh_aabb[finite], self.model.bvh_aabb[finite] * 2.0, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
