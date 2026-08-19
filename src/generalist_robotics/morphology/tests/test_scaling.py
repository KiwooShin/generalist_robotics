"""Unit tests for parametric morphology scaling of MuJoCo models."""

import copy
import dataclasses
import functools
import unittest

import mujoco
import numpy as np

from generalist_robotics.morphology import scaling as morphology

# A small robot that exercises every path: a mesh geom, a free joint, hinge and
# slide joints, a joint-level torque limit, passive damping, dry friction and a
# joint spring, a position servo with a velocity gain, and a direct-drive motor.
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
               range="-1 1" actuatorfrcrange="-7 7" armature="0.01"
               damping="0.4" frictionloss="0.05"/>
        <geom name="leg" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/>
      </body>
      <body name="arm" pos="0.1 0 0.1">
        <joint name="shoulder" type="hinge" axis="1 0 0" range="-2 2" stiffness="1.5"/>
        <geom name="forearm" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.4"/>
      </body>
      <body name="slider" pos="-0.1 0 0">
        <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.2" armature="0.5"
               damping="2" frictionloss="0.1" stiffness="3"/>
        <geom name="pad" type="sphere" size="0.04" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_servo" joint="hip" kp="20" kv="2" ctrlrange="-1 1"/>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-5 5" forcerange="-6 6"/>
    <position name="lift_servo" joint="lift" kp="100" ctrlrange="-0.2 0.2"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0 0.3 -0.4 0.05"/>
  </keyframe>
</mujoco>
"""

PLAYGROUND_ENV = "BerkeleyHumanoidJoystickFlatTerrain"

# 400 steps is 0.8 s of base-robot time. Over that window the Berkeley humanoid sags out
# of its keyframe stance under gravity with four to eight contacts live at every step, so
# the comparison exercises contact, actuation and inertia rather than free flight.
SIMILARITY_STEPS = 400

# The scaled models reproduce the base trajectory to floating-point roundoff, ~1e-15 rad
# over that window. The assertion is loosened to 1e-6 so that a different MuJoCo or BLAS
# build cannot fail it on summation order alone; that is still five orders of magnitude
# below the 0.098 m of height and 0.31 rad of joint travel the base robot covers, so a
# real break in similarity cannot hide under it.
SIMILARITY_TOLERANCE = 1e-6

# Reviewer's torque_scale reproduction: a size 2, mass 8 robot stepped in MJX. Thirty
# steps is enough for the actuators to separate the trajectories, and short enough that
# one compile dominates the cost.
MJX_TORQUE_STEPS = 30
MJX_TORQUE_TOLERANCE = 1e-3


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


def hinge_dofs(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of the degrees of freedom belonging to hinge joints."""
    mask = np.zeros(model.nv, dtype=bool)
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
            mask[model.jnt_dofadr[joint]] = True
    return mask


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


def trajectory(model: mujoco.MjModel, steps: int, time_scale: float = 1.0) -> np.ndarray:
    """Return the qpos trajectory from the keyframe, on a clock stretched by time_scale."""
    if time_scale != 1.0:
        model = copy.deepcopy(model)
        model.opt.timestep *= time_scale
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    data.ctrl[:] = model.key_ctrl[0]
    frames = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        frames.append(data.qpos.copy())
    return np.array(frames)


@functools.cache
def jitted_mjx_step():
    """Return a jitted MJX step, compiled once per process and shared by every model."""
    import jax
    from mujoco import mjx

    return jax.jit(mjx.step)


def mjx_qvel_after(model: mujoco.MjModel, steps: int) -> np.ndarray:
    """Return qvel after stepping a model in MJX from its keyframe holding keyframe controls."""
    import jax.numpy as jnp
    from mujoco import mjx

    mjx_model = mjx.put_model(model)
    state = mjx.make_data(mjx_model).replace(
        qpos=jnp.asarray(model.key_qpos[0]), ctrl=jnp.asarray(model.key_ctrl[0])
    )
    step = jitted_mjx_step()
    for _ in range(steps):
        state = step(mjx_model, state)
    return np.asarray(state.qvel)


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
        with self.assertRaises(dataclasses.FrozenInstanceError):
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

    def test_torque_is_the_product_of_the_mass_and_size_axes(self):
        # A generalized torque is inertia * angle / time**2 = (m k**2) k**-1, so the
        # torque axis of a similar robot is exactly mass_scale * size_scale.
        for factor in (0.5, 2.0, 3.0):
            self.assertAlmostEqual(
                morphology.similar_torque_scale(factor),
                morphology.similar_mass_scale(factor) * factor,
            )


class TestInterpolate(unittest.TestCase):
    """Checks blending between two morphologies."""

    def test_endpoints_are_reproduced(self):
        start = morphology.MorphParams(1.0, 1.0, 1.0)
        end = morphology.dynamic_similarity_params(2.0)
        self.assertEqual(morphology.interpolate(start, end, 0.0), start)
        for a, b in zip(
            morphology.interpolate(start, end, 1.0).__dict__.values(),
            end.__dict__.values(),
            strict=True,
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

    def dof_of(self, joint_name):
        """Return the degree-of-freedom index of a named joint."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_dofadr[joint])

    def actuator_of(self, actuator_name):
        """Return the index of a named actuator."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def test_input_model_is_not_mutated(self):
        before = {
            name: getattr(self.model, name).copy()
            for name in (
                "body_pos",
                "body_mass",
                "body_inertia",
                "mesh_vert",
                "jnt_actfrcrange",
                "dof_damping",
                "actuator_gainprm",
                "geom_solref",
            )
        }
        morphology.apply_morphology(self.model, morphology.dynamic_similarity_params(2.5))
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(self.model, name), value)

    def test_identity_params_reproduce_the_model(self):
        same = morphology.apply_morphology(self.model, morphology.MorphParams())
        compared = 0
        for name in dir(self.model):
            if name.startswith("_"):
                continue
            original, copied = getattr(self.model, name), getattr(same, name)
            if not isinstance(original, np.ndarray) or not isinstance(copied, np.ndarray):
                continue
            np.testing.assert_array_equal(copied, original, err_msg=f"{name} changed")
            compared += 1
        self.assertGreater(compared, 50)

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
                getattr(scaled, name),
                getattr(self.model, name) * 3.0,
                rtol=1e-9,
                atol=1e-12,
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
        hip, lift = self.dof_of("hip"), self.dof_of("lift")
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

    def test_passive_joint_terms_follow_size_and_mass(self):
        size, mass = 2.0, 5.0
        scaled = morphology.apply_morphology(
            self.model, morphology.MorphParams(size_scale=size, mass_scale=mass)
        )
        hip, lift = self.dof_of("hip"), self.dof_of("lift")
        # A hinge coordinate is dimensionless: damping is mass * size**1.5, dry friction
        # is a torque, mass * size. A slide coordinate is a length, which drops one power
        # of size from each: mass / sqrt(size) and mass.
        self.assertAlmostEqual(
            scaled.dof_damping[hip], self.model.dof_damping[hip] * mass * size**1.5
        )
        self.assertAlmostEqual(
            scaled.dof_damping[lift], self.model.dof_damping[lift] * mass / size**0.5
        )
        self.assertAlmostEqual(
            scaled.dof_frictionloss[hip], self.model.dof_frictionloss[hip] * mass * size
        )
        self.assertAlmostEqual(
            scaled.dof_frictionloss[lift], self.model.dof_frictionloss[lift] * mass
        )

    def test_joint_springs_follow_size_and_mass(self):
        size, mass = 2.0, 5.0
        scaled = morphology.apply_morphology(
            self.model, morphology.MorphParams(size_scale=size, mass_scale=mass)
        )
        shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
        lift = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift")
        self.assertAlmostEqual(
            scaled.jnt_stiffness[shoulder], self.model.jnt_stiffness[shoulder] * mass * size
        )
        self.assertAlmostEqual(
            scaled.jnt_stiffness[lift], self.model.jnt_stiffness[lift] * mass / size
        )

    def test_passive_joint_terms_ignore_the_torque_axis(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=6.0))
        np.testing.assert_array_equal(scaled.dof_damping, self.model.dof_damping)
        np.testing.assert_array_equal(scaled.dof_frictionloss, self.model.dof_frictionloss)
        np.testing.assert_array_equal(scaled.jnt_stiffness, self.model.jnt_stiffness)

    def test_servo_gains_carry_the_torque_axis_and_the_clock(self):
        size, torque = 2.0, 3.0
        scaled = morphology.apply_morphology(
            self.model, morphology.MorphParams(size_scale=size, torque_scale=torque)
        )
        hip, lift = self.actuator_of("hip_servo"), self.actuator_of("lift_servo")
        # A hinge servo commands an angle, so its position gain is a torque and its
        # velocity gain is a torque per angular velocity, one clock factor larger.
        self.assertAlmostEqual(
            scaled.actuator_gainprm[hip, 0], self.model.actuator_gainprm[hip, 0] * torque
        )
        self.assertAlmostEqual(
            scaled.actuator_biasprm[hip, 1], self.model.actuator_biasprm[hip, 1] * torque
        )
        self.assertAlmostEqual(
            scaled.actuator_biasprm[hip, 2],
            self.model.actuator_biasprm[hip, 2] * torque * size**0.5,
        )
        # A slide servo commands a length, so both of its gains lose a power of size.
        self.assertAlmostEqual(
            scaled.actuator_gainprm[lift, 0], self.model.actuator_gainprm[lift, 0] * torque / size
        )
        self.assertAlmostEqual(
            scaled.actuator_biasprm[lift, 1], self.model.actuator_biasprm[lift, 1] * torque / size
        )

    def test_direct_drive_gain_is_left_to_the_control_range(self):
        # A motor's ctrl is already a torque, so its strength lives in the range and the
        # limit rather than in a gain that would double count the torque axis.
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=3.0))
        shoulder = self.actuator_of("shoulder_motor")
        self.assertAlmostEqual(
            scaled.actuator_gainprm[shoulder, 0], self.model.actuator_gainprm[shoulder, 0]
        )
        np.testing.assert_allclose(
            scaled.actuator_ctrlrange[shoulder], self.model.actuator_ctrlrange[shoulder] * 3.0
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

    def test_torque_scaling_reaches_actuators_that_declare_no_limit(self):
        # The lift servo has no torque limit, so before the gains were scaled its realised
        # force was invariant and the torque axis was inert wherever the limit did not bind.
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(torque_scale=3.0))
        base = saturated_actuator_force(self.model)
        grown = saturated_actuator_force(scaled)
        free = (np.abs(base) > 1e-9) & ~torque_limited_dofs(self.model)
        self.assertTrue(free.any())
        np.testing.assert_allclose(grown[free] / base[free], 3.0, rtol=1e-6)

    def test_torque_scaling_moves_stepped_dynamics_in_both_directions(self):
        reference = simulate(self.model, steps=100)
        for factor in (0.5, 2.0):
            scaled = morphology.apply_morphology(
                self.model, morphology.MorphParams(torque_scale=factor)
            )
            moved = simulate(scaled, steps=100)
            self.assertGreater(np.abs(moved.qvel - reference.qvel).max(), 1e-3)

    def test_control_ranges_scale_by_the_units_they_carry(self):
        params = morphology.MorphParams(size_scale=2.0, torque_scale=3.0)
        scaled = morphology.apply_morphology(self.model, params)
        hip = self.actuator_of("hip_servo")
        shoulder = self.actuator_of("shoulder_motor")
        lift = self.actuator_of("lift_servo")
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

    def test_constraint_solver_constants_follow_the_clock_and_the_geometry(self):
        size = 4.0
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=size))
        np.testing.assert_allclose(
            scaled.geom_solref[:, 0], self.model.geom_solref[:, 0] * size**0.5, rtol=1e-9
        )
        np.testing.assert_allclose(scaled.geom_solref[:, 1], self.model.geom_solref[:, 1])
        np.testing.assert_allclose(
            scaled.geom_solimp[:, 2], self.model.geom_solimp[:, 2] * size, rtol=1e-9
        )
        # A hinge limit is violated in radians, so its impedance width is dimensionless.
        hinge = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        slide = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift")
        self.assertAlmostEqual(
            scaled.jnt_solimp[hinge, 2], self.model.jnt_solimp[hinge, 2], places=12
        )
        self.assertAlmostEqual(scaled.jnt_solimp[slide, 2], self.model.jnt_solimp[slide, 2] * size)

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

    def test_similarity_exponents_on_the_real_robot(self):
        factor = 2.0
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(factor)
        )
        hinge = hinge_dofs(self.model)
        self.assertTrue(hinge.any())
        for field, power in (
            ("actuator_gainprm", 4.0),
            ("jnt_actfrcrange", 4.0),
        ):
            np.testing.assert_allclose(
                getattr(scaled, field), getattr(self.model, field) * factor**power, rtol=1e-9
            )
        np.testing.assert_allclose(
            scaled.actuator_biasprm[:, 1], self.model.actuator_biasprm[:, 1] * factor**4, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled.dof_damping[hinge], self.model.dof_damping[hinge] * factor**4.5, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled.dof_frictionloss[hinge],
            self.model.dof_frictionloss[hinge] * factor**4,
            rtol=1e-9,
        )

    def test_keyframe_stance_scales_with_the_robot(self):
        factor = 2.0
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(factor)
        )
        base = lowest_contact_height(self.model)
        grown = lowest_contact_height(scaled)
        self.assertAlmostEqual(grown / base, factor, places=5)

    def test_scaled_humanoid_simulates_without_warnings(self):
        scaled = morphology.apply_morphology(self.model, morphology.dynamic_similarity_params(2.0))
        data = simulate(scaled)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertEqual(int(data.warning.number.sum()), 0)

    def test_bounding_volumes_follow_the_geometry(self):
        scaled = morphology.apply_morphology(self.model, morphology.MorphParams(size_scale=2.0))
        np.testing.assert_allclose(scaled.geom_rbound, self.model.geom_rbound * 2.0, rtol=1e-9)
        finite = np.abs(self.model.bvh_aabb).max(axis=1) < morphology.MAX_MODEL_VALUE / 2.0
        np.testing.assert_allclose(
            scaled.bvh_aabb[finite], self.model.bvh_aabb[finite] * 2.0, rtol=1e-6
        )


class TestDynamicSimilarity(unittest.TestCase):
    """Checks that dynamic_similarity_params really produces a dynamically similar robot.

    A robot scaled by k and stepped on a clock stretched by sqrt(k) must trace the base
    robot's trajectory once the scaling is undone: heights divided by k, joint angles and
    body orientation unchanged.
    """

    @classmethod
    def setUpClass(cls):
        from mujoco_playground import registry

        cls.model = registry.load(PLAYGROUND_ENV).mj_model
        cls.base = trajectory(cls.model, SIMILARITY_STEPS)

    def test_the_base_trajectory_is_not_trivial(self):
        # Without this the similarity assertions could pass on a robot that never moves.
        height_travel = np.abs(self.base[:, 2] - self.base[0, 2]).max()
        joint_travel = np.abs(self.base[:, 7:] - self.base[0, 7:]).max()
        self.assertGreater(height_travel, 0.05)
        self.assertGreater(joint_travel, 0.1)

    def test_scaled_robot_reproduces_the_base_trajectory(self):
        for factor in (0.5, 1.5, 2.0, 3.0):
            with self.subTest(size_scale=factor):
                scaled = morphology.apply_morphology(
                    self.model, morphology.dynamic_similarity_params(factor)
                )
                rolled = trajectory(
                    scaled, SIMILARITY_STEPS, time_scale=morphology.similar_time_scale(factor)
                )
                np.testing.assert_allclose(
                    rolled[:, :3] / factor, self.base[:, :3], atol=SIMILARITY_TOLERANCE
                )
                np.testing.assert_allclose(
                    rolled[:, 3:7], self.base[:, 3:7], atol=SIMILARITY_TOLERANCE
                )
                np.testing.assert_allclose(
                    rolled[:, 7:], self.base[:, 7:], atol=SIMILARITY_TOLERANCE
                )

    def test_realised_torques_scale_as_the_fourth_power(self):
        factor = 2.0
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(factor)
        )
        scaled.opt.timestep *= morphology.similar_time_scale(factor)
        base_torque = np.abs(simulate(self.model, SIMILARITY_STEPS).actuator_force).max()
        scaled_torque = np.abs(simulate(scaled, SIMILARITY_STEPS).actuator_force).max()
        self.assertAlmostEqual(scaled_torque / base_torque, factor**4, places=2)

    def test_similarity_needs_the_stretched_clock(self):
        # The morphology alone is not enough: on the base robot's clock the scaled robot
        # is observed at the wrong rate and the trajectories separate. Without this the
        # similarity assertion could be satisfied by a scaling that does nothing at all.
        factor = 2.0
        scaled = morphology.apply_morphology(
            self.model, morphology.dynamic_similarity_params(factor)
        )
        rolled = trajectory(scaled, SIMILARITY_STEPS)
        self.assertGreater(np.abs(rolled[:, 7:] - self.base[:, 7:]).max(), 0.01)


class TestTorqueAxisInMjx(unittest.TestCase):
    """Checks that torque_scale moves the dynamics MJX integrates, not just a stored limit."""

    @classmethod
    def setUpClass(cls):
        from mujoco_playground import registry

        base = registry.load(PLAYGROUND_ENV).mj_model
        cls.qvel = {}
        for factor in (1.0, 2.0, 4.0, 16.0):
            params = morphology.MorphParams(size_scale=2.0, mass_scale=8.0, torque_scale=factor)
            model = morphology.apply_morphology(base, params)
            cls.qvel[factor] = mjx_qvel_after(model, MJX_TORQUE_STEPS)

    def test_stepped_dynamics_separate_between_neighbouring_torque_scales(self):
        for weaker, stronger in ((1.0, 2.0), (2.0, 4.0)):
            with self.subTest(weaker=weaker, stronger=stronger):
                difference = np.abs(self.qvel[weaker] - self.qvel[stronger]).max()
                self.assertGreater(difference, MJX_TORQUE_TOLERANCE)

    def test_the_axis_keeps_moving_where_the_limit_no_longer_binds(self):
        # A limit-only torque axis saturates: at size 2 and mass 8 the realised servo
        # torque sat far below the limit, so 4x and 16x used to step bit-identically.
        difference = np.abs(self.qvel[4.0] - self.qvel[16.0]).max()
        self.assertGreater(difference, MJX_TORQUE_TOLERANCE)

    def test_the_axis_is_monotone_away_from_the_similar_morphology(self):
        # torque_scale 16 is the dynamically similar actuation at size 2 and mass 8, so
        # the distance to it must shrink as the torque axis climbs towards it.
        distances = [
            np.abs(self.qvel[factor] - self.qvel[16.0]).max() for factor in (1.0, 2.0, 4.0)
        ]
        self.assertGreater(distances[0], distances[1])
        self.assertGreater(distances[1], distances[2])


if __name__ == "__main__":
    unittest.main()
