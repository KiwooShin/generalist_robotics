"""Unit tests for annealing degrees of freedom in and out of a MuJoCo model."""

import math
import os
import unittest

import mujoco
import numpy as np

from generalist_robotics.morphology import topology

RUN_INTEGRATION = os.environ.get("GENROBO_SKIP_SLOW_TESTS", "0") != "1"

# A two-link arm on a fixed base: one joint named like a leg and one like an arm, both
# position servos, integrated the way the Playground humanoids are - Euler with the
# implicit damping flag off, which is what makes the lock's damping explicit and bounded.
SYNTHETIC_XML = """
<mujoco model="topology_test">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="Euler">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <body name="thigh" pos="0 0 1">
      <joint name="hip_pitch_joint" type="hinge" axis="0 1 0" range="-2 2" damping="0.1"/>
      <geom name="upper" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="1.5"/>
      <body name="forearm" pos="0 0 -0.3">
        <joint name="elbow_joint" type="hinge" axis="0 1 0" range="-2 2" damping="0.1"/>
        <geom name="lower" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.8"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_pitch_joint" joint="hip_pitch_joint" kp="30" dampratio="1"/>
    <position name="elbow_joint" joint="elbow_joint" kp="30" dampratio="1"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="0.2 0.4"/>
  </keyframe>
</mujoco>
"""

# Names that stand in for a whole ladder of robots without building any of them.
LADDER_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "LL_HAA",
    "LL_KFE",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "right_wrist_yaw_joint",
    "neck_pitch_joint",
)

# Peak of a sinusoidal drive, in radians, large enough to swing the free joint visibly.
DRIVE_AMPLITUDE = 0.8
DRIVE_FREQUENCY = 2.0


def synthetic_model() -> mujoco.MjModel:
    """Compile the two-link test arm."""
    return mujoco.MjModel.from_xml_string(SYNTHETIC_XML)


def drive(model: mujoco.MjModel, steps: int = 5000) -> tuple[np.ndarray, float]:
    """Swing every actuator about the reference pose and report what each joint did.

    Returns:
        The peak absolute excursion of every joint from the reference pose, and the
        largest absolute generalized acceleration seen, which is the witness that the
        explicit lock spring did not run away.
    """
    reference = topology.reference_qpos(model)
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    mujoco.mj_forward(model, data)
    peak = np.zeros(model.nq)
    worst = 0.0
    for step in range(steps):
        moment = step * model.opt.timestep
        data.ctrl[:] = reference + DRIVE_AMPLITUDE * math.sin(
            2.0 * math.pi * DRIVE_FREQUENCY * moment
        )
        mujoco.mj_step(model, data)
        peak = np.maximum(peak, np.abs(data.qpos - reference))
        worst = max(worst, float(np.max(np.abs(data.qacc))))
    return peak, worst


def lock_spectral_rigidity(model: mujoco.MjModel) -> float:
    """Return (fastest spring frequency * timestep)**2 for the springs a model carries."""
    stiffness = np.zeros(model.nv)
    for joint in range(model.njnt):
        stiffness[int(model.jnt_dofadr[joint])] = model.jnt_stiffness[joint]
    radius = topology.coupled_spectral_radius(topology.inverse_mass_matrix(model), stiffness)
    return radius * model.opt.timestep**2


class DofLockTest(unittest.TestCase):
    """A lock is a named set of joints and one number in [0, 1]."""

    def test_joint_names_become_a_tuple_of_strings(self):
        lock = topology.DofLock(joint_names=["a", "b"], lock=1)
        self.assertEqual(lock.joint_names, ("a", "b"))
        self.assertIsInstance(lock.lock, float)

    def test_a_lock_outside_the_unit_interval_is_rejected(self):
        for value in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                topology.DofLock(joint_names=("a",), lock=value)

    def test_a_repeated_joint_is_rejected(self):
        with self.assertRaises(ValueError):
            topology.DofLock(joint_names=("a", "a"), lock=1.0)


class JointGroupTest(unittest.TestCase):
    """Groups are read off joint names, token by token."""

    def test_legs_include_both_descriptive_and_abbreviated_names(self):
        self.assertEqual(
            topology.group_joint_names(LADDER_NAMES, "legs"),
            (
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_roll_joint",
                "LL_HAA",
                "LL_KFE",
            ),
        )

    def test_arms_waist_and_neck_are_disjoint_from_legs(self):
        arms = topology.group_joint_names(LADDER_NAMES, "arms")
        waist = topology.group_joint_names(LADDER_NAMES, "waist")
        neck = topology.group_joint_names(LADDER_NAMES, "neck")
        legs = topology.group_joint_names(LADDER_NAMES, "legs")
        self.assertEqual(len(arms), 3)
        self.assertEqual(waist, ("waist_yaw_joint",))
        self.assertEqual(neck, ("neck_pitch_joint",))
        self.assertEqual(set(arms) & set(legs) | set(waist) & set(legs), set())

    def test_a_robot_without_arms_reports_none(self):
        self.assertEqual(topology.group_joint_names(("LL_HAA", "LR_KFE"), "arms"), ())

    def test_an_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            topology.group_joint_names(LADDER_NAMES, "tail")

    def test_actuated_joint_names_follow_actuator_order(self):
        self.assertEqual(
            topology.actuated_joint_names(synthetic_model()),
            ("hip_pitch_joint", "elbow_joint"),
        )


class LockStiffnessFractionTest(unittest.TestCase):
    """The lock factor is a logarithmic coordinate on stiffness that reaches zero."""

    def test_the_ends_are_exact(self):
        self.assertEqual(topology.lock_stiffness_fraction(0.0), 0.0)
        self.assertAlmostEqual(topology.lock_stiffness_fraction(1.0), 1.0)

    def test_it_is_monotone(self):
        values = [topology.lock_stiffness_fraction(x) for x in np.linspace(0.0, 1.0, 21)]
        self.assertEqual(values, sorted(values))

    def test_the_midpoint_is_the_geometric_one(self):
        midpoint = topology.lock_stiffness_fraction(0.5)
        self.assertAlmostEqual(
            midpoint, (math.sqrt(1.0 + topology.LOCK_RANGE) - 1.0) / topology.LOCK_RANGE
        )
        self.assertLess(abs(midpoint * math.sqrt(topology.LOCK_RANGE) - 1.0), 0.05)

    def test_a_factor_outside_the_unit_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            topology.lock_stiffness_fraction(1.5)


class JointLockFactorsTest(unittest.TestCase):
    """Locks flatten to one factor per joint, free joints included."""

    def test_groups_merge(self):
        locks = (topology.DofLock(("a", "b"), 1.0), topology.DofLock(("c",), 0.25))
        self.assertEqual(topology.joint_lock_factors(locks), {"a": 1.0, "b": 1.0, "c": 0.25})

    def test_a_free_group_is_kept_so_the_joint_set_does_not_change(self):
        locks = (topology.DofLock(("a",), 0.0),)
        self.assertEqual(topology.joint_lock_factors(locks), {"a": 0.0})

    def test_two_locks_disagreeing_about_one_joint_are_rejected(self):
        locks = (topology.DofLock(("a",), 1.0), topology.DofLock(("a",), 0.5))
        with self.assertRaises(ValueError):
            topology.joint_lock_factors(locks)


class InterpolateLocksTest(unittest.TestCase):
    """Blending lock states is linear per group and keeps the grouping."""

    def setUp(self):
        self.start = (topology.DofLock(("a", "b"), 1.0), topology.DofLock(("c",), 0.5))
        self.end = (topology.DofLock(("a", "b"), 0.0), topology.DofLock(("c",), 0.0))

    def test_the_ends_are_reproduced(self):
        self.assertEqual(topology.interpolate_locks(self.start, self.end, 0.0), self.start)
        self.assertEqual(topology.interpolate_locks(self.start, self.end, 1.0), self.end)

    def test_the_middle_is_the_average_of_each_group(self):
        blended = topology.interpolate_locks(self.start, self.end, 0.5)
        self.assertEqual([lock.lock for lock in blended], [0.5, 0.25])
        self.assertEqual([lock.joint_names for lock in blended], [("a", "b"), ("c",)])

    def test_a_group_named_on_one_side_only_is_free_on_the_other(self):
        blended = topology.interpolate_locks((topology.DofLock(("a",), 1.0),), (), 0.25)
        self.assertEqual(blended, (topology.DofLock(("a",), 0.75),))

    def test_a_non_finite_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            topology.interpolate_locks(self.start, self.end, float("nan"))


class LockedModelTest(unittest.TestCase):
    """What a lock writes into the model, and what it leaves alone."""

    def setUp(self):
        self.model = synthetic_model()
        self.elbow = topology.joint_id(self.model, "elbow_joint")
        self.hip = topology.joint_id(self.model, "hip_pitch_joint")
        self.locked = topology.lock_joints(self.model, (topology.DofLock(("elbow_joint",), 1.0),))

    def test_a_fully_unlocked_model_is_identical_to_the_original(self):
        free = topology.lock_joints(self.model, (topology.DofLock(("elbow_joint",), 0.0),))
        for field in (
            "jnt_stiffness",
            "dof_damping",
            "qpos_spring",
            "actuator_gainprm",
            "actuator_biasprm",
            "dof_armature",
            "body_mass",
            "jnt_range",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(free, field)), np.asarray(getattr(self.model, field)), field
            )

    def test_the_locked_joint_gains_a_spring_and_the_free_one_does_not(self):
        self.assertGreater(self.locked.jnt_stiffness[self.elbow], 0.0)
        self.assertEqual(self.locked.jnt_stiffness[self.hip], self.model.jnt_stiffness[self.hip])

    def test_the_spring_holds_the_reference_pose(self):
        address = int(self.model.jnt_qposadr[self.elbow])
        self.assertEqual(
            self.locked.qpos_spring[address], topology.reference_qpos(self.model)[address]
        )

    def test_the_spring_is_damped(self):
        dof = int(self.model.jnt_dofadr[self.elbow])
        self.assertGreater(self.locked.dof_damping[dof], self.model.dof_damping[dof])

    def test_the_locked_actuator_loses_all_of_its_gain(self):
        self.assertEqual(self.locked.actuator_gainprm[1, 0], 0.0)
        self.assertEqual(self.locked.actuator_biasprm[1, 1], 0.0)
        self.assertEqual(self.locked.actuator_biasprm[1, 2], 0.0)
        self.assertEqual(self.locked.actuator_gainprm[0, 0], self.model.actuator_gainprm[0, 0])

    def test_the_reference_pose_is_the_last_keyframe(self):
        np.testing.assert_array_equal(
            topology.reference_qpos(self.model), np.asarray(self.model.key_qpos[-1])
        )

    def test_the_lock_sits_exactly_on_its_stability_budget(self):
        self.assertAlmostEqual(
            lock_spectral_rigidity(self.locked), topology.LOCK_RIGIDITY, places=9
        )

    def test_annealing_the_lock_lowers_the_spring_geometrically(self):
        springs = [
            topology.lock_springs(self.model, (topology.DofLock(("elbow_joint",), lock),))[
                "elbow_joint"
            ].stiffness
            for lock in (1.0, 0.5, 0.0)
        ]
        self.assertGreater(springs[0], springs[1])
        self.assertGreater(springs[1], 0.0)
        self.assertEqual(springs[2], 0.0)
        self.assertAlmostEqual(
            springs[1] / springs[0], topology.lock_stiffness_fraction(0.5), places=9
        )

    def test_the_anneal_never_exceeds_the_budget(self):
        for lock in np.linspace(0.0, 1.0, 11):
            model = topology.lock_joints(self.model, (topology.DofLock(("elbow_joint",), lock),))
            self.assertLessEqual(lock_spectral_rigidity(model), topology.LOCK_RIGIDITY + 1e-9)

    def test_an_unknown_joint_is_rejected(self):
        with self.assertRaises(ValueError):
            topology.lock_joints(self.model, (topology.DofLock(("tail_joint",), 1.0),))

    def test_the_input_model_is_left_alone(self):
        self.assertEqual(float(self.model.jnt_stiffness[self.elbow]), 0.0)


class ImmobilisationTest(unittest.TestCase):
    """A locked joint stops moving, an unlocked one moves exactly as it always did."""

    def setUp(self):
        self.model = synthetic_model()
        self.locked = topology.lock_joints(self.model, (topology.DofLock(("elbow_joint",), 1.0),))
        self.elbow = int(self.model.jnt_qposadr[topology.joint_id(self.model, "elbow_joint")])
        self.hip = int(self.model.jnt_qposadr[topology.joint_id(self.model, "hip_pitch_joint")])

    def test_the_locked_joint_holds_still_while_the_free_one_swings(self):
        nominal, _ = drive(self.model)
        locked, _ = drive(self.locked)
        self.assertGreater(nominal[self.elbow], 0.5)
        self.assertGreater(locked[self.hip], 0.3)
        self.assertLess(locked[self.elbow], 0.01)
        self.assertGreater(nominal[self.elbow] / locked[self.elbow], 50.0)

    def test_the_locked_model_stays_numerically_sane(self):
        _, worst_locked = drive(self.locked, steps=20000)
        _, worst_nominal = drive(self.model, steps=20000)
        self.assertTrue(math.isfinite(worst_locked))
        self.assertLess(worst_locked, 100.0 * worst_nominal)

    def test_unlocking_restores_the_original_trajectory(self):
        free = topology.lock_joints(self.model, (topology.DofLock(("elbow_joint",), 0.0),))
        np.testing.assert_array_equal(drive(free)[0], drive(self.model)[0])


class ActiveDofTest(unittest.TestCase):
    """The degree-of-freedom count follows control authority, not the joint list."""

    def setUp(self):
        self.model = synthetic_model()

    def locks(self, lock: float) -> tuple[topology.DofLock, ...]:
        return (topology.DofLock(("elbow_joint",), lock),)

    def test_a_rigid_joint_does_not_count(self):
        self.assertEqual(topology.active_dof_count(self.model, self.locks(1.0)), 1)

    def test_a_free_joint_counts(self):
        self.assertEqual(topology.active_dof_count(self.model, self.locks(0.0)), 2)
        self.assertEqual(topology.active_dof_count(self.model, ()), 2)

    def test_the_count_never_falls_as_the_lock_is_annealed_away(self):
        counts = [
            topology.active_dof_count(self.model, self.locks(lock))
            for lock in np.linspace(1.0, 0.0, 21)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual((counts[0], counts[-1]), (1, 2))

    def test_authority_rises_monotonically_as_the_lock_falls(self):
        authority = [
            topology.joint_control_authority(self.model, self.locks(lock))["elbow_joint"]
            for lock in np.linspace(1.0, 0.0, 21)
        ]
        self.assertEqual(authority, sorted(authority))
        self.assertLess(authority[0], 0.2)
        self.assertAlmostEqual(authority[-1], 1.0)


@unittest.skipUnless(RUN_INTEGRATION, "GENROBO_SKIP_SLOW_TESTS=1 disables building real robots")
class RealRobotGroupTest(unittest.TestCase):
    """The groups the milestone depends on are the ones G1 and Berkeley Humanoid have."""

    def test_g1_splits_into_twelve_legs_fourteen_arms_and_three_waist_joints(self):
        self.assertEqual(len(topology.joint_group("g1", "legs")), 12)
        self.assertEqual(len(topology.joint_group("g1", "arms")), 14)
        self.assertEqual(len(topology.joint_group("g1", "waist")), 3)
        self.assertEqual(topology.joint_group("g1", "neck"), ())

    def test_berkeley_humanoid_is_twelve_leg_joints_and_nothing_else(self):
        self.assertEqual(len(topology.joint_group("berkeley_humanoid", "legs")), 12)
        self.assertEqual(topology.joint_group("berkeley_humanoid", "arms"), ())

    def test_locking_g1s_waist_and_arms_leaves_berkeleys_twelve_degrees_of_freedom(self):
        locks = (
            topology.DofLock(topology.joint_group("g1", "waist"), 1.0),
            topology.DofLock(topology.joint_group("g1", "arms"), 1.0),
        )
        model = topology.robot_model("g1")
        self.assertEqual(topology.active_dof_count(model, locks), 12)
        self.assertEqual(topology.active_dof_count(model, ()), 29)
