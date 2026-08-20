"""Growing and removing degrees of freedom by annealing how rigidly named joints are held."""

import contextlib
import copy
import dataclasses
import functools
import math
from collections.abc import Iterator, Sequence
from typing import Any

import mujoco
import numpy as np

from generalist_robotics.morphology.scaling import position_servo_actuators

# Square of (fastest lock frequency * simulation timestep). MuJoCo integrates a joint
# spring explicitly, so the springs a lock adds are a set of oscillators sampled at the
# simulation timestep, and the sampled oscillator is stable only while omega * dt < 2.
# The frequency that matters is the largest eigenvalue of inverse_inertia * stiffness and
# not stiffness / M_jj: the diagonal of the mass matrix overstates a joint's local
# inertia by 3x to 14x on G1, because a joint accelerates against its articulated inertia
# and not against the whole subtree, and springs on seventeen joints at once couple.
# lock_springs therefore normalises the whole lock by that eigenvalue.
# LOCK_RIGIDITY = 0.5 puts omega * dt at 0.71: half the largest value measured stable on
# this robot and a quarter of the smallest measured unstable one.
LOCK_RIGIDITY = 0.5

# Ceiling on (fastest lock damping rate * simulation timestep). These XMLs carry
# <flag eulerdamp="disable"/>, which takes joint damping out of the implicit part of the
# Euler step, so the lock's damping is explicit too and is stable only while that product
# stays below 2. Critical damping of the stiffness above asks for 2 * sqrt(LOCK_RIGIDITY)
# = 1.41, so this cap binds at full lock and holds the lock slightly under-damped rather
# than let the damping term be the thing that goes unstable.
LOCK_DAMPING_CEILING = 1.0

# Dynamic range of the lock -> stiffness anneal. It sets the curvature of that map, not
# its peak: the sweep is geometric, so lock = 0.5 lands a factor of sqrt(LOCK_RANGE)
# below full rigidity. A joint stops being absent and starts being a degree of freedom
# where its lock spring falls through the servo gain that drives it, which for G1's waist
# and arms is 7x to 164x below full rigidity, so LOCK_RANGE = 1e3 puts that hand-over in
# the middle of a unit path and staggers it across the joints: the elbows and shoulder
# yaws arrive first, then the shoulders and the waist, and the weak wrists last.
LOCK_RANGE = 1.0e3

# Damping of the lock spring, in units of its own critical damping. A rigid joint that
# rings is not rigid, so the lock is critically damped.
LOCK_DAMPING_RATIO = 1.0

# Fraction of its nominal commanded excursion a joint must retain before it is counted as
# a degree of freedom the policy actually has; see joint_control_authority.
ACTIVE_DOF_AUTHORITY = 0.5

# Name tokens that place a joint in a named group. Joint names are split into lowercase
# alphanumeric tokens and matched exactly, so "left_hip_pitch_joint" is a leg and
# Berkeley Humanoid's "LL_HAA" is one too. A robot whose XML does not name its joints
# descriptively needs its own tokens added here.
JOINT_GROUP_TOKENS: dict[str, tuple[str, ...]] = {
    "legs": ("hip", "knee", "ankle", "haa", "hfe", "hr", "kfe", "ffe", "faa"),
    "arms": ("shoulder", "elbow", "wrist", "arm"),
    "waist": ("waist", "torso", "trunk", "spine"),
    "neck": ("neck", "head"),
}

# Transmissions whose actuator_trnid names a joint directly.
JOINT_TRANSMISSIONS = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)

# Joint types that carry exactly one scalar coordinate, which is what a lock spring acts on.
SCALAR_JOINT_TYPES = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


@dataclasses.dataclass(frozen=True)
class DofLock:
    """How rigidly a named set of joints is held, on a 0 (free) to 1 (locked) scale.

    Attributes:
        joint_names: joints the lock applies to, by their MuJoCo names.
        lock: 1.0 holds the joints rigid, so the robot behaves as if they and their
            actuators were absent; 0.0 leaves the model untouched. Values in between
            interpolate geometrically in stiffness, see lock_stiffness_fraction.
    """

    joint_names: tuple[str, ...]
    lock: float

    def __post_init__(self):
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        object.__setattr__(self, "lock", float(self.lock))
        if not 0.0 <= self.lock <= 1.0:
            raise ValueError(f"lock must be in [0, 1], got {self.lock!r}")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"joint_names must not repeat, got {self.joint_names!r}")


def name_tokens(name: str) -> set[str]:
    """Split a joint name into the lowercase alphanumeric tokens a group is matched on."""
    tokens: list[str] = []
    current: list[str] = []
    for character in name.lower():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return set(tokens)


def group_joint_names(names: Sequence[str], group: str) -> tuple[str, ...]:
    """Select the joints of one named group out of a sequence of joint names.

    Args:
        names: joint names, in the order they should be returned.
        group: key of JOINT_GROUP_TOKENS, e.g. "legs", "arms", "waist" or "neck".

    Raises:
        ValueError: if the group is not a known one.
    """
    if group not in JOINT_GROUP_TOKENS:
        raise ValueError(
            f"Unknown joint group {group!r}. Known groups: {sorted(JOINT_GROUP_TOKENS)}"
        )
    tokens = set(JOINT_GROUP_TOKENS[group])
    return tuple(name for name in names if name_tokens(name) & tokens)


def actuated_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Names of the joints a model's actuators drive, in actuator order."""
    names: list[str] = []
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] not in JOINT_TRANSMISSIONS:
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name is not None:
            names.append(name)
    return tuple(names)


@functools.cache
def robot_model(robot: str) -> mujoco.MjModel:
    """The unmorphed, unlocked MuJoCo model of a robot, built once and kept.

    This is the reference every lock is sized and counted against, so that the same alpha
    means the same lock no matter which body along a path it is applied to.
    """
    # Imported here so that morphology does not depend on envs at module scope: the
    # dependency runs envs -> morphology, and this is the one place that needs it back.
    from generalist_robotics.envs.locomotion import make_locomotion_env

    return make_locomotion_env(robot).mj_model


def robot_joint_names(robot: str) -> tuple[str, ...]:
    """Names of the joints a robot's actuators drive, read off the unmorphed model."""
    return actuated_joint_names(robot_model(robot))


def joint_group(robot: str, group: str) -> tuple[str, ...]:
    """Named joint groups for a robot, e.g. 'legs', 'arms', 'waist', 'neck'.

    Args:
        robot: short robot name from envs.locomotion.available_robots.
        group: key of JOINT_GROUP_TOKENS.

    Returns:
        The actuated joints of that group, in actuator order; empty when the robot has
        none, as Berkeley Humanoid has no arms.
    """
    return group_joint_names(robot_joint_names(robot), group)


def lock_stiffness_fraction(lock: float) -> float:
    """Return the fraction of peak stiffness a lock factor asks for.

    The map is geometric over LOCK_RANGE decades and exactly zero at lock = 0, so
    annealing the lock down grows a degree of freedom continuously: stiffness falls
    through the servo gain that drives the joint somewhere in the middle of the anneal
    rather than in its last percent, and reaches free without a step.
    """
    if not 0.0 <= lock <= 1.0:
        raise ValueError(f"lock must be in [0, 1], got {lock!r}")
    return float(math.expm1(lock * math.log1p(LOCK_RANGE)) / LOCK_RANGE)


def joint_lock_factors(locks: Sequence[DofLock]) -> dict[str, float]:
    """Flatten locks into one lock factor per joint name, dropping the free ones.

    Joints named at lock 0 are kept, with a factor of 0: they are the joints this lock is
    about, and lock_springs needs the whole set to normalise the lock the same way at
    every point of an anneal.

    Raises:
        ValueError: if two locks disagree about the same joint.
    """
    factors: dict[str, float] = {}
    for entry in locks:
        for name in entry.joint_names:
            previous = factors.get(name)
            if previous is not None and previous != entry.lock:
                raise ValueError(
                    f"joint {name!r} is locked twice, at {previous!r} and {entry.lock!r}"
                )
            factors[name] = entry.lock
    return factors


def reference_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Pose a locked joint is held at: the model's last keyframe, or qpos0.

    Playground humanoids author their most specific standing pose last - G1's
    "knees_bent" after its "home" - and that is the pose their environment resets to and
    adds the policy's action to. Holding a locked joint there makes the lock free of
    charge in the reward: the pose and joint-limit penalties are all measured against the
    same pose, so a locked joint sits at their optimum instead of fighting them.
    """
    if int(model.nkey) > 0:
        return np.asarray(model.key_qpos[int(model.nkey) - 1]).copy()
    return np.asarray(model.qpos0).copy()


def joint_id(model: mujoco.MjModel, name: str) -> int:
    """Return a joint's id, raising a readable error when the model has no such joint."""
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if index < 0:
        raise ValueError(f"model has no joint named {name!r}")
    return int(index)


@dataclasses.dataclass(frozen=True)
class LockSpring:
    """The virtual spring one locked joint is held by.

    Attributes:
        stiffness: spring constant in generalized force per unit of the joint coordinate.
        damping: damping coefficient of that spring, added to the joint's own.
    """

    stiffness: float
    damping: float


def inverse_mass_matrix(model: mujoco.MjModel) -> np.ndarray:
    """Return the inverse of the mass matrix at the reference pose, armature included.

    Its diagonal entry at a degree of freedom is that joint's local inverse inertia: the
    acceleration one unit of generalized force there produces once the rest of the
    mechanism is free to respond. That is the inertia a joint spring is integrated
    against, and on G1 it is 3 to 14 times smaller than the mass-matrix diagonal.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = reference_qpos(model)
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return np.linalg.inv(full)


def coupled_spectral_radius(inverse: np.ndarray, diagonal: np.ndarray) -> float:
    """Largest eigenvalue of inverse @ diag(diagonal), for a positive semidefinite pair.

    The product is not symmetric, but it is similar to the symmetric positive
    semidefinite sqrt(diag) @ inverse @ sqrt(diag), so its spectrum is real, non-negative
    and can be read off that one.
    """
    root = np.sqrt(np.maximum(diagonal, 0.0))
    if not root.any():
        return 0.0
    symmetric = root[:, None] * inverse * root[None, :]
    return float(np.max(np.linalg.eigvalsh(symmetric)))


def lock_springs(
    model: mujoco.MjModel, locks: Sequence[DofLock], timestep: float | None = None
) -> dict[str, LockSpring]:
    """Return the spring each locked joint is held by, keyed by joint name.

    The shape of the lock across joints is set per joint - each one is stiffened in
    proportion to its own local inertia, so that every locked joint is equally rigid
    relative to what moves it - and the overall magnitude is then set once, by requiring
    that the fastest mode the lock adds to the mechanism satisfies (omega * dt)**2 =
    LOCK_RIGIDITY. That is a statement about the integrator, so it is the quantity that
    has to be bounded, and normalising the whole lock by it is what makes seventeen
    simultaneous locks as stable as one.

    Args:
        model: the robot the locks apply to.
        locks: how rigidly each group of joints is held.
        timestep: simulation timestep the model will be integrated at, which sets the
            stability ceiling; defaults to the model's own opt.timestep.

    Raises:
        ValueError: if a named joint is missing, carries more than one coordinate, or the
            timestep is not positive.
    """
    factors = joint_lock_factors(locks)
    if not factors:
        return {}
    dt = float(timestep if timestep is not None else model.opt.timestep)
    if dt <= 0.0:
        raise ValueError(f"timestep must be positive, got {dt!r}")

    inverse = inverse_mass_matrix(model)
    dofs: dict[str, int] = {}
    rigid = np.zeros(model.nv)
    for name in factors:
        joint = joint_id(model, name)
        if model.jnt_type[joint] not in SCALAR_JOINT_TYPES:
            raise ValueError(f"joint {name!r} is not a hinge or slide, so it cannot be locked")
        dof = int(model.jnt_dofadr[joint])
        dofs[name] = dof
        rigid[dof] = 1.0 / (inverse[dof, dof] * dt**2)

    # The normalisation is a property of the joint set, measured once at full lock, so
    # that annealing a lock down scales the springs it holds instead of renormalising
    # them back to the ceiling. Lowering any factor only lowers the spectral radius, so
    # the bound the full lock satisfies holds everywhere along the anneal.
    radius = coupled_spectral_radius(inverse, rigid)
    if radius <= 0.0:
        return {name: LockSpring(0.0, 0.0) for name in factors}
    scale = LOCK_RIGIDITY / (dt**2 * radius)
    stiffness = np.zeros(model.nv)
    for name, dof in dofs.items():
        stiffness[dof] = scale * rigid[dof] * lock_stiffness_fraction(factors[name])

    damping = np.zeros(model.nv)
    for dof in dofs.values():
        damping[dof] = 2.0 * LOCK_DAMPING_RATIO * math.sqrt(stiffness[dof] / inverse[dof, dof])

    damping_radius = coupled_spectral_radius(inverse, damping)
    if damping_radius * dt > LOCK_DAMPING_CEILING:
        damping *= LOCK_DAMPING_CEILING / (damping_radius * dt)

    return {
        name: LockSpring(stiffness=float(stiffness[dof]), damping=float(damping[dof]))
        for name, dof in dofs.items()
    }


def joint_actuators(model: mujoco.MjModel) -> dict[int, list[int]]:
    """Map each joint id to the ids of the actuators that drive it directly."""
    driven: dict[int, list[int]] = {}
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] not in JOINT_TRANSMISSIONS:
            continue
        driven.setdefault(int(model.actuator_trnid[actuator, 0]), []).append(actuator)
    return driven


def joint_position_gains(model: mujoco.MjModel) -> np.ndarray:
    """Return the position-servo gain acting on each joint, or 0 where no servo drives it."""
    gains = np.zeros(model.njnt)
    servo = position_servo_actuators(model)
    for joint, actuators in joint_actuators(model).items():
        gains[joint] = sum(abs(model.actuator_gainprm[a, 0]) for a in actuators if servo[a])
    return gains


def lock_joints(
    model: mujoco.MjModel, locks: tuple[DofLock, ...], timestep: float | None = None
) -> mujoco.MjModel:
    """Return a copy with the named joints stiffened toward rigidity by their lock factor.

    A lock is three consistent things at once, so that a fully locked joint is absent both
    mechanically and from the policy's reach:

      - a spring of lock_stiffnesses strength holding the joint at reference_qpos, which
        is what actually immobilises it;
      - the critical damping of that spring, added to the joint's own, so the rigid joint
        does not ring at the frequency the spring just gave it;
      - the actuator's gains scaled by 1 - lock, so a locked joint's action channel drives
        nothing and the policy's action space is effectively narrowed without changing
        its width.

    Joints whose lock is 0 are not written at all, so a fully unlocked model is a plain
    copy of its input, bit for bit.

    Args:
        model: base robot model.
        locks: how rigidly each group of joints is held.
        timestep: simulation timestep the model will be integrated at; defaults to the
            model's own opt.timestep.

    Returns:
        A locked deep copy of the model.

    Raises:
        ValueError: if a named joint is missing, carries more than one coordinate, or is
            locked twice at different factors.
    """
    locked = copy.deepcopy(model)
    factors = {name: lock for name, lock in joint_lock_factors(locks).items() if lock > 0.0}
    if not factors:
        return locked

    springs = lock_springs(model, locks, timestep)
    reference = reference_qpos(model)
    driven = joint_actuators(model)

    for name, lock in factors.items():
        if lock <= 0.0:
            continue
        joint = joint_id(model, name)
        address = int(model.jnt_qposadr[joint])
        dof = int(model.jnt_dofadr[joint])
        locked.jnt_stiffness[joint] = springs[name].stiffness
        locked.qpos_spring[address] = reference[address]
        locked.dof_damping[dof] += springs[name].damping
        for actuator in driven.get(joint, []):
            locked.actuator_gainprm[actuator, 0] *= 1.0 - lock
            locked.actuator_biasprm[actuator, 1] *= 1.0 - lock
            locked.actuator_biasprm[actuator, 2] *= 1.0 - lock
    return locked


def joint_control_authority(
    model: mujoco.MjModel, locks: Sequence[DofLock], timestep: float | None = None
) -> dict[str, float]:
    """Return the fraction of its nominal commanded excursion each actuated joint keeps.

    A position servo of gain kp asked to move a joint by one unit reaches, against a lock
    spring of stiffness k, only kp / (kp + k) of that unit, and the lock has already
    scaled the gain itself down to (1 - lock) * kp. The ratio of the two excursions is
    what a policy actually loses, and unlike the lock factor it is comparable across
    joints with very different servo gains and inertias.
    """
    springs = lock_springs(model, locks, timestep)
    factors = joint_lock_factors(locks)
    gains = joint_position_gains(model)
    authority: dict[str, float] = {}
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] not in JOINT_TRANSMISSIONS:
            continue
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name is None:
            continue
        lock = factors.get(name, 0.0)
        gain = (1.0 - lock) * gains[joint]
        held = springs[name].stiffness if name in springs else 0.0
        total = gain + held
        authority[name] = float(gain / total) if total > 0.0 else 1.0 - lock
    return authority


def active_dof_count(
    model: mujoco.MjModel, locks: Sequence[DofLock], timestep: float | None = None
) -> int:
    """Return how many actuated degrees of freedom the policy effectively still has.

    A joint counts when it keeps at least ACTIVE_DOF_AUTHORITY of the excursion the same
    action would produce on the unlocked robot, so the count is a statement about control
    authority rather than about how many joints the model happens to contain: locking
    G1's waist and arms rigid takes it from 29 to 12, the leg count of the Berkeley
    Humanoid, and unlocking them brings it back.
    """
    authority = joint_control_authority(model, locks, timestep)
    return int(sum(value >= ACTIVE_DOF_AUTHORITY for value in authority.values()))


def interpolate_locks(
    start: tuple[DofLock, ...], end: tuple[DofLock, ...], alpha: float
) -> tuple[DofLock, ...]:
    """Blend two lock specifications, linearly in the lock factor.

    The groups are matched by their joint names, and a group named on one side only is
    read as free on the other. Blending is linear because the lock factor is already a
    logarithmic coordinate on stiffness, see lock_stiffness_fraction, so a linear sweep of
    it is a geometric sweep of the spring that holds the joint.

    Args:
        start: locks at alpha = 0.
        end: locks at alpha = 1.
        alpha: blending coordinate, clamped to [0, 1] only in the sense that the result
            must stay a valid lock.

    Raises:
        ValueError: if alpha is not finite, or the blend leaves [0, 1].
    """
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    start_by_group = {entry.joint_names: entry.lock for entry in start}
    end_by_group = {entry.joint_names: entry.lock for entry in end}
    groups = list(start_by_group) + [g for g in end_by_group if g not in start_by_group]
    blended: list[DofLock] = []
    for group in groups:
        low = start_by_group.get(group, 0.0)
        high = end_by_group.get(group, 0.0)
        blended.append(DofLock(joint_names=group, lock=(1.0 - alpha) * low + alpha * high))
    return tuple(blended)


@contextlib.contextmanager
def locked_compilation(
    locks: tuple[DofLock, ...], timestep: float | None = None
) -> Iterator[list[mujoco.MjModel]]:
    """Patch MuJoCo XML compilation so every model compiled inside the block is locked.

    This mirrors envs.locomotion.morphed_compilation and composes with it: entering this
    block first and building a morphed environment inside it locks the freshly compiled
    robot and then morphs the locked model, so the lock spring and its damping are scaled
    by the morph exactly as any other passive joint property is. On a dynamic-similarity
    path that is what holds the lock's rigidity, and its stability margin, constant: the
    stiffness follows mass * size, the joint inertia mass * size**2 and the timestep
    sqrt(size), so stiffness * timestep**2 / inertia does not move.

    The patch is global for the duration of the block and is not thread safe.

    Args:
        locks: how rigidly each group of joints is held.
        timestep: simulation timestep the models will be integrated at; defaults to each
            model's own opt.timestep.

    Yields:
        The list of locked models handed out, so the caller can verify the lock landed.
    """
    produced: list[mujoco.MjModel] = []
    compile_original = mujoco.MjModel.from_xml_string

    def compile_locked(*args: Any, **kwargs: Any) -> mujoco.MjModel:
        model = lock_joints(compile_original(*args, **kwargs), tuple(locks), timestep)
        produced.append(model)
        return model

    mujoco.MjModel.from_xml_string = staticmethod(compile_locked)
    try:
        yield produced
    finally:
        mujoco.MjModel.from_xml_string = staticmethod(compile_original)
