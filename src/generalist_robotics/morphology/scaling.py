"""Parametric morphology scaling of MuJoCo models along independent size, mass and torque axes."""

import copy
import dataclasses

import mujoco
import numpy as np

# MuJoCo treats magnitudes at or above mjMAXVAL as infinite; scaling must not overflow it.
MAX_MODEL_VALUE = mujoco.mjMAXVAL

# Power of length in the gravitational time scale sqrt(length / gravity): with gravity
# fixed, a robot k times longer runs sqrt(k) times slower. Every rate follows from it.
TIME_LENGTH_POWER = 0.5

# Similarity bookkeeping, writing s = size_scale, m = mass_scale, t = torque_scale and p
# for the power of length in a degree of freedom's generalized inertia (2 for rotation,
# 0 for translation; see dof_inertia_length_power). Its coordinate scales as s**(1 - p/2),
# so with time as above:
#   generalized inertia   m * s**p
#   generalized force     m * s**(p/2)        inertia * coordinate / time**2
#   damping coefficient   m * s**(p - 0.5)    force / (coordinate / time)
#   spring stiffness      m * s**(p - 1)      force / coordinate
# Actuator strength is the torque axis rather than a consequence of geometry, so an
# actuator's generalized force scales as t alone and its gains only convert t into the
# units of the signal they multiply: t * s**-c for a position gain and t * s**(0.5 - c)
# for a velocity gain, where c = 1 - p/2 is the power of length in the coordinate driven.
# At dynamic similarity (m = s**3, t = s**4) this reproduces the textbook exponents for a
# hinge: torque s**4, position gain s**4, velocity gain and joint damping s**4.5, dry
# friction s**4 - and it keeps the size, mass and torque axes from double counting s**4.

# Fields whose entries are plain lengths, scaled by size_scale.
LENGTH_FIELDS = (
    "body_pos",
    "body_ipos",
    "jnt_pos",
    "jnt_margin",
    "geom_pos",
    "geom_size",
    "geom_rbound",
    "geom_margin",
    "geom_gap",
    "site_pos",
    "site_size",
    "cam_pos",
    "cam_ipd",
    "light_pos",
    "light_bulbradius",
    "mesh_vert",
    "mesh_pos",
    "flex_vert",
    "flex_node",
    "flex_radius",
    "hfield_size",
    "tendon_range",
    "tendon_margin",
    "tendon_length0",
    "tendon_lengthspring",
    "actuator_length0",
    "actuator_lengthrange",
    "actuator_cranklength",
)

# Axis-aligned bounding boxes: lengths that may already sit at the infinity sentinel.
BOUNDING_BOX_FIELDS = ("geom_aabb", "bvh_aabb", "oct_aabb")

# Constraint solver references whose first entry is a time constant while it is positive.
SOLVER_REFERENCE_FIELDS = (
    "geom_solref",
    "pair_solref",
    "pair_solreffriction",
    "jnt_solref",
    "dof_solref",
    "eq_solref",
    "tendon_solref_lim",
    "tendon_solref_fri",
)

# Solver impedances whose width entry is a distance measured along a length: contacts,
# explicit geom pairs and tendons.
LENGTH_IMPEDANCE_FIELDS = (
    "geom_solimp",
    "pair_solimp",
    "tendon_solimp_lim",
    "tendon_solimp_fri",
)


@dataclasses.dataclass(frozen=True)
class MorphParams:
    """Multiplicative morphology scaling factors relative to a base robot.

    The three axes are deliberately independent: size_scale alone changes geometry
    without touching mass, so size and mass can be swept separately. Use
    dynamic_similarity_params to build the physically similar combination.

    Attributes:
        size_scale: factor k applied to every length in the model.
        mass_scale: factor applied to body masses; inertia follows mass * size**2.
        torque_scale: factor applied to actuator strength, meaning both the servo gains
            and the torque limits, so that a stronger actuator is stiffer as well as
            higher limit and the axis moves the realised dynamics in both directions.
    """

    size_scale: float = 1.0
    mass_scale: float = 1.0
    torque_scale: float = 1.0

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field.name} must be finite and positive, got {value!r}")


def similar_mass_scale(size_scale: float) -> float:
    """Return the mass factor of a geometrically similar robot at constant density."""
    return float(size_scale) ** 3


def similar_torque_scale(size_scale: float) -> float:
    """Return the torque factor that keeps a constant-density robot dynamically similar."""
    return float(size_scale) ** 4


def similar_time_scale(size_scale: float) -> float:
    """Return the gait-period factor under dynamic similarity in fixed gravity."""
    return float(size_scale) ** TIME_LENGTH_POWER


def dynamic_similarity_params(size_scale: float) -> MorphParams:
    """Return the morphology that is dynamically similar to the base robot at size k."""
    return MorphParams(
        size_scale=float(size_scale),
        mass_scale=similar_mass_scale(size_scale),
        torque_scale=similar_torque_scale(size_scale),
    )


def interpolate(start: MorphParams, end: MorphParams, alpha: float) -> MorphParams:
    """Blend two morphologies, geometrically because the factors are multiplicative.

    Geometric blending keeps the dynamic-similarity family closed: every point
    between dynamic_similarity_params(a) and dynamic_similarity_params(b) is itself
    a dynamic-similarity morphology. Values of alpha outside [0, 1] extrapolate.

    Args:
        start: morphology at alpha = 0.
        end: morphology at alpha = 1.
        alpha: blending coordinate.
    """
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    blended = {}
    for field in dataclasses.fields(MorphParams):
        low = np.log(getattr(start, field.name))
        high = np.log(getattr(end, field.name))
        blended[field.name] = float(np.exp((1.0 - alpha) * low + alpha * high))
    return MorphParams(**blended)


def length_scaled_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """Return the qpos indices that carry a length: free-joint translation and slide joints."""
    indices: list[int] = []
    for joint in range(model.njnt):
        address = int(model.jnt_qposadr[joint])
        joint_type = model.jnt_type[joint]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            indices.extend([address, address + 1, address + 2])
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            indices.append(address)
    return np.array(indices, dtype=int)


def dof_inertia_length_power(model: mujoco.MjModel) -> np.ndarray:
    """Return, per degree of freedom, the power of length in its generalized inertia.

    Rotational degrees of freedom carry mass * length**2, translational ones carry mass.
    """
    powers = np.full(model.nv, 2.0)
    for joint in range(model.njnt):
        address = int(model.jnt_dofadr[joint])
        joint_type = model.jnt_type[joint]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            powers[address : address + 3] = 0.0
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            powers[address] = 0.0
    return powers


def actuator_inertia_length_power(model: mujoco.MjModel) -> np.ndarray:
    """Return, per actuator, the power of length in its reflected armature."""
    powers = np.full(model.nu, 2.0)
    joint_powers = dof_inertia_length_power(model)
    joint_transmissions = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] in joint_transmissions:
            joint = int(model.actuator_trnid[actuator, 0])
            powers[actuator] = joint_powers[int(model.jnt_dofadr[joint])]
    return powers


def actuator_coordinate_length_power(model: mujoco.MjModel) -> np.ndarray:
    """Return, per actuator, the power of length in the coordinate its transmission drives.

    A hinge angle is dimensionless, while slide travel and tendon length are lengths.
    """
    powers = np.zeros(model.nu)
    joint_transmissions = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    for actuator in range(model.nu):
        transmission = model.actuator_trntype[actuator]
        if transmission == mujoco.mjtTrn.mjTRN_TENDON:
            powers[actuator] = 1.0
        elif transmission in joint_transmissions:
            joint = int(model.actuator_trnid[actuator, 0])
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
                powers[actuator] = 1.0
    return powers


def position_servo_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of actuators whose gain multiplies a configuration.

    MuJoCo's affine bias is biasprm[0] + biasprm[1] * length + biasprm[2] * velocity, and
    MuJoCo's position, intvelocity and damped general actuators are exactly those that
    feed back length, so a non-zero biasprm[1] identifies a position gain.
    """
    return (model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE) & (
        model.actuator_biasprm[:, 1] != 0.0
    )


def velocity_servo_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of affine-bias actuators whose gain multiplies a velocity."""
    return (model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE) & (
        model.actuator_biasprm[:, 1] == 0.0
    )


def length_commanded_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of actuators whose control signal is a length rather than an angle."""
    mask = np.zeros(model.nu, dtype=bool)
    joint_transmissions = (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)
    for actuator in range(model.nu):
        if model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE:
            continue  # Not a position or velocity servo, so ctrl is not a configuration.
        transmission = model.actuator_trntype[actuator]
        if transmission == mujoco.mjtTrn.mjTRN_TENDON:
            mask[actuator] = True
        elif transmission in joint_transmissions:
            joint = int(model.actuator_trnid[actuator, 0])
            mask[actuator] = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
    return mask


def force_commanded_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Return a mask of direct-drive actuators whose control signal is a force or torque."""
    return (model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED) & (
        model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE
    )


def scale_lengths(model: mujoco.MjModel, size_scale: float) -> None:
    """Scale every length-dimensioned quantity of a model in place.

    Includes mesh vertices, without which mesh-based robots keep their original
    extent no matter what geom_size says, and the cached bounding volumes that
    MuJoCo uses for broad-phase collision, plus the dimensional constants of the
    constraint solver, without which contacts would not be similar.
    """
    for name in LENGTH_FIELDS:
        field = getattr(model, name, None)
        if field is not None and field.size:
            field *= size_scale
    for name in BOUNDING_BOX_FIELDS:
        field = getattr(model, name, None)
        if field is not None and field.size:
            field *= size_scale
            np.clip(field, -MAX_MODEL_VALUE, MAX_MODEL_VALUE, out=field)

    indices = length_scaled_qpos_indices(model)
    if indices.size:
        model.qpos0[indices] *= size_scale
        model.qpos_spring[indices] *= size_scale
        if model.nkey:
            model.key_qpos[:, indices] *= size_scale

    slide = model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE
    model.jnt_range[slide] *= size_scale

    commanded = length_commanded_actuators(model)
    if commanded.any():
        model.actuator_ctrlrange[commanded] *= size_scale
        model.actuator_actrange[commanded] *= size_scale

    scale_constraint_solver(model, size_scale)

    model.stat.extent *= size_scale
    model.stat.meansize *= size_scale
    model.stat.center *= size_scale


def scale_constraint_solver(model: mujoco.MjModel, size_scale: float) -> None:
    """Scale the dimensional constants of the soft-constraint model, in place.

    The model is inertia normalised, so the only quantities in it that carry units are the
    solref time constant, which follows the robot's gravitational clock, and the solimp
    width, which is a distance along the constraint's own coordinate: a length for a
    contact or a tendon, and the joint coordinate for a joint limit or dry friction. A
    negative solref pair states stiffness and damping directly and is left alone, since no
    robot in this study authors one. So are the equality impedances, whose residual mixes
    lengths and angles.
    """
    time_scale = size_scale**TIME_LENGTH_POWER
    for name in SOLVER_REFERENCE_FIELDS:
        field = getattr(model, name, None)
        if field is not None and field.size:
            column = field[:, 0]
            np.multiply(column, time_scale, out=column, where=column > 0.0)
    for name in LENGTH_IMPEDANCE_FIELDS:
        field = getattr(model, name, None)
        if field is not None and field.size:
            field[:, 2] *= size_scale

    coordinate = 1.0 - dof_inertia_length_power(model) / 2.0
    if model.nv:
        model.dof_solimp[:, 2] *= size_scale**coordinate
    if model.njnt:
        model.jnt_solimp[:, 2] *= size_scale ** coordinate[model.jnt_dofadr]

    override = model.opt.o_solref
    if override[0] > 0.0:
        override[0] *= time_scale
    model.opt.o_solimp[2] *= size_scale
    model.opt.o_margin *= size_scale


def scale_masses(model: mujoco.MjModel, mass_scale: float, size_scale: float) -> None:
    """Scale masses and the inertias that depend on both mass and length, in place.

    Rotational inertia carries mass * length**2, so it must follow both axes even
    when only one of them moves.
    """
    model.body_mass *= mass_scale
    model.body_inertia *= mass_scale * size_scale**2
    model.dof_armature *= mass_scale * size_scale ** dof_inertia_length_power(model)
    if model.nu:
        model.actuator_armature *= mass_scale * size_scale ** actuator_inertia_length_power(model)


def scale_passive_joints(model: mujoco.MjModel, mass_scale: float, size_scale: float) -> None:
    """Scale the passive joint properties that carry mass and length, in place.

    Joint damping, dry friction and springs belong to the mechanism rather than to the
    motors, so they follow the size and mass axes and never the torque axis. Leaving
    them fixed is what makes a nominally similar robot behave like a different machine:
    at size k its inertia grows as mass * k**2 while a fixed damping coefficient loses
    k**4.5 of relative authority.
    """
    power = dof_inertia_length_power(model)
    model.dof_damping *= mass_scale * size_scale ** (power - TIME_LENGTH_POWER)
    model.dof_frictionloss *= mass_scale * size_scale ** (power / 2.0)
    model.jnt_stiffness *= mass_scale * size_scale ** (power[model.jnt_dofadr] - 1.0)
    if model.ntendon:
        # A tendon coordinate is a length, which is the p = 0 column of the table above.
        model.tendon_stiffness *= mass_scale / size_scale
        model.tendon_damping *= mass_scale / size_scale**TIME_LENGTH_POWER
        model.tendon_frictionloss *= mass_scale


def scale_actuator_gains(model: mujoco.MjModel, torque_scale: float, size_scale: float) -> None:
    """Scale servo gains onto the scaled robot's force and clock, in place.

    A gain turns the signal it multiplies into a generalized force, so it carries the
    actuation-strength factor and, on top of that, only the unit conversion of its
    signal: a position gain is divided by the coordinate the transmission drives, and a
    velocity gain by that coordinate per unit of the robot's gravitational time scale.
    """
    coordinate = actuator_coordinate_length_power(model)
    position_gain = torque_scale * size_scale ** (-coordinate)
    velocity_gain = torque_scale * size_scale ** (TIME_LENGTH_POWER - coordinate)

    position = position_servo_actuators(model)
    velocity = velocity_servo_actuators(model)
    model.actuator_gainprm[position, 0] *= position_gain[position]
    model.actuator_gainprm[velocity, 0] *= velocity_gain[velocity]

    affine = position | velocity
    model.actuator_biasprm[affine, 0] *= torque_scale
    model.actuator_biasprm[affine, 1] *= position_gain[affine]
    model.actuator_biasprm[affine, 2] *= velocity_gain[affine]


def scale_torques(model: mujoco.MjModel, torque_scale: float, size_scale: float) -> None:
    """Scale actuator strength, meaning both the servo gains and the torque limits, in place.

    Two different fields carry the limit depending on how the robot was authored:
    actuator_forcerange on the actuator, and jnt_actfrcrange on the joint. The
    MuJoCo Playground humanoids use the joint-level one, so scaling only the
    actuator field would leave their torque budget untouched. Scaling limits alone is
    just as inert in the other direction: a position servo realises kp * (target - q),
    which for these robots sits far below the limit, so a torque_scale that moves only
    the limit changes nothing above the point where the limit stops binding.

    actuator_gear is deliberately left alone. For the joint transmissions these
    robots use it is the joint-to-actuator ratio, so scaling it would rescale the
    position targets the policy sends rather than the torque the robot can produce.
    """
    model.actuator_forcerange *= torque_scale
    model.jnt_actfrcrange *= torque_scale
    commanded = force_commanded_actuators(model)
    if commanded.any():
        model.actuator_ctrlrange[commanded] *= torque_scale
    scale_actuator_gains(model, torque_scale, size_scale)


def refresh_derived_constants(model: mujoco.MjModel) -> None:
    """Recompute the constants MuJoCo caches from mass and geometry, in place.

    mj_setConst also rebuilds the visualization statistics from scratch, discarding
    the values authored in the XML, so those three are carried across the call. The
    mean mass and mean inertia are genuinely derived and are left to be recomputed.
    """
    statistics = model.stat
    extent, meansize = statistics.extent, statistics.meansize
    center = np.array(statistics.center)

    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)

    statistics.extent, statistics.meansize = extent, meansize
    statistics.center[:] = center


def apply_morphology(model: mujoco.MjModel, params: MorphParams) -> mujoco.MjModel:
    """Return a new model scaled by the given morphology factors.

    The input model is left untouched. The three axes are kept disjoint so that no
    quantity is scaled twice: geometry and inertia follow size and mass, the passive
    joint damping, dry friction and springs follow size and mass because they are
    properties of the mechanism, and everything the motors contribute - the servo gains
    and the torque limits - follows the torque axis, with the size axis adding only the
    sqrt(size) clock factor that converts a force gain into a velocity gain.
    dynamic_similarity_params therefore lands on the similar robot exactly once: its
    torque_scale of k**4 is the actuator factor similarity asks for, while mass and size
    deliver the k**4.5 damping and k**4 dry friction.

    Args:
        model: base robot model.
        params: multiplicative size, mass and torque factors.

    Returns:
        A scaled deep copy of the model with its cached constants refreshed.
    """
    scaled = copy.deepcopy(model)
    if params.size_scale != 1.0:
        scale_lengths(scaled, params.size_scale)
    if params.mass_scale != 1.0 or params.size_scale != 1.0:
        scale_masses(scaled, params.mass_scale, params.size_scale)
        scale_passive_joints(scaled, params.mass_scale, params.size_scale)
    if params.torque_scale != 1.0 or params.size_scale != 1.0:
        scale_torques(scaled, params.torque_scale, params.size_scale)
    refresh_derived_constants(scaled)
    return scaled


def geom_bounding_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Return points in the geom frame that bound it: mesh vertices, or box corners."""
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[geom] >= 0:
        mesh = int(model.geom_dataid[geom])
        address = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        return model.mesh_vert[address : address + count]
    center = model.geom_aabb[geom, :3]
    half = model.geom_aabb[geom, 3:]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    return center + signs * half


def physical_extent(model: mujoco.MjModel) -> float:
    """Return the world-space bounding-box diagonal of the robot at its default pose.

    Mesh geoms are measured from their actual vertices rather than the compiler's
    geom_size cache, so this reports the true extent of mesh-based robots. Geometry
    attached to the world body, such as the floor or terrain, is excluded.
    """
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        world = data.geom_xpos[geom] + geom_bounding_points(model, geom) @ rotation.T
        low = np.minimum(low, world.min(axis=0))
        high = np.maximum(high, world.max(axis=0))
    if not np.isfinite(low).all():
        return 0.0
    return float(np.linalg.norm(high - low))


def total_mass(model: mujoco.MjModel) -> float:
    """Return the total mass of every body in a model."""
    return float(model.body_mass.sum())
