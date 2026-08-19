"""Parametric morphology scaling of MuJoCo models along independent size, mass and torque axes."""

import copy
import dataclasses
from typing import List

import mujoco
import numpy as np

# MuJoCo treats magnitudes at or above mjMAXVAL as infinite; scaling must not overflow it.
MAX_MODEL_VALUE = mujoco.mjMAXVAL

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


@dataclasses.dataclass(frozen=True)
class MorphParams:
    """Multiplicative morphology scaling factors relative to a base robot.

    The three axes are deliberately independent: size_scale alone changes geometry
    without touching mass, so size and mass can be swept separately. Use
    dynamic_similarity_params to build the physically similar combination.

    Attributes:
        size_scale: factor k applied to every length in the model.
        mass_scale: factor applied to body masses; inertia follows mass * size**2.
        torque_scale: factor applied to actuator torque limits.
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
    return float(size_scale) ** 0.5


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
    indices: List[int] = []
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
    MuJoCo uses for broad-phase collision.
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

    model.stat.extent *= size_scale
    model.stat.meansize *= size_scale
    model.stat.center *= size_scale


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


def scale_torques(model: mujoco.MjModel, torque_scale: float) -> None:
    """Scale the actuator torque limits of a model in place.

    Two different fields carry the limit depending on how the robot was authored:
    actuator_forcerange on the actuator, and jnt_actfrcrange on the joint. The
    MuJoCo Playground humanoids use the joint-level one, so scaling only the
    actuator field would leave their torque budget untouched. An actuator that
    declares no limit in either field has no torque budget to scale and is
    unaffected; only its servo gain would bound it, and gain is a stiffness axis
    rather than a morphology axis.

    actuator_gear is deliberately left alone. For the joint transmissions these
    robots use it is the joint-to-actuator ratio, so scaling it would rescale the
    position targets the policy sends rather than the torque the robot can produce.
    """
    model.actuator_forcerange *= torque_scale
    model.jnt_actfrcrange *= torque_scale
    commanded = force_commanded_actuators(model)
    if commanded.any():
        model.actuator_ctrlrange[commanded] *= torque_scale


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

    The input model is left untouched. Joint damping and dry friction are not
    scaled: they are separate actuation axes of this study, not consequences of
    geometry.

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
    if params.torque_scale != 1.0:
        scale_torques(scaled, params.torque_scale)
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
