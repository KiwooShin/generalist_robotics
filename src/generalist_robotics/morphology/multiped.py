"""A procedural multiped whose legs can be grown in continuously, one at a time."""

import copy
import dataclasses
import math
from collections.abc import Sequence

import mujoco
import numpy as np

from generalist_robotics.morphology.scaling import refresh_derived_constants
from generalist_robotics.morphology.topology import DofLock, lock_joints

# Degrees of freedom every leg carries, in the order they appear in the model: an
# abduction hinge about the world x axis and two fore-aft hinges about the world y axis.
# The axes are deliberately not rotated with the leg's place on the hip ring, because
# every leg of a walker swings fore-aft whatever side of the body it hangs from; only the
# hip's position is radial.
LEG_JOINT_SUFFIXES = ("hip_roll", "hip_pitch", "knee")
LEG_JOINT_AXES = ("1 0 0", "0 1 0", "0 1 0")
LEG_JOINT_RANGES = ((-0.6, 0.6), (-1.2, 1.2), (0.0, 2.0))

# Standing pose, in radians: the hip pitches back by HOME_HIP_PITCH and the knee closes by
# twice that, which puts the foot exactly under its hip and leaves the leg the same
# crouch whatever its length.
HOME_HIP_PITCH = -0.3
HOME_KNEE = 0.6

# Fractions of leg_mass carried by the thigh, the shin and the foot.
LEG_MASS_SHARES = (0.5, 0.35, 0.15)

# Link geometry as fractions of leg_length: the two segments are half a leg each, the
# limb is slender, and the foot is a sphere a little fatter than the shin.
SEGMENT_FRACTION = 0.5
LIMB_RADIUS_FRACTION = 0.05
FOOT_RADIUS_FRACTION = 0.07

# Torso box half-extents, as fractions of hip_spacing for the two horizontal axes so the
# body just spans its own hip ring, and in metres for the vertical one.
TORSO_HALF_LENGTH_FRACTION = 1.0
TORSO_HALF_WIDTH_FRACTION = 0.6
TORSO_HALF_HEIGHT = 0.05

# Integration and control periods. Five substeps per control step is the ratio Playground
# uses for its quadrupeds, and eulerdamp is disabled for the same reason its XMLs disable
# it: morphology.topology sizes a lock spring's damping against an explicit Euler step.
SIM_TIMESTEP = 0.004
CTRL_TIMESTEP = 0.02

# Position-servo gains and torque ceiling of a fully grown leg, in SI units. kp is set so
# that a knee holding the standing crouch sags by well under a tenth of a radian, and kv
# leaves the servo slightly under critically damped against a segment's own inertia.
JOINT_GAIN = 60.0
JOINT_DAMPING_GAIN = 3.0
JOINT_FORCE_LIMIT = 30.0
JOINT_DAMPING = 0.3
JOINT_ARMATURE = 0.01

# Where a leg starts from when its growth is 0: a stub GROWTH_LENGTH_FLOOR of the full
# leg long carrying GROWTH_MASS_FLOOR of its mass. Neither is zero, because MuJoCo needs a
# positive inertia on any body that carries a joint and a mass matrix whose smallest
# diagonal is a billionth of its largest is a badly conditioned one. Both are small enough
# that the stub is mechanically absent: at leg_length 0.5 m and leg_mass 1.5 kg it is a two
# centimetre, six gram bump on the hip, far too short to reach the floor from a hip half a
# metre up and four parts in ten thousand of the robot's mass.
GROWTH_LENGTH_FLOOR = 0.04
GROWTH_MASS_FLOOR = 4.0e-3


@dataclasses.dataclass(frozen=True)
class MultipedSpec:
    """Parametric multiped: torso plus n_legs identical 3-DoF legs on a radial layout.

    The hips sit on a circle of radius hip_spacing around the torso centre, at angles
    spaced 2*pi / n_legs apart with the first slot on the left of the body. Legs are
    indexed in the growth order of leg_angles: the lateral pair first, so that the
    two-leg prefix of any spec is a left-right biped, then the rear leg, then the front
    one.

    Attributes:
        n_legs: number of legs the model carries. Two is a left-right biped, three a
            tripod with a rear leg, four a quadruped with a leg at each compass point.
        leg_length: hip-to-foot length of a fully extended leg, in metres, split evenly
            between thigh and shin.
        torso_mass: mass of the torso box, in kilograms.
        leg_mass: mass of one fully grown leg, in kilograms, split by LEG_MASS_SHARES.
        hip_spacing: distance from the torso centre to each hip, in metres, i.e. the
            radius of the hip ring rather than a gap between neighbours.
    """

    n_legs: int = 2
    leg_length: float = 0.5
    torso_mass: float = 8.0
    leg_mass: float = 1.5
    hip_spacing: float = 0.15

    def __post_init__(self):
        object.__setattr__(self, "n_legs", int(self.n_legs))
        for field in ("leg_length", "torso_mass", "leg_mass", "hip_spacing"):
            object.__setattr__(self, field, float(getattr(self, field)))
        if self.n_legs < 1:
            raise ValueError(f"n_legs must be at least 1, got {self.n_legs!r}")
        for field in ("leg_length", "torso_mass", "leg_mass", "hip_spacing"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive, got {value!r}")


@dataclasses.dataclass(frozen=True)
class LegGrowth:
    """How grown a leg is: 0 = massless and rigidly locked, 1 = fully present.

    Attributes:
        leg_index: which leg of the spec this refers to.
        growth: 0 shrinks the leg to a stub of GROWTH_LENGTH_FLOOR of its length carrying
            GROWTH_MASS_FLOOR of its mass and holds its three joints rigid; 1 leaves the
            leg exactly as the spec authored it. Values in between interpolate linearly.
    """

    leg_index: int
    growth: float

    def __post_init__(self):
        object.__setattr__(self, "leg_index", int(self.leg_index))
        object.__setattr__(self, "growth", float(self.growth))
        if self.leg_index < 0:
            raise ValueError(f"leg_index must not be negative, got {self.leg_index!r}")
        if not 0.0 <= self.growth <= 1.0:
            raise ValueError(f"growth must be in [0, 1], got {self.growth!r}")


def leg_angles(spec: MultipedSpec) -> tuple[float, ...]:
    """Yaw angle of each leg's hip, in radians from the forward axis, in growth order.

    The slots themselves are the evenly spaced ring the spec describes, with the first on
    the left of the body. The order they are handed out in is what makes a prefix of the
    legs a usable robot: slots are sorted by how lateral they are, so the lateral pair
    comes first, and the fore-aft slots follow with the rear one before the front one. A
    four-leg spec is therefore indexed left, right, rear, front, so growing legs 2 and 3
    in walks a left-right biped through a rear-legged tripod to a quadruped. The sort keys
    are rounded because the ring puts genuinely equal angles a few ulps apart, and an
    ordering that depends on those bits would not be reproducible.
    """
    slots = [math.pi / 2.0 + 2.0 * math.pi * index / spec.n_legs for index in range(spec.n_legs)]

    def laterality(angle: float) -> tuple[float, float]:
        return (-round(abs(math.sin(angle)), 9), round(math.cos(angle), 9))

    return tuple(sorted(slots, key=laterality))


def hip_position(spec: MultipedSpec, leg_index: int) -> tuple[float, float, float]:
    """Position of one hip in the torso frame, in metres."""
    angle = leg_angles(spec)[check_leg_index(spec, leg_index)]
    return (spec.hip_spacing * math.cos(angle), spec.hip_spacing * math.sin(angle), 0.0)


def check_leg_index(spec: MultipedSpec, leg_index: int) -> int:
    """Return the leg index, raising if the spec has no such leg."""
    index = int(leg_index)
    if not 0 <= index < spec.n_legs:
        raise ValueError(f"spec has {spec.n_legs} legs, so leg_index {leg_index!r} is out of range")
    return index


def leg_joint_names(spec: MultipedSpec, leg_index: int) -> tuple[str, ...]:
    """Names of one leg's three actuated joints, in model order."""
    index = check_leg_index(spec, leg_index)
    return tuple(f"leg{index}_{suffix}" for suffix in LEG_JOINT_SUFFIXES)


def joint_names(spec: MultipedSpec) -> tuple[str, ...]:
    """Names of every actuated joint, in model order."""
    return tuple(name for leg in range(spec.n_legs) for name in leg_joint_names(spec, leg))


def leg_body_names(spec: MultipedSpec, leg_index: int) -> tuple[str, ...]:
    """Names of the two bodies one leg is built from, proximal first."""
    index = check_leg_index(spec, leg_index)
    return (f"leg{index}_thigh", f"leg{index}_shin")


def leg_geom_names(spec: MultipedSpec, leg_index: int) -> tuple[str, ...]:
    """Names of the three geoms one leg is built from, proximal first."""
    index = check_leg_index(spec, leg_index)
    return (f"leg{index}_thigh", f"leg{index}_shin", f"leg{index}_foot")


def foot_site_name(spec: MultipedSpec, leg_index: int) -> str:
    """Name of the site at the centre of one foot, which is where stance is measured."""
    return f"leg{check_leg_index(spec, leg_index)}_foot"


def foot_site_names(spec: MultipedSpec) -> tuple[str, ...]:
    """Names of every foot site, in leg order."""
    return tuple(foot_site_name(spec, leg) for leg in range(spec.n_legs))


def segment_length(spec: MultipedSpec) -> float:
    """Length of one leg segment, in metres."""
    return spec.leg_length * SEGMENT_FRACTION


def standing_height(spec: MultipedSpec) -> float:
    """Height of the torso centre above the floor in the standing pose, in metres.

    The home crouch pitches the hip back by HOME_HIP_PITCH and closes the knee by twice
    that, so both segments lean by the same angle and the foot hangs a segment's cosine
    twice below the hip, plus the foot's own radius.
    """
    drop = 2.0 * segment_length(spec) * math.cos(HOME_HIP_PITCH)
    return drop + FOOT_RADIUS_FRACTION * spec.leg_length


def home_qpos(spec: MultipedSpec) -> np.ndarray:
    """Standing pose of the whole model: torso upright at standing_height, legs crouched."""
    pose = [0.0, 0.0, standing_height(spec), 1.0, 0.0, 0.0, 0.0]
    for _ in range(spec.n_legs):
        pose.extend([0.0, HOME_HIP_PITCH, HOME_KNEE])
    return np.asarray(pose, dtype=float)


def home_ctrl(spec: MultipedSpec) -> np.ndarray:
    """Position-servo targets that hold the standing pose."""
    return np.asarray([0.0, HOME_HIP_PITCH, HOME_KNEE] * spec.n_legs, dtype=float)


def leg_xml(spec: MultipedSpec, leg_index: int) -> str:
    """Return the MJCF of one leg, as a body to be nested under the torso."""
    names = leg_joint_names(spec, leg_index)
    bodies = leg_body_names(spec, leg_index)
    geoms = leg_geom_names(spec, leg_index)
    hip = hip_position(spec, leg_index)
    half = segment_length(spec)
    radius = LIMB_RADIUS_FRACTION * spec.leg_length
    foot_radius = FOOT_RADIUS_FRACTION * spec.leg_length
    thigh_mass, shin_mass, foot_mass = (share * spec.leg_mass for share in LEG_MASS_SHARES)
    joints = [
        f'<joint name="{name}" axis="{axis}" range="{low} {high}"/>'
        for name, axis, (low, high) in zip(names, LEG_JOINT_AXES, LEG_JOINT_RANGES, strict=True)
    ]
    return f"""
      <body name="{bodies[0]}" pos="{hip[0]:.6f} {hip[1]:.6f} {hip[2]:.6f}">
        {joints[0]}
        {joints[1]}
        <geom name="{geoms[0]}" type="capsule" class="limb"
              fromto="0 0 0 0 0 {-half:.6f}" size="{radius:.6f}" mass="{thigh_mass:.6f}"/>
        <body name="{bodies[1]}" pos="0 0 {-half:.6f}">
          {joints[2]}
          <geom name="{geoms[1]}" type="capsule" class="limb"
                fromto="0 0 0 0 0 {-half:.6f}" size="{radius:.6f}" mass="{shin_mass:.6f}"/>
          <geom name="{geoms[2]}" type="sphere" class="foot"
                pos="0 0 {-half:.6f}" size="{foot_radius:.6f}" mass="{foot_mass:.6f}"/>
          <site name="{foot_site_name(spec, leg_index)}" pos="0 0 {-half:.6f}" size="0.01"/>
        </body>
      </body>"""


def actuator_xml(spec: MultipedSpec) -> str:
    """Return the MJCF of the position servos, one per actuated joint."""
    lines = []
    for leg in range(spec.n_legs):
        for name, (low, high) in zip(leg_joint_names(spec, leg), LEG_JOINT_RANGES, strict=True):
            lines.append(f'    <position name="{name}" joint="{name}" ctrlrange="{low} {high}"/>')
    return "\n".join(lines)


def build_multiped_xml(spec: MultipedSpec) -> str:
    """Return the MJCF of a multiped built to the spec, as a string.

    Only the feet and the torso collide with the floor, and nothing on the robot collides
    with anything else on it: the thighs and shins are visual, which keeps the contact
    set at n_legs + 1 pairs and stops a leg that has not grown yet from tripping the ones
    that have.
    """
    torso_half = (
        TORSO_HALF_LENGTH_FRACTION * spec.hip_spacing,
        TORSO_HALF_WIDTH_FRACTION * spec.hip_spacing,
        TORSO_HALF_HEIGHT,
    )
    legs = "".join(leg_xml(spec, leg) for leg in range(spec.n_legs))
    pose = " ".join(f"{value:.6f}" for value in home_qpos(spec))
    control = " ".join(f"{value:.6f}" for value in home_ctrl(spec))
    return f"""<mujoco model="multiped{spec.n_legs}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_TIMESTEP}" integrator="Euler" iterations="4" ls_iterations="8">
    <flag eulerdamp="disable"/>
  </option>
  <default>
    <joint type="hinge" limited="true" damping="{JOINT_DAMPING}" armature="{JOINT_ARMATURE}"/>
    <position kp="{JOINT_GAIN}" kv="{JOINT_DAMPING_GAIN}"
              forcerange="{-JOINT_FORCE_LIMIT} {JOINT_FORCE_LIMIT}"/>
    <default class="limb">
      <geom contype="0" conaffinity="0" rgba="0.55 0.6 0.68 1"/>
    </default>
    <default class="foot">
      <geom contype="0" conaffinity="1" friction="0.9 0.02 0.01" rgba="0.9 0.5 0.2 1"/>
    </default>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1"
          friction="0.9 0.02 0.01" rgba="0.3 0.32 0.36 1"/>
    <body name="torso" pos="0 0 {standing_height(spec):.6f}">
      <freejoint name="root"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <geom name="torso" type="box" contype="0" conaffinity="1" rgba="0.2 0.45 0.75 1"
            size="{torso_half[0]:.6f} {torso_half[1]:.6f} {torso_half[2]:.6f}"
            mass="{spec.torso_mass:.6f}"/>{legs}
    </body>
  </worldbody>
  <actuator>
{actuator_xml(spec)}
  </actuator>
  <keyframe>
    <key name="home" qpos="{pose}" ctrl="{control}"/>
  </keyframe>
</mujoco>
"""


def build_multiped_model(spec: MultipedSpec) -> mujoco.MjModel:
    """Compile a multiped spec into a MuJoCo model."""
    return mujoco.MjModel.from_xml_string(build_multiped_xml(spec))


def growth_by_leg(spec: MultipedSpec, growth: Sequence[LegGrowth]) -> tuple[float, ...]:
    """Flatten a growth specification into one factor per leg, defaulting to fully grown.

    Raises:
        ValueError: if a leg is named twice or does not exist in the spec.
    """
    factors = [1.0] * spec.n_legs
    seen: set[int] = set()
    for entry in growth:
        index = check_leg_index(spec, entry.leg_index)
        if index in seen:
            raise ValueError(f"leg {index} is given a growth twice")
        seen.add(index)
        factors[index] = entry.growth
    return tuple(factors)


def growth_length_scale(growth: float) -> float:
    """Length factor of a leg at a given growth, from GROWTH_LENGTH_FLOOR to 1."""
    return GROWTH_LENGTH_FLOOR + (1.0 - GROWTH_LENGTH_FLOOR) * float(growth)


def growth_mass_scale(growth: float) -> float:
    """Mass factor of a leg at a given growth, from GROWTH_MASS_FLOOR to 1.

    Linear rather than geometric, because mass enters the dynamics linearly and the point
    of the anneal is that the load a leg adds to the mechanism arrives smoothly. Lengths
    follow their own linear map, and inertia is then mass * length**2 as it must be, so a
    half-grown leg is a consistent smaller, lighter limb rather than a scaled number.
    """
    return GROWTH_MASS_FLOOR + (1.0 - GROWTH_MASS_FLOOR) * float(growth)


def leg_growth_locks(spec: MultipedSpec, growth: Sequence[LegGrowth]) -> tuple[DofLock, ...]:
    """Return the joint locks that make a partly grown leg's joints correspondingly rigid.

    A leg that is not there cannot be steered, so its lock is exactly one minus its
    growth: at growth 0 all three joints are held rigid and their actuators are dead, and
    the lock anneals away as the leg arrives.

    Every leg is named, fully grown ones at lock 0 included, because
    morphology.topology.lock_springs normalises a lock against the whole set of joints it
    is handed. Dropping a leg from the set the moment it finishes growing would rescale
    the springs still holding the others, which is exactly the discontinuity the anneal
    exists to avoid.
    """
    return tuple(
        DofLock(joint_names=leg_joint_names(spec, index), lock=1.0 - factor)
        for index, factor in enumerate(growth_by_leg(spec, growth))
    )


def scale_leg_geometry(
    model: mujoco.MjModel, spec: MultipedSpec, leg_index: int, length: float, mass: float
) -> None:
    """Shrink one leg's links, geoms and actuators in place by a length and a mass factor.

    Everything a limb is made of follows one of three exponents: lengths follow the length
    factor, masses the mass factor, and every rotational inertia the product mass *
    length**2, which is what keeps a shrunken leg a physically consistent small leg rather
    than a light long one. The joint damping, the armature and the servo gains are
    rotational inertias in that same sense, so they follow the same product and a
    half-grown leg's servo keeps the natural frequency of a fully grown one.

    The hip's own place on the torso is deliberately left alone: a leg grows outward from
    where it is attached, it does not migrate.
    """
    inertia = mass * length**2
    for depth, name in enumerate(leg_body_names(spec, leg_index)):
        body = model.body(name)
        # The proximal link hangs at the hip, whose ring radius is not part of the leg.
        if depth > 0:
            body.pos[:] = body.pos * length
        body.ipos[:] = body.ipos * length
        body.mass[:] = body.mass * mass
        body.inertia[:] = body.inertia * inertia
    for name in leg_geom_names(spec, leg_index):
        geom = model.geom(name)
        geom.pos[:] = geom.pos * length
        geom.size[:] = geom.size * length
    for name in leg_joint_names(spec, leg_index):
        joint = model.joint(name)
        model.dof_damping[int(joint.dofadr[0])] *= inertia
        model.dof_armature[int(joint.dofadr[0])] *= inertia
        actuator = model.actuator(name)
        actuator.gainprm[0] *= inertia
        actuator.biasprm[1] *= inertia
        actuator.biasprm[2] *= inertia
        actuator.forcerange[:] = actuator.forcerange * inertia
    site = model.site(foot_site_name(spec, leg_index))
    site.pos[:] = site.pos * length


def apply_leg_growth(
    model: mujoco.MjModel, spec: MultipedSpec, growth: tuple[LegGrowth, ...]
) -> mujoco.MjModel:
    """Return a copy of the model whose named legs are grown to the given extent.

    Growth is two consistent things at once, which is what makes a leg that is not there
    yet genuinely absent rather than merely unused:

      - the limb is shrunk toward a stub at its hip, so an ungrown leg carries no mass,
        adds no inertia and cannot reach the floor to take load;
      - its three joints are held rigid by morphology.topology and their actuators are
        scaled to nothing, so the policy's action channels for that leg drive nothing.

    The order matters. Masses are scaled first and the lock is sized afterwards, because
    the lock spring is normalised against the mass matrix of the body it is applied to: a
    one-and-a-half-gram stub needs a spring a thousand times weaker than a full leg, and
    sizing the lock on the unscaled model would put the integrator well past the stability
    bound morphology.topology bracketed.

    Args:
        model: a model compiled from the same spec.
        spec: the spec the model was built from.
        growth: how grown each named leg is; legs not named are left fully grown.

    Returns:
        A grown deep copy of the model with its cached constants refreshed.

    Raises:
        ValueError: if a leg is named twice or does not exist in the spec.
    """
    factors = growth_by_leg(spec, growth)
    grown = copy.deepcopy(model)
    for index, factor in enumerate(factors):
        if factor >= 1.0:
            continue
        scale_leg_geometry(
            grown, spec, index, growth_length_scale(factor), growth_mass_scale(factor)
        )
    refresh_derived_constants(grown)
    return lock_joints(grown, leg_growth_locks(spec, growth))


def settle(model: mujoco.MjModel, seconds: float = 2.0) -> mujoco.MjData:
    """Drop the model into its standing pose and hold the servos there for a while.

    Returns:
        The data at the end of the settle, which a caller can read a resting height and a
        residual velocity off to check that the body is physically sane.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    return data


def total_mass(model: mujoco.MjModel) -> float:
    """Total mass of every body in a model, in kilograms."""
    return float(np.sum(model.body_mass))
