"""Factory for MuJoCo Playground locomotion environments whose robot model is morphed."""

import contextlib
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

from generalist_robotics.morphology.scaling import (
    TIME_LENGTH_POWER,
    MorphParams,
    apply_morphology,
    similar_time_scale,
)

# Short robot name -> MuJoCo Playground environment id. Every entry must build its
# MjModel through mujoco.MjModel.from_xml_string, which is where the morph is injected;
# BarkourJoystick assembles its model from an MjSpec instead and is therefore excluded.
ROBOT_ENVIRONMENTS: dict[str, str] = {
    "berkeley_humanoid": "BerkeleyHumanoidJoystickFlatTerrain",
    "berkeley_humanoid_rough": "BerkeleyHumanoidJoystickRoughTerrain",
    "g1": "G1JoystickFlatTerrain",
    "g1_rough": "G1JoystickRoughTerrain",
    "t1": "T1JoystickFlatTerrain",
    "t1_rough": "T1JoystickRoughTerrain",
    "h1": "H1JoystickGaitTracking",
    "op3": "Op3Joystick",
    "apollo": "ApolloJoystickFlatTerrain",
    "go1": "Go1JoystickFlatTerrain",
    "go1_rough": "Go1JoystickRoughTerrain",
    "spot": "SpotFlatTerrainJoystick",
}

# Fields the morph writes that the MJX model must have inherited. One per morph axis, so
# the check stays sharp for a morph that moves only size, only mass or only torque.
MORPH_WITNESS_FIELDS = ("body_mass", "geom_size", "dof_damping", "actuator_gainprm")

# Exponent of size_scale carried by each physical dimension under dynamic similarity in
# fixed gravity: a robot k times longer is sqrt(k) times slower, so lengths follow k,
# speeds sqrt(k), rates and frequencies 1/sqrt(k), durations sqrt(k) and forces k**3.
LENGTH_EXPONENT = 1.0
LINEAR_VELOCITY_EXPONENT = TIME_LENGTH_POWER
ANGULAR_VELOCITY_EXPONENT = -TIME_LENGTH_POWER
DURATION_EXPONENT = TIME_LENGTH_POWER
FREQUENCY_EXPONENT = -TIME_LENGTH_POWER
FORCE_EXPONENT = 3.0

# A joystick command is (forward speed, lateral speed, yaw rate), so the three entries of
# a command vector do not share one exponent.
COMMAND_EXPONENTS = (
    LINEAR_VELOCITY_EXPONENT,
    LINEAR_VELOCITY_EXPONENT,
    ANGULAR_VELOCITY_EXPONENT,
)

# Every Playground locomotion config entry that carries a physical unit, keyed by the
# flattened override name and valued by the power of size_scale it follows. Robots expose
# different subsets, so an entry that a robot does not define is skipped. Entries that are
# dimensionless in these environments are deliberately absent and listed in the
# similarity_task_overrides docstring together with the ones left alone on purpose.
TASK_SIZE_EXPONENTS: dict[str, float | tuple[float, ...]] = {
    # Swing-foot and base height targets, in metres.
    "reward_config.max_foot_height": LENGTH_EXPONENT,
    "max_foot_height": LENGTH_EXPONENT,
    "reward_config.base_height_target": LENGTH_EXPONENT,
    "foot_height": LENGTH_EXPONENT,
    "obs_noise.scales.feet_pos": LENGTH_EXPONENT,
    # Joystick command ranges and the dead zones applied to them.
    "lin_vel_x": LINEAR_VELOCITY_EXPONENT,
    "lin_vel_y": LINEAR_VELOCITY_EXPONENT,
    "ang_vel_yaw": ANGULAR_VELOCITY_EXPONENT,
    "command_config.lin_vel_x": LINEAR_VELOCITY_EXPONENT,
    "command_config.lin_vel_y": LINEAR_VELOCITY_EXPONENT,
    "command_config.ang_vel_yaw": ANGULAR_VELOCITY_EXPONENT,
    "command_config.lin_vel_threshold": LINEAR_VELOCITY_EXPONENT,
    "command_config.ang_vel_threshold": ANGULAR_VELOCITY_EXPONENT,
    "command_config.a": COMMAND_EXPONENTS,
    "command_config.min": COMMAND_EXPONENTS,
    "command_config.max": COMMAND_EXPONENTS,
    # Commanded gait frequency, in hertz.
    "gait_frequency": FREQUENCY_EXPONENT,
    # Disturbances: the kick is added straight to the root velocity, and its schedule is
    # measured on the robot's own gravitational clock.
    "push_config.magnitude_range": LINEAR_VELOCITY_EXPONENT,
    "push_config.interval_range": DURATION_EXPONENT,
    "pert_config.velocity_kick": LINEAR_VELOCITY_EXPONENT,
    "pert_config.kick_durations": DURATION_EXPONENT,
    "pert_config.kick_wait_times": DURATION_EXPONENT,
    "velocity_kick": LINEAR_VELOCITY_EXPONENT,
    "kick_durations": DURATION_EXPONENT,
    "kick_wait_times": DURATION_EXPONENT,
    # Observation noise on the sensors that read a speed or a rate.
    "noise_config.scales.linvel": LINEAR_VELOCITY_EXPONENT,
    "noise_config.scales.gyro": ANGULAR_VELOCITY_EXPONENT,
    "noise_config.scales.joint_vel": ANGULAR_VELOCITY_EXPONENT,
    "obs_noise.scales.gyro": ANGULAR_VELOCITY_EXPONENT,
    "obs_noise.scales.joint_vel": ANGULAR_VELOCITY_EXPONENT,
    # Contact-force budget, in newtons.
    "reward_config.max_contact_force": FORCE_EXPONENT,
}

# Go1, Spot and Op3 keep their servo gains in the task config and write them onto the
# compiled model, so the morph has to be re-applied through the config for those robots.
POSITION_GAIN_CONFIG_KEY = "Kp"
JOINT_DAMPING_CONFIG_KEY = "Kd"

# dof_damping of a rotational joint carries mass * length**(2 - 0.5); see
# scaling.scale_passive_joints for where the generalized-inertia exponent comes from.
ROTATIONAL_DAMPING_SIZE_EXPONENT = 2.0 - TIME_LENGTH_POWER


def available_robots() -> dict[str, str]:
    """Map short robot name -> MuJoCo Playground environment id."""
    return dict(ROBOT_ENVIRONMENTS)


def environment_id(robot: str) -> str:
    """Return the Playground environment id registered for a short robot name."""
    if robot not in ROBOT_ENVIRONMENTS:
        raise ValueError(f"Unknown robot {robot!r}. Available robots: {sorted(ROBOT_ENVIRONMENTS)}")
    return ROBOT_ENVIRONMENTS[robot]


def flatten_config(config: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a Playground config into the dotted keys its override mechanism accepts."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if hasattr(value, "items"):
            flat.update(flatten_config(value, f"{name}."))
        else:
            flat[name] = value
    return flat


@contextlib.contextmanager
def morphed_compilation(params: MorphParams) -> Iterator[list[mujoco.MjModel]]:
    """Patch MuJoCo XML compilation so every model compiled inside the block is morphed.

    A Playground environment compiles its MjModel, immediately hands it to
    mjx.put_model, and then caches keyframe poses, joint limits and torque budgets read
    off that model. Morphing the model after construction would leave both the MJX model
    that env.step integrates and all of those caches on the original robot, so the morph
    is injected at compile time instead and the environment builds itself around the
    scaled robot from the start.

    The patch is global for the duration of the block and is not thread safe; construct
    morphed environments from a single thread.

    Args:
        params: multiplicative size, mass and torque factors.

    Yields:
        The list of morphed models handed out, so the caller can verify that the
        environment really kept one of them.
    """
    produced: list[mujoco.MjModel] = []
    compile_original = mujoco.MjModel.from_xml_string

    def compile_morphed(*args: Any, **kwargs: Any) -> mujoco.MjModel:
        model = apply_morphology(compile_original(*args, **kwargs), params)
        produced.append(model)
        return model

    mujoco.MjModel.from_xml_string = staticmethod(compile_morphed)
    try:
        yield produced
    finally:
        mujoco.MjModel.from_xml_string = staticmethod(compile_original)


def check_morph_reached_env(env: mjx_env.MjxEnv, morphed_models: list[mujoco.MjModel]) -> None:
    """Raise unless the environment is simulating one of the morphed models.

    env.step integrates env.mjx_model, so a morph that only reached env.mj_model would
    be silently inert. Both links are checked rather than assumed: that the environment
    kept a morphed MjModel, and that its MJX model still agrees with that model on one
    field per morph axis, so the check cannot pass on a stale MJX model just because the
    morph happened not to move masses.

    Args:
        env: freshly constructed Playground environment.
        morphed_models: models produced by morphed_compilation during construction.

    Raises:
        RuntimeError: if the morph did not reach the simulated model.
    """
    if not any(env.mj_model is model for model in morphed_models):
        raise RuntimeError(
            f"{type(env).__name__} did not build its MjModel through "
            "mujoco.MjModel.from_xml_string, so the morphology was not applied."
        )
    for name in MORPH_WITNESS_FIELDS:
        simulated = getattr(env.mjx_model, name, None)
        compiled = getattr(env.mj_model, name, None)
        if simulated is None or compiled is None:
            continue
        if not np.allclose(np.asarray(simulated), np.asarray(compiled)):
            raise RuntimeError(
                f"{type(env).__name__}.mjx_model.{name} disagrees with the morphed "
                "mj_model, so the MJX model was not built from it."
            )


def similarity_time_overrides(env_id: str, size_scale: float) -> dict[str, float]:
    """Return ctrl_dt/sim_dt overrides putting a size-scaled robot on a similar clock.

    Under dynamic similarity in fixed gravity a robot k times larger moves sqrt(k)
    times slower, so both the control period and the integrator step stretch by the
    same factor and the number of substeps per control step is unchanged.

    Args:
        env_id: Playground environment id, used to read the base timing.
        size_scale: factor k applied to every length in the model.
    """
    factor = similar_time_scale(size_scale)
    config = registry.get_default_config(env_id)
    return {
        "ctrl_dt": float(config.ctrl_dt) * factor,
        "sim_dt": float(config.sim_dt) * factor,
    }


def scaled_task_value(value: Any, exponent: float | tuple[float, ...], size_scale: float) -> Any:
    """Scale one config entry by size_scale raised to the exponent of its physical unit.

    A tuple of exponents scales a vector entry component by component, which is what a
    joystick command needs because its speeds and its yaw rate scale in opposite
    directions.

    Raises:
        ValueError: if a per-component exponent does not match the length of the entry.
    """
    if isinstance(exponent, tuple):
        if not isinstance(value, list | tuple) or len(value) != len(exponent):
            raise ValueError(
                f"Expected a sequence of {len(exponent)} entries to scale, got {value!r}."
            )
        return [
            float(entry) * size_scale**power for entry, power in zip(value, exponent, strict=True)
        ]
    if isinstance(value, list | tuple):
        return [float(entry) * size_scale**exponent for entry in value]
    return float(value) * size_scale**exponent


def similarity_task_overrides(env_id: str, size_scale: float) -> dict[str, Any]:
    """Return config overrides restating the task in the k-scaled robot's own units.

    Playground keeps the task specification in lengths, speeds and durations that belong
    to the base robot, so morphing the body alone rewards a k times larger robot for
    lifting its feet to the small robot's clearance and commands it at the small robot's
    speed. Every entry in TASK_SIZE_EXPONENTS is restated on the scaled robot: foot-height
    and base-height targets and feet-position noise follow k, joystick speed ranges,
    velocity kicks, linear-velocity noise and disturbance intervals follow sqrt(k), yaw
    commands, gyro and joint-velocity noise and gait frequency follow 1/sqrt(k), and the
    contact-force budget follows k**3. Keys a robot does not define are skipped, so the
    same call works for a quadruped and a humanoid.

    Left alone on purpose, because they are dimensionless in these environments: reward
    weights (reward_config.scales.*), action_scale and the joint-angle noise scales, which
    are angles, soft_joint_pos_limit_factor, noise_config.level, command zero
    probabilities (command_config.b, command_config.zero_prob), episode_length, which is a
    step count and already covers sqrt(k) more seconds once scale_time stretches the
    clock, and the solver budgets naconmax and njmax.

    Left alone despite carrying a unit, because no single exponent is right:
    reward_config.tracking_sigma is one scalar dividing both a linear-velocity squared
    error, which follows k, and an angular-velocity squared error, which follows 1/k, so
    scaling it would fix one tracking term by breaking the other by k**2. Op3's scalar
    obs_noise applies to a mixed observation vector for the same reason.

    Beyond reach of any config override, and therefore a residual difference across a
    morphology sweep: the swing-height tolerance hard coded as exp(-error / 0.01) in the
    feet_phase reward, the 0.2 s and 0.5 s feet_air_time thresholds, the 0.1 command dead
    zones, and the reset randomisation of root position and velocity. Reward weights being
    dimensionless also means terms whose value carries a unit, such as the torque and
    energy costs, still change magnitude with k, and the per-step reward is multiplied by
    dt, so an episode return is a time integral and is comparable across the sweep only
    after dividing by sqrt(k).

    Args:
        env_id: Playground environment id, used to read the base task specification.
        size_scale: factor k applied to every length in the model.
    """
    config = flatten_config(registry.get_default_config(env_id))
    overrides: dict[str, Any] = {}
    for key, exponent in TASK_SIZE_EXPONENTS.items():
        if key in config:
            overrides[key] = scaled_task_value(config[key], exponent, size_scale)
    return overrides


def morph_gain_overrides(env_id: str, params: MorphParams) -> dict[str, float]:
    """Return Kp/Kd overrides for robots whose config overwrites the morphed servo gains.

    Go1, Spot and Op3 assign config.Kp to actuator_gainprm and actuator_biasprm and
    config.Kd to dof_damping on the compiled model, before mjx.put_model, which throws
    away the gains apply_morphology had just scaled. Restating them through the config is
    the only way the morph reaches the controller of those robots. Kp is a position gain
    and follows actuator strength alone; Kd is written into dof_damping of rotational
    joints, so it belongs to the mechanism and follows mass * size**1.5. Robots that do
    not expose these keys get an empty result.

    Args:
        env_id: Playground environment id, used to read the base gains.
        params: multiplicative size, mass and torque factors.
    """
    config = flatten_config(registry.get_default_config(env_id))
    overrides: dict[str, float] = {}
    if POSITION_GAIN_CONFIG_KEY in config:
        overrides[POSITION_GAIN_CONFIG_KEY] = (
            float(config[POSITION_GAIN_CONFIG_KEY]) * params.torque_scale
        )
    if JOINT_DAMPING_CONFIG_KEY in config:
        overrides[JOINT_DAMPING_CONFIG_KEY] = (
            float(config[JOINT_DAMPING_CONFIG_KEY])
            * params.mass_scale
            * params.size_scale**ROTATIONAL_DAMPING_SIZE_EXPONENT
        )
    return overrides


def make_locomotion_env(
    robot: str = "berkeley_humanoid",
    params: MorphParams | None = None,
    config_overrides: dict[str, object] | None = None,
    scale_time: bool = False,
    scale_task: bool = False,
) -> mjx_env.MjxEnv:
    """Return a Playground locomotion env whose robot model is scaled by params.

    Neither the clock nor the task is scaled by default, and each is a separate opt-in so
    that no quantity is ever half scaled. apply_morphology deliberately leaves the
    integrator timestep and the control decimation alone, so a size-scaled robot otherwise
    runs on the base robot's clock; pass scale_time=True to stretch ctrl_dt and sim_dt by
    similar_time_scale(size_scale). The task specification is likewise authored in the base
    robot's metres and metres per second, so a size-scaled robot is otherwise rewarded for
    the base robot's foot clearance and commanded at the base robot's speed, which makes
    episode return and mean forward velocity mean different things at different sizes; pass
    scale_task=True to restate it on the scaled robot, exactly as documented in
    similarity_task_overrides. A morphology sweep meant to be dynamically similar wants
    both flags.

    Servo gains that a robot keeps in its task config are always restated, whatever the
    flags say, because Go1, Spot and Op3 write those gains onto the compiled model and
    would otherwise discard the morph rather than merely leave it unscaled.

    Explicit entries in config_overrides take precedence over all of the above.

    Args:
        robot: short robot name from available_robots.
        params: morphology factors, or None for the unmodified robot.
        config_overrides: flattened Playground config overrides, e.g. {"episode_length": 500}.
        scale_time: whether to put the robot on the dynamically similar clock.
        scale_task: whether to restate the task in the scaled robot's units.

    Returns:
        A constructed Playground environment whose mj_model and mjx_model are both morphed.

    Raises:
        ValueError: if robot is not a known name.
        RuntimeError: if the morph did not reach the model that env.step integrates.
    """
    env_id = environment_id(robot)
    params = params if params is not None else MorphParams()

    overrides: dict[str, object] = dict(morph_gain_overrides(env_id, params))
    if scale_time:
        overrides.update(similarity_time_overrides(env_id, params.size_scale))
    if scale_task:
        overrides.update(similarity_task_overrides(env_id, params.size_scale))
    if config_overrides:
        overrides.update(config_overrides)

    if params == MorphParams():
        return registry.load(env_id, config_overrides=overrides or None)

    with morphed_compilation(params) as morphed_models:
        env = registry.load(env_id, config_overrides=overrides or None)
    check_morph_reached_env(env, morphed_models)
    return env


def simulated_total_mass(env: mjx_env.MjxEnv) -> float:
    """Return the total body mass of the MJX model that env.step actually integrates."""
    return float(np.asarray(env.mjx_model.body_mass).sum())
