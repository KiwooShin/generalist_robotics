"""Factory for MuJoCo Playground locomotion environments whose robot model is morphed."""

import contextlib
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

from generalist_robotics.morphology.scaling import (
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


def available_robots() -> dict[str, str]:
    """Map short robot name -> MuJoCo Playground environment id."""
    return dict(ROBOT_ENVIRONMENTS)


def environment_id(robot: str) -> str:
    """Return the Playground environment id registered for a short robot name."""
    if robot not in ROBOT_ENVIRONMENTS:
        raise ValueError(f"Unknown robot {robot!r}. Available robots: {sorted(ROBOT_ENVIRONMENTS)}")
    return ROBOT_ENVIRONMENTS[robot]


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
    kept a morphed MjModel, and that its MJX model was put from that same model.

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
    if not np.allclose(np.asarray(env.mjx_model.body_mass), env.mj_model.body_mass):
        raise RuntimeError(
            f"{type(env).__name__}.mjx_model was not built from the morphed mj_model."
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


def make_locomotion_env(
    robot: str = "berkeley_humanoid",
    params: MorphParams | None = None,
    config_overrides: dict[str, object] | None = None,
    scale_time: bool = False,
) -> mjx_env.MjxEnv:
    """Return a Playground locomotion env whose robot model is scaled by params.

    Time is not scaled by default. apply_morphology deliberately leaves the integrator
    timestep and the control decimation alone, so a size-scaled robot otherwise runs on
    the base robot's clock and is observed and commanded at the base rate, which breaks
    dynamic similarity even when the mass and torque axes are set to preserve it. Pass
    scale_time=True to stretch ctrl_dt and sim_dt by similar_time_scale(size_scale);
    explicit entries in config_overrides take precedence over that stretch.

    Args:
        robot: short robot name from available_robots.
        params: morphology factors, or None for the unmodified robot.
        config_overrides: flattened Playground config overrides, e.g. {"episode_length": 500}.
        scale_time: whether to put the robot on the dynamically similar clock.

    Returns:
        A constructed Playground environment whose mj_model and mjx_model are both morphed.

    Raises:
        ValueError: if robot is not a known name.
        RuntimeError: if the morph did not reach the model that env.step integrates.
    """
    env_id = environment_id(robot)
    params = params if params is not None else MorphParams()

    overrides: dict[str, object] = {}
    if scale_time:
        overrides.update(similarity_time_overrides(env_id, params.size_scale))
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
