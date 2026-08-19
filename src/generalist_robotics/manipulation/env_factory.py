"""Factory for robosuite environments that run the same task on any arm."""

from typing import Any, Dict, Optional

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from generalist_robotics.manipulation.embodiments import get_embodiment

DEFAULT_HORIZON = 500
DEFAULT_CONTROL_FREQ = 20


def make_env(
    task: str,
    arm: str,
    horizon: int = DEFAULT_HORIZON,
    control_freq: int = DEFAULT_CONTROL_FREQ,
    use_camera_obs: bool = False,
    camera_names: str = "agentview",
    camera_height: int = 128,
    camera_width: int = 128,
    seed: Optional[int] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a robosuite environment for one (task, arm) pair.

    The same task identifier works for every arm because robosuite tasks are
    robot agnostic and the operational-space controller exposes a delta
    end-effector action space of identical width regardless of joint count.

    Args:
        task: robosuite environment name, e.g. "Lift".
        arm: robosuite robot name, e.g. "Panda".
        horizon: maximum steps per episode.
        control_freq: policy control frequency in Hz.
        use_camera_obs: whether to return rendered camera observations.
        camera_names: camera to render when use_camera_obs is True.
        camera_height: rendered image height.
        camera_width: rendered image width.
        seed: optional seed for the global numpy generator robosuite samples from.
        extra_kwargs: additional keyword arguments forwarded to robosuite.make.

    Returns:
        A constructed robosuite environment.
    """
    get_embodiment(arm)
    if seed is not None:
        np.random.seed(seed)

    controller_config = load_composite_controller_config(controller=None, robot=arm)
    kwargs: Dict[str, Any] = dict(
        robots=arm,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        reward_shaping=True,
        horizon=horizon,
        control_freq=control_freq,
    )
    if use_camera_obs:
        kwargs.update(
            camera_names=camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return robosuite.make(task, **kwargs)


def action_dim(env) -> int:
    """Return the action dimensionality of a constructed environment."""
    low, _ = env.action_spec
    return int(low.shape[0])


def arm_dof(env) -> int:
    """Return the controllable joint count of the first robot in an environment."""
    return int(env.robots[0].dof)
