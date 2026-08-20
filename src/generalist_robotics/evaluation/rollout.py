"""Rollout harness answering whether a policy still locomotes on a given morphology."""

import dataclasses
import inspect
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

# Observation dict -> action.
Policy = Callable[[dict[str, jax.Array]], jax.Array]

# Playground locomotion envs default to 1000 control steps per episode. The raw
# env never truncates on its own, so the harness imposes the horizon.
DEFAULT_EPISODE_LENGTH = 1000

# Gravity is the one dimensional constant morphology.scaling deliberately holds fixed,
# which is exactly what makes a k times longer robot run sqrt(k) times slower. The Froude
# number below is built on that same constant, so it inherits the similarity family.
STANDARD_GRAVITY = 9.81

# Speed floor separating walking from standing, as a Froude number Fr = v**2 / (g * L)
# with L the standing base height. Three things fix the value:
#
# 1. Similarity. Along the continuation path length scales as k and speed as sqrt(k), so
#    Fr = v**2 / (g * L) is invariant and a single number is valid for every morphology.
#    A bare m/s bar is not (0.3 m/s is a brisk walk for the 0.24 m Op3 and a shuffle for
#    the 1.08 m Apollo) and neither is body-lengths per second, which drifts as k**-0.5.
# 2. Biomechanics. Alexander's dynamic similarity puts a preferred walk near Fr = 0.25
#    and the walk-run transition near Fr = 0.5. Fr = 0.01 is a twenty-fifth of a
#    preferred walk: a deliberately permissive "clearly not standing" floor rather than a
#    performance target, so a degraded but real gait still counts as viable.
# 3. The task. Playground joystick commands are uniform over +-max forward speed, so a
#    perfectly tracking policy averages max/2 and sits at Fr = 0.050 (Berkeley Humanoid),
#    0.033 (G1), 0.053 (Apollo). Fr = 0.01 asks for 40-55% of that nominal speed, namely
#    0.155 m/s for Op3, 0.225 m/s for Berkeley Humanoid, 0.326 m/s for Apollo. A robot
#    standing still with the usual base bob lands near Fr = 1e-4, two orders below.
MIN_WALKING_FROUDE = 0.01

# Minimum fraction of the episode a viable policy must survive.
MIN_SURVIVED_FRACTION = 0.8


class EpisodeMetrics(NamedTuple):
    """Accumulators for a single episode, all masked at the termination step.

    Attributes:
        alive_steps: number of control steps taken up to and including the one that
            first reported termination.
        forward_distance: time integral of body-frame forward velocity over those steps,
            in metres. Signed, and measured along the robot's own instantaneous heading.
        forward_path_length: time integral of the magnitude of that velocity, in metres.
        net_displacement: straight-line horizontal distance in metres from the reset base
            position to the base position at the last live step.
        episode_return: summed reward over those steps.
    """

    alive_steps: jax.Array
    forward_distance: jax.Array
    forward_path_length: jax.Array
    net_displacement: jax.Array
    episode_return: jax.Array


@dataclasses.dataclass(frozen=True)
class RolloutStats:
    """Aggregate statistics from evaluating a policy on one morphology.

    Attributes:
        survived_fraction: mean fraction of the episode completed before termination.
        mean_forward_velocity: signed m/s along the robot's forward axis, pooled over all
            pre-termination steps of all episodes. Playground joystick commands are
            symmetric about zero, so a policy that tracks them perfectly averages near
            zero here; this reports which way the robot went, not how much it walked.
        mean_forward_speed: the same pooled mean taken over the magnitude of that
            velocity, m/s. This is the "is it still walking" statistic, and it does not
            cancel over forward and backward commands. Purely lateral locomotion does not
            register in it, since it is measured on the forward axis alone.
        distance_travelled: mean per episode of the time integral of forward velocity,
            metres. This is distance covered along the robot's own heading, not net
            displacement: a robot walking a circle accrues it while returning to its
            start. It is mean_forward_velocity times the mean time alive.
        net_displacement: mean per episode of the straight-line horizontal distance from
            the reset base position to the base position at termination, metres. Vertical
            motion is excluded so that falling does not read as travel. The ratio
            net_displacement / distance_travelled measures how straight the path was.
        episode_return: mean total reward per episode. Playground multiplies per-step
            reward by dt, so this is a time integral rather than a step sum; across a
            morphology sweep an episode lasts sqrt(k) times longer in seconds, so returns
            are comparable only after dividing by sqrt(k).
        nominal_leg_length: standing base height of the robot in metres, the
            characteristic length of the Froude number. 0.0 when the env exposes no
            MuJoCo model with a free-joint base.
        num_episodes: number of episodes rolled out.
        num_steps: total environment steps executed, num_episodes * episode_length.
    """

    survived_fraction: float
    mean_forward_velocity: float
    mean_forward_speed: float
    distance_travelled: float
    net_displacement: float
    episode_return: float
    nominal_leg_length: float
    num_episodes: int
    num_steps: int


def zero_policy(action_size: int) -> Policy:
    """Return a policy emitting zero actions — a baseline and a test double."""
    zeros = jnp.zeros((action_size,))

    def policy(obs: dict[str, jax.Array]) -> jax.Array:
        del obs
        return zeros

    return policy


def finite_or_zero(value: jax.Array) -> jax.Array:
    """Replace non-finite values by zero so one blown-up step cannot void a rollout."""
    return jnp.where(jnp.isfinite(value), value, 0.0)


def rotate_by_quat(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """Rotate a 3-vector by a wxyz quaternion."""
    axis = quat[1:4]
    twice_cross = 2.0 * jnp.cross(axis, vec)
    return vec + quat[0] * twice_cross + jnp.cross(axis, twice_cross)


def free_joint_forward_velocity(data: Any) -> jax.Array:
    """Forward (body x) velocity of a free-joint base read from qpos/qvel."""
    quat = data.qpos[3:7]
    inverse = quat * jnp.array([1.0, -1.0, -1.0, -1.0])
    return rotate_by_quat(inverse, data.qvel[:3])[0]


# G1 alone declares get_local_linvel(data, frame) with no default; its own
# observation code passes "pelvis".
VELOCITY_FRAME_BY_ROBOT = {"g1": "pelvis"}
DEFAULT_VELOCITY_FRAME = "pelvis"


def forward_velocity(env: Any, data: Any) -> jax.Array:
    """Forward velocity of the robot base, preferring the env's own local-frame sensor."""
    local_linvel = getattr(env, "get_local_linvel", None)
    if local_linvel is None:
        return free_joint_forward_velocity(data)
    required = [
        name
        for name, spec in inspect.signature(local_linvel).parameters.items()
        if spec.default is inspect.Parameter.empty
    ]
    if len(required) > 1:
        return local_linvel(data, DEFAULT_VELOCITY_FRAME)[0]
    return local_linvel(data)[0]


def base_position(data: Any) -> jax.Array:
    """World position of the free-joint base, the leading three entries of qpos."""
    return data.qpos[:3]


def episode_length_for(env: Any) -> int:
    """Episode horizon declared by the env config, or the locomotion default."""
    # Playground exposes its horizon only through the locked config object.
    config = getattr(env, "_config", None)
    length = getattr(config, "episode_length", None) if config is not None else None
    return int(length) if length is not None else DEFAULT_EPISODE_LENGTH


def model_of(env: Any) -> Any:
    """MuJoCo model the env was built from, preferring the CPU one, or None."""
    for name in ("mj_model", "mjx_model"):
        model = getattr(env, name, None)
        if model is not None:
            return model
    return None


def home_keyframe_qpos(model: Any) -> np.ndarray | None:
    """Reference pose of a model: its "home" keyframe, its first keyframe, or qpos0."""
    for index in range(int(getattr(model, "nkey", 0) or 0)):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, index)
        except (TypeError, ValueError):
            break
        if name == "home":
            return np.asarray(model.key_qpos[index])
    if int(getattr(model, "nkey", 0) or 0) > 0:
        return np.asarray(model.key_qpos[0])
    qpos0 = getattr(model, "qpos0", None)
    return None if qpos0 is None else np.asarray(qpos0)


def nominal_leg_length(env: Any) -> float:
    """Standing base height of the robot in metres, the Froude number's length scale.

    Froude scaling of legged gait is conventionally taken about hip height, and for these
    robots the free joint's z in the home keyframe is exactly that: 0.244 m for Op3,
    0.515 m for Berkeley Humanoid, 0.785 m for G1, 1.080 m for Apollo, 0.278 m for Go1.
    morphology.scaling scales key_qpos and qpos0 along with every other length, so this
    follows a size-scaled robot by the same factor k and keeps the Froude number
    invariant along the continuation path.

    Returns:
        The height, or 0.0 when the env exposes no MuJoCo model with a free-joint base.
    """
    model = model_of(env)
    if model is None:
        return 0.0
    joint_types = np.asarray(getattr(model, "jnt_type", np.empty(0)))
    free = np.flatnonzero(joint_types == mujoco.mjtJoint.mjJNT_FREE)
    qpos = home_keyframe_qpos(model)
    if free.size == 0 or qpos is None:
        return 0.0
    address = int(np.asarray(model.jnt_qposadr)[free[0]])
    height = float(qpos[address + 2])
    return height if np.isfinite(height) and height > 0.0 else 0.0


def froude_number(speed: float, leg_length: float, gravity: float = STANDARD_GRAVITY) -> float:
    """Dimensionless gait speed v**2 / (g * L), invariant along the similarity family.

    Raises:
        ValueError: if leg_length is not positive, which means the body scale is unknown.
    """
    if leg_length <= 0.0:
        raise ValueError(f"leg_length must be positive, got {leg_length}")
    return float(speed) ** 2 / (gravity * float(leg_length))


def rollout_episode(
    env: Any, policy: Policy, episode_length: int, rng: jax.Array
) -> EpisodeMetrics:
    """Roll one episode out, accumulating only the steps taken before termination."""
    dt = float(env.dt)
    initial_state = env.reset(rng)
    start_position = base_position(initial_state.data)
    zero = jnp.zeros(())

    def step_once(carry, _):
        state, alive, alive_steps, distance, path_length, position, total_reward = carry
        next_state = env.step(state, policy(state.obs))
        reward = finite_or_zero(next_state.reward)
        velocity = finite_or_zero(forward_velocity(env, next_state.data))
        next_position = base_position(next_state.data)
        alive_steps = alive_steps + jnp.where(alive, 1.0, 0.0)
        distance = distance + jnp.where(alive, velocity * dt, 0.0)
        path_length = path_length + jnp.where(alive, jnp.abs(velocity) * dt, 0.0)
        total_reward = total_reward + jnp.where(alive, reward, 0.0)
        # A blown-up pose must not be mistaken for the robot having walked to the origin.
        moved = jnp.logical_and(alive, jnp.all(jnp.isfinite(next_position)))
        position = jnp.where(moved, next_position, position)
        # done is recomputed from the current state each step and can flicker back
        # to zero after a fall, so survival is latched here rather than trusted.
        alive = jnp.logical_and(alive, next_state.done < 0.5)
        carry = (next_state, alive, alive_steps, distance, path_length, position, total_reward)
        return carry, None

    carry = (initial_state, jnp.array(True), zero, zero, zero, start_position, zero)
    (_, _, alive_steps, distance, path_length, position, total_reward), _ = jax.lax.scan(
        step_once, carry, None, length=episode_length
    )
    return EpisodeMetrics(
        alive_steps=alive_steps,
        forward_distance=distance,
        forward_path_length=path_length,
        net_displacement=jnp.linalg.norm((position - start_position)[:2]),
        episode_return=total_reward,
    )


def evaluate_policy(
    env: Any,
    policy: Policy,
    num_episodes: int = 8,
    seed: int = 0,
    episode_length: int | None = None,
) -> RolloutStats:
    """Roll a policy out for num_episodes and aggregate statistics."""
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    length = episode_length if episode_length is not None else episode_length_for(env)
    if length < 1:
        raise ValueError(f"episode_length must be positive, got {length}")

    rollout = jax.jit(functools.partial(rollout_episode, env, policy, length))
    rngs = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    metrics = [rollout(rng) for rng in rngs]

    alive_steps = jnp.stack([m.alive_steps for m in metrics])
    distances = jnp.stack([m.forward_distance for m in metrics])
    path_lengths = jnp.stack([m.forward_path_length for m in metrics])
    displacements = jnp.stack([m.net_displacement for m in metrics])
    returns = jnp.stack([m.episode_return for m in metrics])
    alive_time = jnp.maximum(jnp.sum(alive_steps) * float(env.dt), 1e-9)

    return RolloutStats(
        survived_fraction=float(jnp.mean(alive_steps) / length),
        mean_forward_velocity=float(jnp.sum(distances) / alive_time),
        mean_forward_speed=float(jnp.sum(path_lengths) / alive_time),
        distance_travelled=float(jnp.mean(distances)),
        net_displacement=float(jnp.mean(displacements)),
        episode_return=float(jnp.mean(returns)),
        nominal_leg_length=nominal_leg_length(env),
        num_episodes=num_episodes,
        num_steps=num_episodes * length,
    )


def is_viable(
    stats: RolloutStats,
    min_survived_fraction: float = MIN_SURVIVED_FRACTION,
    min_froude_number: float = MIN_WALKING_FROUDE,
    min_forward_speed: float | None = None,
) -> bool:
    """Return True when the policy still locomotes acceptably on this morphology.

    Locomotion is judged on mean_forward_speed rather than the signed
    mean_forward_velocity, which cancels over the symmetric joystick commands, and the
    threshold is dimensionless so that one number holds for every robot on the
    continuation path; see MIN_WALKING_FROUDE for where the value comes from.

    Args:
        stats: aggregate statistics from evaluate_policy.
        min_survived_fraction: fraction of the episode the policy must survive.
        min_froude_number: speed floor as v**2 / (g * L), ignored if min_forward_speed
            is given.
        min_forward_speed: speed floor in m/s, for envs whose body scale is unknown.

    Raises:
        ValueError: if no speed floor can be applied because stats carry no body scale
            and no explicit min_forward_speed was given.
    """
    if min_forward_speed is not None:
        walks = stats.mean_forward_speed >= min_forward_speed
    elif stats.nominal_leg_length <= 0.0:
        raise ValueError(
            "stats carry no body scale (nominal_leg_length is 0.0), so a Froude "
            "threshold cannot be applied; pass min_forward_speed in m/s instead."
        )
    else:
        speed = froude_number(stats.mean_forward_speed, stats.nominal_leg_length)
        walks = speed >= min_froude_number
    return bool(walks and stats.survived_fraction >= min_survived_fraction)
