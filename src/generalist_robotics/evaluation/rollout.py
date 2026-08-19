"""Rollout harness answering whether a policy still locomotes on a given morphology."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Observation dict -> action.
Policy = Callable[[dict[str, jax.Array]], jax.Array]

# Playground locomotion envs default to 1000 control steps per episode. The raw
# env never truncates on its own, so the harness imposes the horizon.
DEFAULT_EPISODE_LENGTH = 1000


class EpisodeMetrics(NamedTuple):
    """Accumulators for a single episode, all masked at the termination step.

    Attributes:
        alive_steps: number of control steps taken before termination.
        forward_distance: net forward distance in metres over those steps.
        episode_return: summed reward over those steps.
    """

    alive_steps: jax.Array
    forward_distance: jax.Array
    episode_return: jax.Array


@dataclasses.dataclass(frozen=True)
class RolloutStats:
    """Aggregate statistics from evaluating a policy on one morphology.

    Attributes:
        survived_fraction: mean fraction of the episode completed before termination.
        mean_forward_velocity: m/s along the robot's forward axis, pooled over
            all pre-termination steps of all episodes.
        distance_travelled: mean net forward distance per episode, metres.
        episode_return: mean total reward per episode.
        num_episodes: number of episodes rolled out.
        num_steps: total environment steps executed, num_episodes * episode_length.
    """

    survived_fraction: float
    mean_forward_velocity: float
    distance_travelled: float
    episode_return: float
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


def forward_velocity(env: Any, data: Any) -> jax.Array:
    """Forward velocity of the robot base, preferring the env's own local-frame sensor."""
    local_linvel = getattr(env, "get_local_linvel", None)
    if local_linvel is not None:
        return local_linvel(data)[0]
    return free_joint_forward_velocity(data)


def episode_length_for(env: Any) -> int:
    """Episode horizon declared by the env config, or the locomotion default."""
    # Playground exposes its horizon only through the locked config object.
    config = getattr(env, "_config", None)
    length = getattr(config, "episode_length", None) if config is not None else None
    return int(length) if length is not None else DEFAULT_EPISODE_LENGTH


def rollout_episode(
    env: Any, policy: Policy, episode_length: int, rng: jax.Array
) -> EpisodeMetrics:
    """Roll one episode out, accumulating only the steps taken before termination."""
    dt = float(env.dt)
    initial_state = env.reset(rng)
    zero = jnp.zeros(())

    def step_once(carry, _):
        state, alive, alive_steps, distance, total_reward = carry
        next_state = env.step(state, policy(state.obs))
        reward = finite_or_zero(next_state.reward)
        velocity = finite_or_zero(forward_velocity(env, next_state.data))
        alive_steps = alive_steps + jnp.where(alive, 1.0, 0.0)
        distance = distance + jnp.where(alive, velocity * dt, 0.0)
        total_reward = total_reward + jnp.where(alive, reward, 0.0)
        # done is recomputed from the current state each step and can flicker back
        # to zero after a fall, so survival is latched here rather than trusted.
        alive = jnp.logical_and(alive, next_state.done < 0.5)
        return (next_state, alive, alive_steps, distance, total_reward), None

    carry = (initial_state, jnp.array(True), zero, zero, zero)
    (_, _, alive_steps, distance, total_reward), _ = jax.lax.scan(
        step_once, carry, None, length=episode_length
    )
    return EpisodeMetrics(
        alive_steps=alive_steps,
        forward_distance=distance,
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
    returns = jnp.stack([m.episode_return for m in metrics])
    alive_time = jnp.maximum(jnp.sum(alive_steps) * float(env.dt), 1e-9)

    return RolloutStats(
        survived_fraction=float(jnp.mean(alive_steps) / length),
        mean_forward_velocity=float(jnp.sum(distances) / alive_time),
        distance_travelled=float(jnp.mean(distances)),
        episode_return=float(jnp.mean(returns)),
        num_episodes=num_episodes,
        num_steps=num_episodes * length,
    )


def is_viable(
    stats: RolloutStats,
    min_survived_fraction: float = 0.8,
    min_forward_velocity: float = 0.0,
) -> bool:
    """Return True when the policy still locomotes acceptably on this morphology."""
    return bool(
        stats.survived_fraction >= min_survived_fraction
        and stats.mean_forward_velocity >= min_forward_velocity
    )
