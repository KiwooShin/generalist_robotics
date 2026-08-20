"""Walking a locomotion policy along a path on which degrees of freedom appear."""

import dataclasses
import inspect
import json
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import mujoco
import numpy as np
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

from generalist_robotics.continuation.path import (
    ALPHA_TOLERANCE,
    CHECKPOINT_PREFIX,
    CONFIG_RECORD,
    MAX_STEP_ALPHA,
    PROGRESS_DIRNAME,
    STEP_GROWTH,
    WAYPOINT_RECORD,
    append_record,
    stats_record,
)
from generalist_robotics.envs.locomotion import environment_id, make_locomotion_env
from generalist_robotics.evaluation.rollout import RolloutStats, evaluate_policy, is_viable
from generalist_robotics.morphology.scaling import MorphParams, interpolate
from generalist_robotics.morphology.topology import (
    DofLock,
    active_dof_count,
    interpolate_locks,
    joint_id,
    joint_lock_factors,
    locked_compilation,
    robot_model,
)
from generalist_robotics.runtime.gpu import gpu_lock
from generalist_robotics.training import ppo

# Bodies a local-velocity sensor may sit on, most preferred first. G1 and T1 carry one
# per frame and ask which; the base body is the one the Froude length is measured at.
LOCAL_VELOCITY_FRAMES = ("pelvis", "torso", "base", "trunk")
LOCAL_VELOCITY_SENSOR = "local_linvel"

# Per-waypoint file recording the body and the locks that waypoint was measured on, so a
# renderer can rebuild it exactly from the checkpoint directory alone.
WAYPOINT_MORPHOLOGY_FILENAME = "morphology.json"


@dataclasses.dataclass(frozen=True)
class DofWaypoint:
    """A waypoint on a path where both the body and its active degrees of freedom change.

    Attributes:
        alpha: path coordinate, 0 at the start body and 1 at the target.
        params: size, mass and torque factors, interpolated at alpha.
        locks: how rigidly each joint group is held at alpha.
        active_dof: actuated degrees of freedom the policy effectively still has here,
            counted by morphology.topology.active_dof_count.
        stats_before: rollout statistics of the incoming policy on this body.
        viable_before: whether the incoming policy already locomoted here.
        stats_after: statistics after the last fine-tune round, or None when none was run.
        finetune_steps: environment steps spent fine-tuning at this waypoint.
        cumulative_steps: environment steps spent over the whole run up to and including
            this waypoint.
    """

    alpha: float
    params: MorphParams
    locks: tuple[DofLock, ...]
    active_dof: int
    stats_before: RolloutStats
    viable_before: bool
    stats_after: RolloutStats | None
    finetune_steps: int
    cumulative_steps: int


@dataclasses.dataclass(frozen=True)
class DofContinuationResult:
    """Outcome of walking a policy from one body and lock state to another.

    Attributes:
        waypoints: every point the walk visited, in order, rejected ones included.
        reached_target: whether the walk arrived at alpha = 1 with a viable policy.
        total_finetune_steps: environment steps spent over the whole run.
        wall_clock_seconds: seconds spent inside the walk.
        final_policy_params: the policy at the last accepted waypoint.
    """

    waypoints: list[DofWaypoint]
    reached_target: bool
    total_finetune_steps: int
    wall_clock_seconds: float
    final_policy_params: object


@dataclasses.dataclass(frozen=True)
class DofContinuationConfig:
    """Everything one waypoint visit needs that does not change along the path.

    Attributes:
        robot: short robot name from envs.locomotion.available_robots, whose model is the
            superset every point of the path is expressed in.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: fine-tune rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed; evaluation uses it directly so every body is judged on the
            same episodes, and fine-tune round r uses seed + r.
        checkpoint_dir: directory receiving per-waypoint checkpoints, or None.
        ppo_overrides: PPO hyperparameters replacing the tuned ones.
        env_kwargs: forwarded to envs.locomotion.make_locomotion_env.
    """

    robot: str
    finetune_timesteps: int
    max_finetune_rounds: int
    num_eval_episodes: int
    seed: int
    checkpoint_dir: pathlib.Path | None = None
    ppo_overrides: Mapping[str, Any] | None = None
    env_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def base_sim_dt(robot: str) -> float:
    """Simulation timestep the robot's environment integrates its unmorphed model at.

    The lock is sized against this rather than against whatever the XML happens to
    declare, because Playground overwrites the model's timestep with the config's one
    after compiling it, and the lock is a spring whose stability is a statement about
    the timestep it is actually integrated at.
    """
    return float(registry.get_default_config(environment_id(robot)).sim_dt)


def local_velocity_frame(model: mujoco.MjModel) -> str:
    """Return the body frame whose local-velocity sensor should be read for forward speed.

    Raises:
        RuntimeError: if the model carries none of the frames in LOCAL_VELOCITY_FRAMES.
    """
    for frame in LOCAL_VELOCITY_FRAMES:
        name = f"{LOCAL_VELOCITY_SENSOR}_{frame}"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0:
            return frame
    raise RuntimeError(
        f"model has no {LOCAL_VELOCITY_SENSOR} sensor on any of {LOCAL_VELOCITY_FRAMES}"
    )


def bind_local_velocity_frame(env: mjx_env.MjxEnv) -> mjx_env.MjxEnv:
    """Give the rollout harness a one-argument local-velocity accessor, and return the env.

    evaluation.rollout reads forward speed through env.get_local_linvel(data). The G1 and
    T1 environments carry that sensor on two bodies and take a second argument naming
    which, so the accessor is replaced here by one that defaults it to the base body's
    frame. It stays a two-argument call for the environment's own observation code, which
    names the frame explicitly. Environments whose accessor already takes the data alone
    are returned untouched.
    """
    accessor = getattr(env, "get_local_linvel", None)
    if accessor is None:
        return env
    required = [
        parameter
        for parameter in inspect.signature(accessor).parameters.values()
        if parameter.default is inspect.Parameter.empty
    ]
    if len(required) < 2:
        return env
    base_frame = local_velocity_frame(env.mj_model)

    def local_linvel(data: Any, frame: str = base_frame) -> jax.Array:
        """Local linear velocity of the robot, defaulting to its base body's frame."""
        return accessor(data, frame)

    env.get_local_linvel = local_linvel
    return env


def check_locks_reached_env(env: mjx_env.MjxEnv, locks: Sequence[DofLock]) -> None:
    """Raise unless the locks really are in the model that env.step integrates.

    Two links are checked rather than assumed: that the compiled model carries a spring on
    every joint the locks hold, and that the MJX model still agrees with it, so a lock
    cannot be silently left behind on the CPU model the way a morph can.

    Raises:
        RuntimeError: if the lock did not reach the simulated model.
    """
    compiled = np.asarray(env.mj_model.jnt_stiffness)
    simulated = np.asarray(env.mjx_model.jnt_stiffness)
    if not np.allclose(compiled, simulated):
        raise RuntimeError(
            f"{type(env).__name__}.mjx_model.jnt_stiffness disagrees with the locked "
            "mj_model, so the MJX model was not built from it."
        )
    for name, lock in joint_lock_factors(locks).items():
        if lock > 0.0 and compiled[joint_id(env.mj_model, name)] <= 0.0:
            raise RuntimeError(f"joint {name!r} is locked at {lock} but carries no spring")


def make_dof_env(
    robot: str,
    params: MorphParams,
    locks: Sequence[DofLock],
    **env_kwargs: Any,
) -> mjx_env.MjxEnv:
    """Build a locomotion environment whose robot is both morphed and partly locked.

    The lock is applied at compile time, outside the morph, so the morph then scales the
    lock spring and its damping exactly as it scales any other passive joint property.
    """
    with locked_compilation(tuple(locks), base_sim_dt(robot)):
        env = make_locomotion_env(robot, params, **env_kwargs)
    check_locks_reached_env(env, locks)
    return bind_local_velocity_frame(env)


def waypoint_active_dof(robot: str, locks: Sequence[DofLock]) -> int:
    """Actuated degrees of freedom the policy effectively has at one lock state.

    Counted against the robot's own unmorphed model and base timestep, so that the same
    alpha means the same count whichever body along the path carries it: a morph that is
    dynamically similar scales the lock spring and the joint inertia together and leaves
    the count where it was.
    """
    return active_dof_count(robot_model(robot), locks, base_sim_dt(robot))


def dof_waypoint_accepted(waypoint: DofWaypoint) -> bool:
    """Whether the policy locomoted at this waypoint, before or after fine-tuning."""
    if waypoint.viable_before:
        return True
    return waypoint.stats_after is not None and is_viable(waypoint.stats_after)


def evaluate_dof_morphology(
    config: DofContinuationConfig,
    params: MorphParams,
    locks: tuple[DofLock, ...],
    policy_params: object,
) -> RolloutStats:
    """Roll the policy out on one body-and-lock state and report how it did.

    The environment is rebuilt for every visit because both the morph and the lock are
    baked into the model at compile time, and the whole call holds runtime.gpu.gpu_lock.
    """
    with gpu_lock():
        env = make_dof_env(config.robot, params, locks, **dict(config.env_kwargs))
        policy = ppo.make_policy(config.robot, policy_params)
        return evaluate_policy(env, policy, num_episodes=config.num_eval_episodes, seed=config.seed)


def finetune_progress_path(
    config: DofContinuationConfig, index: int, round_index: int
) -> pathlib.Path | None:
    """Return the JSONL file one fine-tune round streams its metrics to, if any."""
    if config.checkpoint_dir is None:
        return None
    name = f"{CHECKPOINT_PREFIX}_{index:03d}_round_{round_index}.jsonl"
    return pathlib.Path(config.checkpoint_dir) / PROGRESS_DIRNAME / name


def finetune_on_dof_morphology(
    config: DofContinuationConfig,
    params: MorphParams,
    locks: tuple[DofLock, ...],
    policy_params: object,
    index: int,
    round_index: int,
) -> ppo.TrainingResult:
    """Run one warm-started PPO round on the given body-and-lock state."""
    with locked_compilation(tuple(locks), base_sim_dt(config.robot)):
        return ppo.train_policy(
            robot=config.robot,
            params=params,
            num_timesteps=config.finetune_timesteps,
            seed=config.seed + round_index,
            init_params=policy_params,
            progress_path=finetune_progress_path(config, index, round_index),
            ppo_overrides=config.ppo_overrides,
            **dict(config.env_kwargs),
        )


def visit_dof_morphology(
    config: DofContinuationConfig,
    index: int,
    alpha: float,
    params: MorphParams,
    locks: tuple[DofLock, ...],
    policy_params: object,
    cumulative_steps: int,
    correct: bool = True,
) -> tuple[DofWaypoint, object]:
    """Test the policy on one body-and-lock state and fine-tune it there until it walks.

    This is the corrector of the predictor-corrector loop; the predictor is the step in
    alpha, which now moves the body and the locks together.

    Args:
        config: run-wide settings.
        index: position of this visit in the walk, used to name its artifacts.
        alpha: path coordinate.
        params: the morphology to visit.
        locks: the lock state to visit.
        policy_params: the incoming policy.
        cumulative_steps: environment steps spent before this visit.
        correct: whether a failure may be corrected by fine-tuning. False measures the
            state without touching the policy, which is what the anchor at alpha = 0 is for.

    Returns:
        The waypoint, and the policy as it stands at the end of the visit.
    """
    stats_before = evaluate_dof_morphology(config, params, locks, policy_params)
    viable_before = is_viable(stats_before)
    trial_params = policy_params
    stats_after: RolloutStats | None = None
    spent = 0

    rounds = 0 if viable_before or not correct else config.max_finetune_rounds
    for round_index in range(rounds):
        result = finetune_on_dof_morphology(config, params, locks, trial_params, index, round_index)
        trial_params = result.params
        spent += result.num_timesteps
        stats_after = evaluate_dof_morphology(config, params, locks, trial_params)
        if is_viable(stats_after):
            break

    waypoint = DofWaypoint(
        alpha=float(alpha),
        params=params,
        locks=tuple(locks),
        active_dof=waypoint_active_dof(config.robot, locks),
        stats_before=stats_before,
        viable_before=viable_before,
        stats_after=stats_after,
        finetune_steps=spent,
        cumulative_steps=cumulative_steps + spent,
    )
    return waypoint, trial_params


def morphology_record(waypoint: DofWaypoint) -> dict[str, Any]:
    """Return the body and lock state of one waypoint, as a JSON-serialisable document."""
    return {
        "alpha": waypoint.alpha,
        "params": dataclasses.asdict(waypoint.params),
        "locks": [dataclasses.asdict(lock) for lock in waypoint.locks],
        "active_dof": waypoint.active_dof,
    }


def save_dof_waypoint_checkpoint(
    config: DofContinuationConfig,
    waypoint: DofWaypoint,
    policy_params: object,
    index: int,
    seconds: float,
) -> pathlib.Path | None:
    """Persist the policy at one waypoint, next to the body and locks it was measured on.

    training.ppo.save_checkpoint stores only scalar metrics, and a lock is a set of joint
    names, so the full body-and-lock state is written beside it as its own document; the
    checkpoint directory alone is then enough to rebuild the robot the policy ran on.
    """
    if config.checkpoint_dir is None:
        return None
    stats = waypoint.stats_after if waypoint.stats_after is not None else waypoint.stats_before
    metrics = {
        "alpha": waypoint.alpha,
        "active_dof": float(waypoint.active_dof),
        "cumulative_steps": float(waypoint.cumulative_steps),
        "survived_fraction": stats.survived_fraction,
        "mean_forward_speed": stats.mean_forward_speed,
        "episode_return": stats.episode_return,
        "size_scale": waypoint.params.size_scale,
        "mass_scale": waypoint.params.mass_scale,
        "torque_scale": waypoint.params.torque_scale,
    }
    result = ppo.TrainingResult(
        params=policy_params,
        metrics=metrics,
        num_timesteps=waypoint.finetune_steps,
        wall_clock_seconds=seconds,
        steps_per_second=waypoint.finetune_steps / seconds if seconds > 0.0 else 0.0,
    )
    directory = pathlib.Path(config.checkpoint_dir) / f"{CHECKPOINT_PREFIX}_{index:03d}"
    saved = ppo.save_checkpoint(result, directory)
    (saved / WAYPOINT_MORPHOLOGY_FILENAME).write_text(
        json.dumps(morphology_record(waypoint), indent=2)
    )
    return saved


def dof_waypoint_record(
    waypoint: DofWaypoint, index: int, checkpoint: pathlib.Path | None = None
) -> dict[str, Any]:
    """Return the JSON record of one waypoint: what it measured and what it cost."""
    record = {
        "record": WAYPOINT_RECORD,
        "index": index,
        "viable_before": waypoint.viable_before,
        "accepted": dof_waypoint_accepted(waypoint),
        "finetune_steps": waypoint.finetune_steps,
        "cumulative_steps": waypoint.cumulative_steps,
        "stats_before": stats_record(waypoint.stats_before),
        "stats_after": stats_record(waypoint.stats_after),
        "checkpoint": None if checkpoint is None else str(checkpoint),
    }
    record.update(morphology_record(waypoint))
    return record


def dof_config_record(
    config: DofContinuationConfig,
    start_params: MorphParams,
    end_params: MorphParams,
    start_locks: tuple[DofLock, ...],
    end_locks: tuple[DofLock, ...],
    step_alpha: float,
    min_step_alpha: float,
) -> dict[str, Any]:
    """Return the JSON record heading a run log: how to rebuild every state it visits."""
    return {
        "record": CONFIG_RECORD,
        "robot": config.robot,
        "start_params": dataclasses.asdict(start_params),
        "end_params": dataclasses.asdict(end_params),
        "start_locks": [dataclasses.asdict(lock) for lock in start_locks],
        "end_locks": [dataclasses.asdict(lock) for lock in end_locks],
        "step_alpha": step_alpha,
        "min_step_alpha": min_step_alpha,
        "finetune_timesteps": config.finetune_timesteps,
        "max_finetune_rounds": config.max_finetune_rounds,
        "num_eval_episodes": config.num_eval_episodes,
        "seed": config.seed,
        "env_kwargs": dict(config.env_kwargs),
        "ppo_overrides": dict(config.ppo_overrides) if config.ppo_overrides else None,
        "checkpoint_dir": (None if config.checkpoint_dir is None else str(config.checkpoint_dir)),
    }


def record_dof_waypoint(
    config: DofContinuationConfig,
    log: pathlib.Path | None,
    waypoint: DofWaypoint,
    index: int,
    policy_params: object,
    seconds: float,
) -> None:
    """Checkpoint the policy at a waypoint and append the waypoint to the run log."""
    checkpoint = save_dof_waypoint_checkpoint(config, waypoint, policy_params, index, seconds)
    if log is not None:
        append_record(log, dof_waypoint_record(waypoint, index, checkpoint))


def walk_dof_path(
    robot: str,
    start_params: MorphParams,
    end_params: MorphParams,
    start_locks: tuple[DofLock, ...],
    end_locks: tuple[DofLock, ...],
    init_policy_params: object,
    step_alpha: float = 0.1,
    min_step_alpha: float = 0.01,
    finetune_timesteps: int = 5_000_000,
    max_finetune_rounds: int = 3,
    num_eval_episodes: int = 8,
    seed: int = 0,
    log_path: pathlib.Path | None = None,
    *,
    checkpoint_dir: pathlib.Path | None = None,
    step_growth: float = STEP_GROWTH,
    max_step_alpha: float = MAX_STEP_ALPHA,
    ppo_overrides: Mapping[str, Any] | None = None,
    **env_kwargs: Any,
) -> DofContinuationResult:
    """Walk a policy along a path on which the body morphs and degrees of freedom appear.

    This is continuation.path.walk_morphology_path with one more axis. The path still runs
    in a single robot model, so the policy's action width never changes; what changes is
    how much of that width is connected to anything. A joint held at high stiffness with
    its actuator gain scaled to zero is absent, and annealing the lock down grows the
    degree of freedom back continuously, so a policy trained on G1's twelve leg joints can
    be carried onto the full twenty-nine without ever facing a discontinuity in the model.

    The step control is deliberately the one walk_morphology_path uses, because the whole
    point of the milestone is that only the path changed: the predictor takes a step of
    step_alpha in alpha, moving the morphology and the locks together; the corrector asks
    evaluation.rollout.is_viable whether the policy still locomotes and fine-tunes it there
    for up to max_finetune_rounds rounds if not; a waypoint reached for free grows the
    stride by step_growth up to max_step_alpha; a waypoint that needed training holds it;
    and a rejected waypoint halves it and the walk retreats to the last accepted state,
    stopping when halving would go below min_step_alpha.

    Args:
        robot: short robot name whose model hosts the whole path.
        start_params: morphology the incoming policy was trained on.
        end_params: morphology to arrive at.
        start_locks: how rigidly each joint group is held at alpha = 0.
        end_locks: how rigidly each joint group is held at alpha = 1.
        init_policy_params: brax parameters in the layout training.ppo produces.
        step_alpha: initial stride in path coordinate.
        min_step_alpha: stride below which the walk gives up.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed for evaluation and training.
        log_path: JSONL file receiving a config record and then one record per waypoint,
            written as each waypoint completes.
        checkpoint_dir: directory receiving one checkpoint per waypoint, each carrying its
            own morphology and locks, or None to keep the run in memory.
        step_growth: factor the stride grows by after a free waypoint.
        max_step_alpha: ceiling for that growth.
        ppo_overrides: PPO hyperparameters replacing the tuned ones.
        **env_kwargs: forwarded to make_locomotion_env; a similarity path wants
            scale_time=True and scale_task=True.

    Returns:
        Every waypoint visited, whether the target was reached, the total environment-step
        cost, the wall clock, and the policy at the last accepted waypoint.

    Raises:
        ValueError: if the stride, the stride floor, the round budget or the fine-tune
            budget is not usable.
    """
    if not 0.0 < step_alpha <= 1.0:
        raise ValueError(f"step_alpha must be in (0, 1], got {step_alpha!r}")
    if not 0.0 < min_step_alpha <= step_alpha:
        raise ValueError(
            f"min_step_alpha must be in (0, step_alpha], got {min_step_alpha!r} "
            f"with step_alpha {step_alpha!r}"
        )
    if step_growth < 1.0:
        raise ValueError(f"step_growth must be at least 1, got {step_growth!r}")
    if max_finetune_rounds < 0:
        raise ValueError(f"max_finetune_rounds must not be negative, got {max_finetune_rounds!r}")
    if finetune_timesteps < 1:
        raise ValueError(f"finetune_timesteps must be positive, got {finetune_timesteps!r}")

    config = DofContinuationConfig(
        robot=robot,
        finetune_timesteps=int(finetune_timesteps),
        max_finetune_rounds=int(max_finetune_rounds),
        num_eval_episodes=int(num_eval_episodes),
        seed=int(seed),
        checkpoint_dir=None if checkpoint_dir is None else pathlib.Path(checkpoint_dir),
        ppo_overrides=ppo_overrides,
        env_kwargs=dict(env_kwargs),
    )
    log = pathlib.Path(log_path) if log_path is not None else None
    if log is not None:
        append_record(
            log,
            dof_config_record(
                config, start_params, end_params, start_locks, end_locks, step_alpha, min_step_alpha
            ),
        )

    started = time.perf_counter()
    waypoints: list[DofWaypoint] = []
    policy_params = init_policy_params
    cumulative = 0
    alpha = 0.0
    stride = float(step_alpha)
    ceiling = max(float(step_alpha), float(max_step_alpha))
    reached = False

    anchor, _ = visit_dof_morphology(
        config, 0, 0.0, start_params, tuple(start_locks), policy_params, cumulative, correct=False
    )
    waypoints.append(anchor)
    record_dof_waypoint(config, log, anchor, 0, policy_params, time.perf_counter() - started)

    while alpha < 1.0 - ALPHA_TOLERANCE:
        candidate = min(1.0, alpha + stride)
        index = len(waypoints)
        visit_started = time.perf_counter()
        waypoint, trial_params = visit_dof_morphology(
            config,
            index,
            candidate,
            interpolate(start_params, end_params, candidate),
            interpolate_locks(tuple(start_locks), tuple(end_locks), candidate),
            policy_params,
            cumulative,
        )
        cumulative = waypoint.cumulative_steps
        waypoints.append(waypoint)
        accepted = dof_waypoint_accepted(waypoint)
        record_dof_waypoint(
            config, log, waypoint, index, trial_params, time.perf_counter() - visit_started
        )
        # Every waypoint compiles a fresh model, and the traces of the ones behind us are
        # dead weight on a machine whose GPU memory is its system memory.
        jax.clear_caches()

        if accepted:
            policy_params = trial_params
            alpha = candidate
            if waypoint.finetune_steps == 0:
                stride = min(stride * step_growth, ceiling)
            reached = alpha >= 1.0 - ALPHA_TOLERANCE
        else:
            stride = (candidate - alpha) / 2.0
            if stride < min_step_alpha:
                break

    return DofContinuationResult(
        waypoints=waypoints,
        reached_target=reached,
        total_finetune_steps=cumulative,
        wall_clock_seconds=time.perf_counter() - started,
        final_policy_params=policy_params,
    )


def save_dof_run_log(result: DofContinuationResult, path: pathlib.Path) -> pathlib.Path:
    """Write a finished walk as one JSON document.

    The JSONL streamed during the run stays the primary record: it also carries the run
    configuration and the per-waypoint checkpoint paths, which a result does not hold.

    Args:
        result: the finished walk.
        path: JSON file to write, creating parent directories.

    Returns:
        The absolute path written.
    """
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "reached_target": result.reached_target,
        "total_finetune_steps": result.total_finetune_steps,
        "wall_clock_seconds": result.wall_clock_seconds,
        "num_waypoints": len(result.waypoints),
        "num_accepted_waypoints": sum(dof_waypoint_accepted(w) for w in result.waypoints),
        "waypoints": [
            dof_waypoint_record(waypoint, index) for index, waypoint in enumerate(result.waypoints)
        ],
    }
    path.write_text(json.dumps(document, indent=2))
    return path
