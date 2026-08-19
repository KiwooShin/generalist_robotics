"""Walking a locomotion policy from one morphology to another by numerical continuation."""

import dataclasses
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any

from generalist_robotics.envs.locomotion import make_locomotion_env
from generalist_robotics.evaluation.rollout import RolloutStats, evaluate_policy, is_viable
from generalist_robotics.morphology.scaling import MorphParams, interpolate
from generalist_robotics.runtime.gpu import gpu_lock
from generalist_robotics.training import ppo

# The path coordinate is exact at both ends, but repeated halving can land a hair short of
# the target; this is the width of "arrived".
ALPHA_TOLERANCE = 1e-9

# Stride control. The stride grows only after a waypoint the policy reached for free and
# halves whenever a waypoint is rejected, so it measures how wide the viable basin around
# the current policy actually is: bold where transfer is free, cautious where it is not.
STEP_GROWTH = 1.5
MAX_STEP_ALPHA = 0.5

# Record kinds in the streaming JSONL log.
CONFIG_RECORD = "config"
WAYPOINT_RECORD = "waypoint"

# Per-waypoint checkpoint directory name, so a renderer can pair a policy with a waypoint.
CHECKPOINT_PREFIX = "waypoint"
PROGRESS_DIRNAME = "progress"

# Keys under which a waypoint checkpoint's metadata carries its own morphology, so the
# checkpoint alone is enough to rebuild the body the policy was last viable on.
CHECKPOINT_METRIC_KEYS = ("size_scale", "mass_scale", "torque_scale")


@dataclasses.dataclass(frozen=True)
class Waypoint:
    """One morphology visited along the path, and what it cost to stay viable there.

    Attributes:
        alpha: path coordinate, 0 at the start morphology and 1 at the target.
        params: the morphology itself, interpolated at alpha.
        stats_before: rollout statistics of the incoming policy on this body.
        viable_before: whether the incoming policy already locomoted here.
        stats_after: statistics after the last fine-tune round, or None when no fine-tune
            was needed.
        finetune_steps: environment steps spent fine-tuning at this waypoint; 0 when the
            policy transferred for free.
        cumulative_steps: environment steps spent over the whole run up to and including
            this waypoint.
    """

    alpha: float
    params: MorphParams
    stats_before: RolloutStats
    viable_before: bool
    stats_after: RolloutStats | None
    finetune_steps: int
    cumulative_steps: int


@dataclasses.dataclass(frozen=True)
class ContinuationResult:
    """Outcome of walking a policy from one morphology to another.

    Attributes:
        waypoints: every morphology the walk visited, in the order it visited them. A
            rejected waypoint stays in the list, because the steps it burned are part of
            the cost and the retreat is part of the trajectory; waypoint_accepted tells
            the two apart.
        reached_target: whether the walk arrived at alpha = 1 with a viable policy.
        total_finetune_steps: environment steps spent over the whole run, rejected
            waypoints included.
        wall_clock_seconds: seconds spent inside the walk.
        final_policy_params: the policy at the last accepted waypoint.
    """

    waypoints: list[Waypoint]
    reached_target: bool
    total_finetune_steps: int
    wall_clock_seconds: float
    final_policy_params: object


@dataclasses.dataclass(frozen=True)
class ContinuationConfig:
    """Everything a single waypoint visit needs that does not change along the path.

    Attributes:
        robot: short robot name from envs.locomotion.available_robots.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: fine-tune rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed; evaluation uses it directly so that every morphology is
            judged on the same episodes, and fine-tune round r uses seed + r so a repeated
            round is a genuinely new attempt rather than the same one.
        checkpoint_dir: directory receiving per-waypoint policy checkpoints and per-round
            training progress, or None to keep the run in memory.
        ppo_overrides: PPO hyperparameters replacing the tuned ones, e.g.
            ppo.SMOKE_PPO_OVERRIDES.
        env_kwargs: forwarded to envs.locomotion.make_locomotion_env, i.e.
            config_overrides, scale_time and scale_task.
    """

    robot: str
    finetune_timesteps: int
    max_finetune_rounds: int
    num_eval_episodes: int
    seed: int
    checkpoint_dir: pathlib.Path | None = None
    ppo_overrides: Mapping[str, Any] | None = None
    env_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def waypoint_accepted(waypoint: Waypoint) -> bool:
    """Whether the policy locomoted at this waypoint, before or after fine-tuning."""
    if waypoint.viable_before:
        return True
    return waypoint.stats_after is not None and is_viable(waypoint.stats_after)


def evaluate_morphology(
    config: ContinuationConfig, params: MorphParams, policy_params: object
) -> RolloutStats:
    """Roll the policy out on one morphology and report whether it still locomotes.

    The environment is rebuilt for every visit because a morphology is baked into the
    model at compile time, and the whole call holds runtime.gpu.gpu_lock so that nothing
    else on this machine runs GPU work beside it.
    """
    with gpu_lock():
        env = make_locomotion_env(config.robot, params, **dict(config.env_kwargs))
        policy = ppo.make_policy(config.robot, policy_params)
        return evaluate_policy(env, policy, num_episodes=config.num_eval_episodes, seed=config.seed)


def finetune_progress_path(
    config: ContinuationConfig, index: int, round_index: int
) -> pathlib.Path | None:
    """Return the JSONL file one fine-tune round streams its metrics to, if any."""
    if config.checkpoint_dir is None:
        return None
    name = f"{CHECKPOINT_PREFIX}_{index:03d}_round_{round_index}.jsonl"
    return pathlib.Path(config.checkpoint_dir) / PROGRESS_DIRNAME / name


def finetune_on_morphology(
    config: ContinuationConfig,
    params: MorphParams,
    policy_params: object,
    index: int,
    round_index: int,
) -> ppo.TrainingResult:
    """Run one warm-started PPO round on the given morphology."""
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


def visit_morphology(
    config: ContinuationConfig,
    index: int,
    alpha: float,
    params: MorphParams,
    policy_params: object,
    cumulative_steps: int,
    correct: bool = True,
) -> tuple[Waypoint, object]:
    """Test the policy on one morphology and fine-tune it there until it locomotes again.

    This is the corrector of the predictor-corrector loop: the predictor is the step in
    alpha, and this brings the policy back onto the "still walks" solution branch.

    Args:
        config: run-wide settings.
        index: position of this visit in the walk, used to name its artifacts.
        alpha: path coordinate of the morphology.
        params: the morphology to visit.
        policy_params: the incoming policy.
        cumulative_steps: environment steps spent before this visit.
        correct: whether a failure may be corrected by fine-tuning. False measures the
            morphology without touching the policy, which is what the anchor at alpha = 0
            is for.

    Returns:
        The waypoint, and the policy as it stands at the end of the visit. That policy is
        the incoming one when no fine-tune was needed, and the fine-tuned one otherwise,
        including when the waypoint is rejected: the caller decides what to keep.
    """
    stats_before = evaluate_morphology(config, params, policy_params)
    viable_before = is_viable(stats_before)
    trial_params = policy_params
    stats_after: RolloutStats | None = None
    spent = 0

    rounds = 0 if viable_before or not correct else config.max_finetune_rounds
    for round_index in range(rounds):
        result = finetune_on_morphology(config, params, trial_params, index, round_index)
        trial_params = result.params
        spent += result.num_timesteps
        stats_after = evaluate_morphology(config, params, trial_params)
        if is_viable(stats_after):
            break

    waypoint = Waypoint(
        alpha=float(alpha),
        params=params,
        stats_before=stats_before,
        viable_before=viable_before,
        stats_after=stats_after,
        finetune_steps=spent,
        cumulative_steps=cumulative_steps + spent,
    )
    return waypoint, trial_params


def save_waypoint_checkpoint(
    config: ContinuationConfig,
    waypoint: Waypoint,
    policy_params: object,
    index: int,
    seconds: float,
) -> pathlib.Path | None:
    """Persist the policy as it stood at one waypoint, for replay and for resuming.

    The checkpoint carries the waypoint's morphology and rollout statistics in its
    metadata, so a renderer can rebuild the body from the checkpoint alone.
    """
    if config.checkpoint_dir is None:
        return None
    stats = waypoint.stats_after if waypoint.stats_after is not None else waypoint.stats_before
    metrics = {
        "alpha": waypoint.alpha,
        "cumulative_steps": float(waypoint.cumulative_steps),
        "survived_fraction": stats.survived_fraction,
        "mean_forward_speed": stats.mean_forward_speed,
        "episode_return": stats.episode_return,
    }
    metrics.update({key: getattr(waypoint.params, key) for key in CHECKPOINT_METRIC_KEYS})
    result = ppo.TrainingResult(
        params=policy_params,
        metrics=metrics,
        num_timesteps=waypoint.finetune_steps,
        wall_clock_seconds=seconds,
        steps_per_second=waypoint.finetune_steps / seconds if seconds > 0.0 else 0.0,
    )
    directory = pathlib.Path(config.checkpoint_dir) / f"{CHECKPOINT_PREFIX}_{index:03d}"
    return ppo.save_checkpoint(result, directory)


def stats_record(stats: RolloutStats | None) -> dict[str, Any] | None:
    """Return a JSON-serialisable view of rollout statistics."""
    return None if stats is None else dataclasses.asdict(stats)


def waypoint_record(
    waypoint: Waypoint, index: int, checkpoint: pathlib.Path | None = None
) -> dict[str, Any]:
    """Return the JSON record of one waypoint: what it measured and what it cost."""
    return {
        "record": WAYPOINT_RECORD,
        "index": index,
        "alpha": waypoint.alpha,
        "params": dataclasses.asdict(waypoint.params),
        "viable_before": waypoint.viable_before,
        "accepted": waypoint_accepted(waypoint),
        "finetune_steps": waypoint.finetune_steps,
        "cumulative_steps": waypoint.cumulative_steps,
        "stats_before": stats_record(waypoint.stats_before),
        "stats_after": stats_record(waypoint.stats_after),
        "checkpoint": None if checkpoint is None else str(checkpoint),
    }


def config_record(
    config: ContinuationConfig,
    start: MorphParams,
    end: MorphParams,
    step_alpha: float,
    min_step_alpha: float,
) -> dict[str, Any]:
    """Return the JSON record heading a run log: how to rebuild every body it visits."""
    return {
        "record": CONFIG_RECORD,
        "robot": config.robot,
        "start": dataclasses.asdict(start),
        "end": dataclasses.asdict(end),
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


def append_record(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to a JSONL log, flushed so a killed run stays readable."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def record_waypoint(
    config: ContinuationConfig,
    log: pathlib.Path | None,
    waypoint: Waypoint,
    index: int,
    policy_params: object,
    seconds: float,
) -> None:
    """Checkpoint the policy at a waypoint and append the waypoint to the run log."""
    checkpoint = save_waypoint_checkpoint(config, waypoint, policy_params, index, seconds)
    if log is not None:
        append_record(log, waypoint_record(waypoint, index, checkpoint))


def walk_morphology_path(
    robot: str,
    start: MorphParams,
    end: MorphParams,
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
) -> ContinuationResult:
    """Walk a policy from the start morphology to the end morphology by continuation.

    The path is the geometric interpolation morphology.interpolate draws between the two
    bodies, and the walk is a predictor-corrector march along it. The predictor takes a
    step of step_alpha; the corrector asks evaluation.rollout.is_viable whether the policy
    still locomotes on the new body and, when it does not, fine-tunes it there for up to
    max_finetune_rounds rounds. Viability is therefore the pacing signal of a self-paced
    curriculum: nothing else decides how far the next step goes or how much training the
    walk buys.

    Step control:
      - a waypoint reached for free grows the stride by step_growth, capped at
        max_step_alpha, because the viable basin is evidently wider than assumed;
      - a waypoint that needed fine-tuning holds the stride;
      - a rejected waypoint halves it and the walk retries from the last accepted
        morphology with the last accepted policy, discarding the fine-tuning done on the
        body it could not reach. That reversion is what keeps the walk on the solution
        branch rather than dragging a broken policy forward.
    When halving would put the stride below min_step_alpha the walk stops and reports
    reached_target False.

    Args:
        robot: short robot name from envs.locomotion.available_robots.
        start: morphology the incoming policy was trained on.
        end: morphology to arrive at.
        init_policy_params: brax parameters in the layout training.ppo produces.
        step_alpha: initial stride in path coordinate.
        min_step_alpha: stride below which the walk gives up.
        finetune_timesteps: environment-step budget of one fine-tune round.
        max_finetune_rounds: rounds attempted before a waypoint is rejected.
        num_eval_episodes: episodes per viability evaluation.
        seed: base PRNG seed for evaluation and training.
        log_path: JSONL file receiving a config record and then one record per waypoint,
            written as each waypoint completes so a run that dies partway is analysable.
        checkpoint_dir: directory receiving one policy checkpoint per waypoint and the
            per-round training progress, or None to keep the run in memory.
        step_growth: factor the stride grows by after a free waypoint.
        max_step_alpha: ceiling for that growth.
        ppo_overrides: PPO hyperparameters replacing the tuned ones.
        **env_kwargs: forwarded to make_locomotion_env; a similarity sweep wants
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

    config = ContinuationConfig(
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
        append_record(log, config_record(config, start, end, step_alpha, min_step_alpha))

    started = time.perf_counter()
    waypoints: list[Waypoint] = []
    policy_params = init_policy_params
    cumulative = 0
    alpha = 0.0
    stride = float(step_alpha)
    ceiling = max(float(step_alpha), float(max_step_alpha))
    reached = False

    # The anchor measures the policy on its own body: the reference every later waypoint
    # is read against, and never itself corrected.
    anchor, _ = visit_morphology(config, 0, 0.0, start, policy_params, cumulative, correct=False)
    waypoints.append(anchor)
    record_waypoint(config, log, anchor, 0, policy_params, time.perf_counter() - started)

    while alpha < 1.0 - ALPHA_TOLERANCE:
        candidate = min(1.0, alpha + stride)
        index = len(waypoints)
        visit_started = time.perf_counter()
        waypoint, trial_params = visit_morphology(
            config,
            index,
            candidate,
            interpolate(start, end, candidate),
            policy_params,
            cumulative,
        )
        cumulative = waypoint.cumulative_steps
        waypoints.append(waypoint)
        accepted = waypoint_accepted(waypoint)
        record_waypoint(
            config, log, waypoint, index, trial_params, time.perf_counter() - visit_started
        )

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

    return ContinuationResult(
        waypoints=waypoints,
        reached_target=reached,
        total_finetune_steps=cumulative,
        wall_clock_seconds=time.perf_counter() - started,
        final_policy_params=policy_params,
    )


def save_run_log(result: ContinuationResult, path: pathlib.Path) -> pathlib.Path:
    """Write a finished walk as one JSON document.

    This is the summary; the JSONL streamed during the run is the primary record and is
    the one that also carries the run configuration and the per-waypoint checkpoint paths,
    which a ContinuationResult does not hold.

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
        "num_accepted_waypoints": sum(waypoint_accepted(w) for w in result.waypoints),
        "waypoints": [
            waypoint_record(waypoint, index) for index, waypoint in enumerate(result.waypoints)
        ],
    }
    path.write_text(json.dumps(document, indent=2))
    return path
