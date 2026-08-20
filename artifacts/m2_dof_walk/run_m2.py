"""Driver for milestone M2: walk a G1 policy from a locked, Berkeley-scale biped to full G1.

Stage 1 trains the start policy on G1 with its waist and arms held rigid and its body
morphed to Berkeley Humanoid's scale on the dynamic-similarity manifold. Stage 2 walks
that policy to nominal G1 with everything free. Stage 3 is the control: the same target in
one jump, on the step budget the walk spent.

Run one stage at a time: python run_m2.py {start,walk,jump}.
"""

import dataclasses
import json
import pathlib
import sys
import time

import jax

import generalist_robotics  # noqa: F401  caps JAX memory before any backend starts
from generalist_robotics.continuation import dof_path
from generalist_robotics.evaluation.rollout import evaluate_policy, froude_number, is_viable
from generalist_robotics.morphology.scaling import MorphParams, dynamic_similarity_params
from generalist_robotics.morphology.topology import DofLock, joint_group, locked_compilation
from generalist_robotics.training import ppo

ROBOT = "g1"
ROOT = pathlib.Path(__file__).resolve().parent.parent
START_DIR = ROOT / "m2_start_policy"
WALK_DIR = ROOT / "m2_dof_walk"
JUMP_DIR = ROOT / "m2_one_jump"

# Berkeley Humanoid stands 0.515 m at the hip, G1 0.785 m. The start body is G1 shrunk to
# that height on the dynamic-similarity manifold, so mass and torque follow k**3 and k**4
# and the whole path stays on the ridge M1 measured as the cheap one.
SIZE_SCALE = 0.515 / 0.785
START_PARAMS = dynamic_similarity_params(SIZE_SCALE)
END_PARAMS = MorphParams()

EVAL_EPISODES = 8
SEED = 0
ENV_KWARGS = {"scale_time": True, "scale_task": True}

FINETUNE_TIMESTEPS = 5_000_000
MAX_FINETUNE_ROUNDS = 3
STEP_ALPHA = 0.1
MIN_STEP_ALPHA = 0.025


def path_locks() -> tuple[tuple[DofLock, ...], tuple[DofLock, ...]]:
    """The lock state at both ends: waist and arms rigid, then both free."""
    waist = joint_group(ROBOT, "waist")
    arms = joint_group(ROBOT, "arms")
    start = (DofLock(waist, 1.0), DofLock(arms, 1.0))
    end = (DofLock(waist, 0.0), DofLock(arms, 0.0))
    return start, end


def report(directory: pathlib.Path, params: MorphParams, locks, policy_params, extra: dict) -> dict:
    """Evaluate a policy on one body-and-lock state and write the verdict beside it."""
    env = dof_path.make_dof_env(ROBOT, params, locks, **ENV_KWARGS)
    started = time.perf_counter()
    policy = ppo.make_policy(ROBOT, policy_params)
    stats = evaluate_policy(env, policy, num_episodes=EVAL_EPISODES, seed=SEED)
    document = {
        "robot": ROBOT,
        "params": dataclasses.asdict(params),
        "locks": [dataclasses.asdict(lock) for lock in locks],
        "active_dof": dof_path.waypoint_active_dof(ROBOT, locks),
        "stats": dataclasses.asdict(stats),
        "froude": froude_number(stats.mean_forward_speed, stats.nominal_leg_length),
        "is_viable": is_viable(stats),
        "eval_seconds": time.perf_counter() - started,
        **extra,
    }
    (directory / "baseline.json").write_text(json.dumps(document, indent=2))
    return document


def run_start() -> None:
    """Stage 1: train the legs-only, Berkeley-scale start policy with the tuned config."""
    start_locks, _ = path_locks()
    START_DIR.mkdir(parents=True, exist_ok=True)
    with locked_compilation(start_locks, dof_path.base_sim_dt(ROBOT)):
        result = ppo.train_policy(
            robot=ROBOT,
            params=START_PARAMS,
            num_timesteps=None,
            seed=SEED,
            progress_path=START_DIR / "progress.jsonl",
            **ENV_KWARGS,
        )
    ppo.save_checkpoint(result, START_DIR)
    document = report(
        START_DIR,
        START_PARAMS,
        start_locks,
        result.params,
        {
            "num_timesteps": result.num_timesteps,
            "wall_clock_seconds": result.wall_clock_seconds,
            "steps_per_second": result.steps_per_second,
            "metrics": result.metrics,
        },
    )
    print(json.dumps({k: v for k, v in document.items() if k != "metrics"}, indent=2), flush=True)
    print("final reward", result.metrics.get(ppo.EVAL_REWARD_KEY), flush=True)


def run_walk() -> None:
    """Stage 2: walk the start policy to nominal G1, growing seventeen degrees of freedom."""
    start_locks, end_locks = path_locks()
    params, _ = ppo.load_checkpoint(START_DIR)
    WALK_DIR.mkdir(parents=True, exist_ok=True)
    result = dof_path.walk_dof_path(
        robot=ROBOT,
        start_params=START_PARAMS,
        end_params=END_PARAMS,
        start_locks=start_locks,
        end_locks=end_locks,
        init_policy_params=params,
        step_alpha=STEP_ALPHA,
        min_step_alpha=MIN_STEP_ALPHA,
        finetune_timesteps=FINETUNE_TIMESTEPS,
        max_finetune_rounds=MAX_FINETUNE_ROUNDS,
        num_eval_episodes=EVAL_EPISODES,
        seed=SEED,
        log_path=WALK_DIR / "run.jsonl",
        checkpoint_dir=WALK_DIR,
        **ENV_KWARGS,
    )
    dof_path.save_dof_run_log(result, WALK_DIR / "run.json")
    for waypoint in result.waypoints:
        print(
            f"alpha={waypoint.alpha:.4f} dof={waypoint.active_dof:2d} "
            f"viable_before={waypoint.viable_before} "
            f"accepted={dof_path.dof_waypoint_accepted(waypoint)} "
            f"steps={waypoint.finetune_steps} cumulative={waypoint.cumulative_steps}",
            flush=True,
        )
    print(
        f"reached_target={result.reached_target} total_steps={result.total_finetune_steps} "
        f"wall_clock={result.wall_clock_seconds:.1f}s",
        flush=True,
    )


def run_jump() -> None:
    """Stage 3: the control - the same target in one jump, on the walk's own step budget."""
    start_locks, end_locks = path_locks()
    walk = json.loads((WALK_DIR / "run.json").read_text())
    budget = int(walk["total_finetune_steps"])
    params, _ = ppo.load_checkpoint(START_DIR)
    JUMP_DIR.mkdir(parents=True, exist_ok=True)
    del start_locks

    spent = 0
    rounds = []
    while spent < budget:
        index = len(rounds)
        with locked_compilation(end_locks, dof_path.base_sim_dt(ROBOT)):
            result = ppo.train_policy(
                robot=ROBOT,
                params=END_PARAMS,
                num_timesteps=min(FINETUNE_TIMESTEPS, budget - spent),
                seed=SEED + index,
                init_params=params,
                progress_path=JUMP_DIR / f"progress_round_{index}.jsonl",
                **ENV_KWARGS,
            )
        params = result.params
        spent += result.num_timesteps
        stats = evaluate_policy(
            dof_path.make_dof_env(ROBOT, END_PARAMS, end_locks, **ENV_KWARGS),
            ppo.make_policy(ROBOT, params),
            num_episodes=EVAL_EPISODES,
            seed=SEED,
        )
        rounds.append(
            {
                "round": index,
                "cumulative_steps": spent,
                "stats": dataclasses.asdict(stats),
                "is_viable": is_viable(stats),
                "reward": result.metrics.get(ppo.EVAL_REWARD_KEY),
            }
        )
        (JUMP_DIR / "rounds.json").write_text(
            json.dumps({"budget": budget, "rounds": rounds}, indent=2)
        )
        print(json.dumps(rounds[-1]["stats"]), "viable", rounds[-1]["is_viable"], flush=True)
        jax.clear_caches()
        if is_viable(stats):
            break

    ppo.save_checkpoint(
        ppo.TrainingResult(
            params=params,
            metrics={"cumulative_steps": float(spent)},
            num_timesteps=spent,
            wall_clock_seconds=0.0,
            steps_per_second=0.0,
        ),
        JUMP_DIR,
    )
    report(JUMP_DIR, END_PARAMS, end_locks, params, {"budget": budget, "spent": spent})


STAGES = {"start": run_start, "walk": run_walk, "jump": run_jump}

if __name__ == "__main__":
    STAGES[sys.argv[1]]()
