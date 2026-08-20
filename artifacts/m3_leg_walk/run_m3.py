"""Driver for milestone M3: grow a walking multiped from two legs to four.

Stage 1 trains a biped on the four-leg superset body with legs 2 and 3 shrunk to locked
stubs at their hips. Stage 2 grows leg 2 in by continuation, stage 3 grows leg 3 in from
the policy stage 2 ended on, and both record the stance pattern at every waypoint. Stage 4
is the control: the same two-to-four target in one jump, on the step budget the two walks
spent between them. Stage 5 reads the recorded gaits for a bifurcation.

Run one stage at a time: python run_m3.py {baseline,walk23,walk34,jump,analyse}.
"""

import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

import generalist_robotics  # noqa: F401  caps JAX memory before any backend starts
from generalist_robotics.analysis import gait
from generalist_robotics.continuation import leg_path
from generalist_robotics.continuation.path import CHECKPOINT_PREFIX
from generalist_robotics.evaluation.rollout import evaluate_policy, froude_number, is_viable
from generalist_robotics.morphology.multiped import LegGrowth, MultipedSpec
from generalist_robotics.training import ppo

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "m3_biped_policy"
WALK23_DIR = ROOT / "m3_leg_walk_2_3"
WALK34_DIR = ROOT / "m3_leg_walk_3_4"
JUMP_DIR = ROOT / "m3_one_jump"
REFINE23_DIR = ROOT / "m3_refine_2_3"
REFINE34_DIR = ROOT / "m3_refine_3_4"
ANALYSIS_DIR = ROOT / "m3_analysis"

# The superset body: four legs at the compass points of a hip ring, indexed left, right,
# rear, front, so a prefix of them is a biped and then a tripod. Every point of the path
# lives in this one model, which is what holds the policy's action width fixed at twelve.
SPEC = MultipedSpec(n_legs=4)

BIPED = (LegGrowth(2, 0.0), LegGrowth(3, 0.0))
TRIPOD = (LegGrowth(2, 1.0), LegGrowth(3, 0.0))
QUADRUPED = (LegGrowth(2, 1.0), LegGrowth(3, 1.0))

SEED = 0
EVAL_EPISODES = 8

# Budgets. The baseline is trained in two warm-started halves so a run that is not
# learning can be stopped after the first; the walks get a fifth of the baseline per
# fine-tune round, which is the same ratio M2 used.
BASELINE_TIMESTEPS = 25_000_000
BASELINE_ROUNDS = 2
FINETUNE_TIMESTEPS = 4_000_000
MAX_FINETUNE_ROUNDS = 3

# The stride is capped below continuation.path's own ceiling because the gait is sampled
# once per waypoint: a walk that accelerates to half the path per step arrives in three
# waypoints and cannot say where between them anything happened.
STEP_ALPHA = 0.125
MIN_STEP_ALPHA = 0.05
MAX_STEP_ALPHA = 0.15

# A leg only reaches the floor in the last few percent of its growth: its foot hangs a
# fixed fraction of its own length below the hip, so at 0.9 of full length it is five
# centimetres short of the ground however the knee is held. The whole load transfer
# therefore happens inside one coarse step, which is too little resolution to say whether
# the gait crosses it smoothly. The refine stage re-walks that last interval at
# REFINE_STEP_ALPHA, warm starting from the coarse waypoint that precedes it.
REFINE_STEP_ALPHA = 0.025
REFINE_RECORD_FILENAME = "refine.json"


def report(directory: pathlib.Path, growth: tuple[LegGrowth, ...], params: object, extra: dict):
    """Evaluate a policy on one growth state and write the verdict beside it."""
    env = leg_path.make_multiped_env(SPEC, growth)
    started = time.perf_counter()
    policy = leg_path.make_multiped_policy(SPEC, params)
    stats = evaluate_policy(env, policy, num_episodes=EVAL_EPISODES, seed=SEED)
    signature, trace = leg_path.waypoint_gait(env, policy, seed=SEED)
    document = {
        "spec": dataclasses.asdict(SPEC),
        "growth": [dataclasses.asdict(entry) for entry in growth],
        "total_mass": leg_path.multiped_total_mass(SPEC, growth),
        "stats": dataclasses.asdict(stats),
        "froude": froude_number(stats.mean_forward_speed, stats.nominal_leg_length),
        "is_viable": is_viable(stats),
        "gait": leg_path.signature_record(signature),
        "gait_trace_steps": int(trace.shape[0]),
        "eval_seconds": time.perf_counter() - started,
        **extra,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(document, indent=2))
    np.save(directory / "contacts.npy", trace)
    return document


def run_baseline() -> None:
    """Stage 1: train the biped configuration of the superset body from scratch."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    params = None
    spent = 0
    seconds = 0.0
    for index in range(BASELINE_ROUNDS):
        result = leg_path.train_multiped_policy(
            SPEC,
            BIPED,
            num_timesteps=BASELINE_TIMESTEPS,
            seed=SEED + index,
            init_params=params,
            progress_path=BASELINE_DIR / f"progress_{index}.jsonl",
        )
        params = result.params
        spent += result.num_timesteps
        seconds += result.wall_clock_seconds
        print(
            f"round {index}: {result.num_timesteps} steps, "
            f"{result.wall_clock_seconds:.0f}s, reward {result.metrics.get(ppo.EVAL_REWARD_KEY)}",
            flush=True,
        )
        ppo.save_checkpoint(
            dataclasses.replace(result, num_timesteps=spent, wall_clock_seconds=seconds),
            BASELINE_DIR,
        )
    document = report(
        BASELINE_DIR,
        BIPED,
        params,
        {
            "num_timesteps": spent,
            "wall_clock_seconds": seconds,
            "steps_per_second": spent / seconds,
        },
    )
    print(json.dumps(document, indent=2), flush=True)


def run_walk(directory: pathlib.Path, start_dir: pathlib.Path, start, end) -> None:
    """Grow one leg in, warm starting from the policy in start_dir."""
    params, _ = ppo.load_checkpoint(start_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = leg_path.walk_leg_path(
        spec=SPEC,
        start_growth=start,
        end_growth=end,
        init_policy_params=params,
        step_alpha=STEP_ALPHA,
        min_step_alpha=MIN_STEP_ALPHA,
        finetune_timesteps=FINETUNE_TIMESTEPS,
        max_finetune_rounds=MAX_FINETUNE_ROUNDS,
        num_eval_episodes=EVAL_EPISODES,
        seed=SEED,
        log_path=directory / "run.jsonl",
        checkpoint_dir=directory,
        max_step_alpha=MAX_STEP_ALPHA,
    )
    leg_path.save_leg_run_log(result, SPEC, directory / "run.json")
    ppo.save_checkpoint(
        ppo.TrainingResult(
            params=result.final_policy_params,
            metrics={"total_finetune_steps": float(result.total_finetune_steps)},
            num_timesteps=result.total_finetune_steps,
            wall_clock_seconds=result.wall_clock_seconds,
            steps_per_second=0.0,
        ),
        directory / "final",
    )
    signatures, alphas = leg_path.accepted_gait_path(result)
    accepted = [w for w in result.waypoints if leg_path.leg_waypoint_accepted(w)]
    document = report(
        directory,
        end if result.reached_target else accepted[-1].growth,
        result.final_policy_params,
        {
            "reached_target": result.reached_target,
            "total_finetune_steps": result.total_finetune_steps,
            "wall_clock_seconds": result.wall_clock_seconds,
            "num_waypoints": len(result.waypoints),
            "bifurcation_alphas": (
                gait.detect_bifurcation(signatures, alphas) if len(signatures) > 1 else []
            ),
        },
    )
    print(json.dumps({k: v for k, v in document.items() if k != "stats"}, indent=2), flush=True)


def run_walk23() -> None:
    """Stage 2: grow the rear leg in, taking the biped to a tripod."""
    run_walk(WALK23_DIR, BASELINE_DIR, BIPED, TRIPOD)


def run_walk34() -> None:
    """Stage 3: grow the front leg in, taking the tripod to a quadruped."""
    run_walk(WALK34_DIR, WALK23_DIR / "final", TRIPOD, QUADRUPED)


def last_interior_waypoint(walk: pathlib.Path) -> tuple[int, float]:
    """Index and alpha of the accepted waypoint a finished walk arrived at 1.0 from."""
    document = json.loads((walk / "run.json").read_text())
    interior = [w for w in document["waypoints"] if w["accepted"] and w["alpha"] < 1.0]
    if not interior:
        raise SystemExit(f"{walk} has no accepted waypoint before its target to refine from")
    last = interior[-1]
    return int(last["index"]), float(last["alpha"])


def run_refine(
    directory: pathlib.Path, walk: pathlib.Path, leg: int, grown: tuple[LegGrowth, ...]
) -> None:
    """Re-walk the last interval of one leg's growth finely, to resolve its touchdown.

    Args:
        directory: where the refined walk's artefacts go.
        walk: the finished coarse walk to refine the last interval of.
        leg: the leg being grown in.
        grown: growth of every other leg, held fixed across the refinement.
    """
    index, from_alpha = last_interior_waypoint(walk)
    params, _ = ppo.load_checkpoint(walk / f"{CHECKPOINT_PREFIX}_{index:03d}")
    start = (*grown, LegGrowth(leg, from_alpha))
    end = (*grown, LegGrowth(leg, 1.0))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REFINE_RECORD_FILENAME).write_text(
        json.dumps({"walk": str(walk), "from_alpha": from_alpha, "leg": leg}, indent=2)
    )
    result = leg_path.walk_leg_path(
        spec=SPEC,
        start_growth=start,
        end_growth=end,
        init_policy_params=params,
        step_alpha=REFINE_STEP_ALPHA,
        min_step_alpha=REFINE_STEP_ALPHA,
        finetune_timesteps=FINETUNE_TIMESTEPS,
        max_finetune_rounds=MAX_FINETUNE_ROUNDS,
        num_eval_episodes=EVAL_EPISODES,
        seed=SEED,
        log_path=directory / "run.jsonl",
        checkpoint_dir=directory,
        max_step_alpha=REFINE_STEP_ALPHA,
    )
    leg_path.save_leg_run_log(result, SPEC, directory / "run.json")
    print(
        json.dumps(
            {
                "reached_target": result.reached_target,
                "total_finetune_steps": result.total_finetune_steps,
                "wall_clock_seconds": result.wall_clock_seconds,
                "num_waypoints": len(result.waypoints),
            },
            indent=2,
        ),
        flush=True,
    )


def run_refine23() -> None:
    """Resolve the rear leg's touchdown over the coarse walk's last interval."""
    run_refine(REFINE23_DIR, WALK23_DIR, 2, (LegGrowth(3, 0.0),))


def run_refine34() -> None:
    """Resolve the front leg's touchdown over the coarse walk's last interval."""
    run_refine(REFINE34_DIR, WALK34_DIR, 3, (LegGrowth(2, 1.0),))


def walk_step_cost() -> int:
    """Environment steps the two walks spent between them, which the control must match."""
    total = 0
    for directory in (WALK23_DIR, WALK34_DIR):
        total += int(json.loads((directory / "run.json").read_text())["total_finetune_steps"])
    return total


def run_jump() -> None:
    """Stage 4: the control - the same target in one jump, on the walk's own step budget."""
    params, _ = ppo.load_checkpoint(BASELINE_DIR)
    budget = walk_step_cost()
    JUMP_DIR.mkdir(parents=True, exist_ok=True)
    result = leg_path.train_multiped_policy(
        SPEC,
        QUADRUPED,
        num_timesteps=budget,
        seed=SEED,
        init_params=params,
        progress_path=JUMP_DIR / "progress.jsonl",
    )
    ppo.save_checkpoint(result, JUMP_DIR)
    document = report(
        JUMP_DIR,
        QUADRUPED,
        result.params,
        {
            "num_timesteps": result.num_timesteps,
            "budget_from_walks": budget,
            "wall_clock_seconds": result.wall_clock_seconds,
            "metrics": result.metrics,
        },
    )
    print(json.dumps({k: v for k, v in document.items() if k != "metrics"}, indent=2), flush=True)


def walk_gait_path(directory: pathlib.Path, offset: float) -> tuple[list, list]:
    """Read the accepted waypoints' gaits out of a finished walk's JSON log.

    Args:
        directory: the walk's artefact directory.
        offset: added to each alpha so the two walks lay end to end on one 0-to-2 axis.
    """
    document = json.loads((directory / "run.json").read_text())
    signatures = []
    alphas: list[float] = []
    for record in document["waypoints"]:
        if not record["accepted"] or record["gait"] is None:
            continue
        alpha = float(record["alpha"]) + offset
        if alphas and alpha <= alphas[-1]:
            continue
        signatures.append(gait.GaitSignature(**record["gait"]))
        alphas.append(alpha)
    return signatures, alphas


def gait_path_segments() -> list[tuple[pathlib.Path, float, float]]:
    """The finished walks in path order, each with the offset and span its alphas map to.

    The two coarse walks lay end to end on a 0-to-2 axis, one unit per leg grown. Where a
    refinement of a walk's last interval exists, its own 0-to-1 coordinate is folded into
    that interval, so the combined path is sampled finely exactly where the new leg takes
    load and coarsely where nothing touches the ground.
    """
    segments = []
    for walk, refine, offset in (
        (WALK23_DIR, REFINE23_DIR, 0.0),
        (WALK34_DIR, REFINE34_DIR, 1.0),
    ):
        segments.append((walk, offset, 1.0))
        if (refine / "run.json").exists():
            start = float(json.loads((refine / REFINE_RECORD_FILENAME).read_text())["from_alpha"])
            segments.append((refine, offset + start, 1.0 - start))
    return segments


def endpoint_comparison() -> dict | None:
    """Compare the two policies that end on the same four-legged body, if both exist.

    The walk and the one-jump control finish on identical hardware, so the distance
    between their gaits is a statement about gait space alone: it says whether the two
    routes to a quadruped land in the same gait or in different ones, and a distance that
    clears MIN_BIFURCATION_JUMP while every step of the walk stays far below it says the
    two are separated by a gap the walk never crossed.
    """
    walk = WALK34_DIR / "summary.json"
    jump = JUMP_DIR / "summary.json"
    if not (walk.exists() and jump.exists()):
        return None
    walk_gait = json.loads(walk.read_text())["gait"]
    jump_gait = json.loads(jump.read_text())["gait"]
    if walk_gait is None or jump_gait is None:
        return None
    first = gait.GaitSignature(**walk_gait)
    second = gait.GaitSignature(**jump_gait)
    return {
        "walk_gait": walk_gait,
        "jump_gait": jump_gait,
        "distance": gait.gait_distance(first, second),
        "components": gait.gait_distance_components(first, second),
    }


def run_analyse() -> None:
    """Stage 5: read the recorded gaits of the whole two-to-four path for a qualitative jump."""
    measured: dict[float, object] = {}
    for directory, offset, span in gait_path_segments():
        walk_signatures, walk_alphas = walk_gait_path(directory, 0.0)
        for signature, alpha in zip(walk_signatures, walk_alphas, strict=True):
            measured[round(offset + alpha * span, 6)] = signature
    alphas = sorted(measured)
    signatures = [measured[alpha] for alpha in alphas]
    if len(signatures) < 2:
        raise SystemExit(
            f"only {len(signatures)} waypoints carry a gait, so there is nothing to compare; "
            "the walks did not produce a walking policy at two or more bodies"
        )
    jumps, rates = gait.gait_change_rates(signatures, alphas)
    steps = [
        {
            "from_alpha": alphas[index],
            "to_alpha": alphas[index + 1],
            "distance": float(jumps[index]),
            "rate": float(rates[index]),
            "components": gait.gait_distance_components(signatures[index], signatures[index + 1]),
        }
        for index in range(len(jumps))
    ]
    document = {
        "alphas": alphas,
        "signatures": [dataclasses.asdict(signature) for signature in signatures],
        "steps": steps,
        "median_rate": float(np.median(rates)),
        "largest_step": max(steps, key=lambda step: step["distance"]),
        "bifurcation_alphas": gait.detect_bifurcation(signatures, alphas),
        "min_bifurcation_jump": gait.MIN_BIFURCATION_JUMP,
        "bifurcation_rate_factor": gait.BIFURCATION_RATE_FACTOR,
        "walk_versus_jump": endpoint_comparison(),
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "bifurcation.json").write_text(json.dumps(document, indent=2))
    print(json.dumps(document, indent=2), flush=True)


STAGES = {
    "baseline": run_baseline,
    "walk23": run_walk23,
    "walk34": run_walk34,
    "refine23": run_refine23,
    "refine34": run_refine34,
    "jump": run_jump,
    "analyse": run_analyse,
}


def main() -> None:
    """Run the stage named on the command line."""
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: python run_m3.py {{{','.join(STAGES)}}}")
    STAGES[sys.argv[1]]()


if __name__ == "__main__":
    main()
