# Project Direction — finding the unexplored gap

> **Purpose.** `research.md` maps what the field has done. `supplement_research.md` explains how
> the methods work. **This file decides what *we* do** — specifically, which research direction is
> both genuinely unexplored and executable on one DGX Spark.
>
> Status: literature mapping and gap-refutation in progress (started 2026-08-09).

## The question this file answers

> What has the 2026 state of the art *not* done, that a single researcher with six simulated arms
> and one DGX Spark could do well enough to be worth a frontier lab's attention?

## The candidate idea under test

**The "understanding phase" (working title: Handshake).** When a new robot arrives, don't collect
task demonstrations. Instead issue probe commands through a vision-language model — *"move the
end-effector left by 0.1 m"* — observe the realized motion, and build a calibration between the
command vocabulary and that robot's behaviour. After a handful of probes, a shared
language-command policy drives the new body with no per-robot policy training.

Its appeal is that it changes the headline metric. Instead of *success versus demonstrations* —
the axis the entire field already reports, where we would be competing with Octo's ~100 and
Gemini On-Device 2's <200 — it becomes **success versus number of probe actions**. Probes cost
seconds; demonstrations cost hours.

### The known objection, stated up front

For pure end-effector motion the calibration may be close to free, and therefore close to
trivial. Verified on this machine: robosuite's operational-space controller already exposes a
uniform 7-dim delta-EEF action across Panda, Sawyer, IIWA, Kinova3, UR5e and Jaco, despite UR5e
having 6 joints and the others 7. "Move left 0.1 m" *already* means the same thing on all of
them, because the controller and URDF absorbed the difference — so probing would rediscover what
inverse kinematics gives analytically.

The version that earns its keep must therefore target **what is not in the URDF**: reachable
workspace boundaries and singularities, gripper strength and friction, payload effects,
controller latency and gains, backlash, overshoot under fast commands. Whether that reframing is
novel is exactly what the literature search is testing.

## Method

Deliberately adversarial, because "nobody has done X" is the easiest claim in research to get
wrong:

1. **Map** — six agents cover distinct slices of prior art: probe-then-act lineage (RMA, PEARL,
   hardware-conditioned policies, robot self-modeling, motor babbling); in-context and
   training-free VLA adaptation; active exploration and optimal experiment design; 2026 SOTA in
   new-embodiment adaptation; language grounding in robot capability; benchmarks and adjacent
   fields (adaptive control, classical calibration, sim-to-real online system ID).
2. **Synthesize** — propose 6–8 candidate directions, each stated precisely enough that a single
   paper could refute it.
3. **Refute** — one adversarial agent per direction, whose job is to *find the paper that already
   did it*. A direction survives only if a determined search failed to kill it.

Special attention goes to **limitations and future-work sections** of recent papers, since that
is where the field advertises its own open problems.

## Results

*(pending — the run is in progress)*

## Decision

*(pending)*
