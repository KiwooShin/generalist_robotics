# Generalist Robotics — Project Plan

**Repo:** https://github.com/KiwooShin/generalist_robotics
**Started:** 2026-08-09

> **Status: this plan is a recommendation, not a settled decision.** The stack in §3 and the
> contribution in §4 follow from the survey in `research.md`, and the simulator has been verified
> running on the Spark — but the direction is yours to choose after reading the research. Nothing
> beyond stack verification has been built.

## 1. The paradigm

Train **one policy across many robot embodiments**, then adapt it to a **new, unseen robot** with
a fraction of the data and compute needed to train from scratch.

```
                 ┌─────────────────────────────┐
  Panda demos ───┤                             │
  Sawyer demos ──┤   Cross-embodiment          │──► few demos ──► UR5e   (near transfer)
  IIWA demos ────┤   pre-training              │──► few demos ──► SO-101 (far transfer)
  Kinova demos ──┤   (one generalist policy)   │
                 └─────────────────────────────┘
       vs. baseline: train the new arm from scratch (far more data + time)
```

This is the central bet of Physical Intelligence (π0.5, π0.7), DeepMind (Gemini Robotics 1.5's
Motion Transfer), NVIDIA GR00T, Skild, and Generalist AI. The project reproduces the core
phenomenon — **positive transfer across embodiments plus fast adaptation** — at a scale that fits
one DGX Spark, with a contribution the big labs cannot easily make (§4), and headline demo videos.

**The one number that motivates everything** (`research.md` §0): the same <200-demo recipe took
the SO-101 arm to 6.7% success with Gemini On-Device 1 and 53.3% with On-Device 2. *Adaptation
efficiency is a property of the pre-trained prior, not the fine-tuning procedure.*

## 2. Two deliverables

1. **Showcase** — demo videos (README GIFs per milestone + a full video) proving the paradigm:
   one brain driving many bodies, and a new robot learned in hours. Audience: frontier AI labs
   and robotics companies hiring research engineers.
2. **Learning** — `research.md`: a living, deep survey, maintained as new work appears.

## 3. Proposed stack (M0) — pending review

| Decision | Choice |
|---|---|
| Simulator | **robosuite 1.5 + MimicGen + robomimic** (MuJoCo — robot-agnostic tasks, aarch64-native) |
| Scale-up option | ManiSkill3 for GPU-parallel eval — *gated on verifying SAPIEN aarch64 wheels on the Spark* |
| Training arms | Panda, Sawyer, IIWA, Kinova Gen3 (same gripper class) |
| Held-out (near) | UR5e — "interpolation" |
| Held-out (far) | SO-101 via MuJoCo Menagerie — "extrapolation", 5-DoF |
| Tasks | Lift, Stack, PickPlace-Can, NutAssembly-Square, Door, ToolHang (verify per-arm reachability) |
| Data | ~10 source demos/task → MimicGen-regenerate ~1,000 per (task, arm); held-out adaptation sets {5,10,25,50,100,300} |
| Format | **LeRobotDataset v3** (so π0/GR00T/SmolVLA/X-VLA/ACT all train on our data unchanged) |
| Action head | OpenVLA-OFT recipe: parallel decoding, action chunking, continuous actions, L1/flow |
| Policy A (ours) | HPT-style per-embodiment stems + shared trunk, ~50–200M — full training on Spark |
| Policy B (pretrained) | **X-VLA-0.9B** primary · SmolVLA-450M fallback · GR00T N1.5 if we want NVIDIA's supported pipeline |

Rationale for every row is in `research.md` §7.2.

## 4. The distinctive contribution

A pure reproduction is a weak portfolio piece. Simulation makes possible what no real-robot lab
can run: **decomposing the embodiment gap into its visual and kinematic components** by rendering
arm A's appearance while executing arm B's kinematics, in a 2×2 factorial. This answers a question
the field currently hand-waves — *when a policy fails on a new robot, is it because the robot
looks different or moves differently?*

Secondary angles: the delta-EEF vs padded-joint-space interface ablation; a head-to-head of the
four embodiment-conditioning families (none / ID token / per-arm stems / soft prompts) at matched
scale, which nobody has published; and kinematic-graph morphology encoding for manipulation, which
is mature in locomotion RL and nearly absent from manipulation VLAs.

## 5. Roadmap

- **M0 — Survey & design lock.** ✅ `research.md` v1 (§0–§8); stack chosen. Remaining: verify the
  2026-era claims, confirm robosuite multi-arm behavior hands-on, smoke-test MuJoCo on the Spark.
- **M1 — Multi-embodiment task suite.** robosuite tasks running on all 6 arms behind one unified
  obs/action interface; per-arm reachability audit; scripted/teleop source demos.
  *Demo: grid video of all 6 arms doing all tasks.*
- **M2 — Data engine.** MimicGen regeneration per (task, arm), export to LeRobotDataset v3,
  per-embodiment normalization statistics handled correctly from day one.
  *Demo: one source demo fanning out into five arms' trajectories.*
- **M3 — Per-arm baselines from scratch.** Establishes the from-scratch cost curve every later
  claim is measured against.
- **M4 — Cross-embodiment pre-training.** One policy across the 4 training arms with embodiment
  conditioning; show no regression vs per-arm baselines, plus positive transfer.
  *Demo: "one brain, many bodies" — identical weights, 4 arms, simultaneous.*
- **M5 — Headline: fast adaptation to held-out arms.** Success-vs-demos and success-vs-GPU-hours
  curves for pretrained-on-4 vs pretrained-on-1 vs from-scratch, on UR5e and SO-101.
  *Demo: split-screen learning race with live success counters + animated curves.*
- **M6 — Gap decomposition + conditioning ablation.** The distinctive contribution (§4).
  *Demo: 2×2 factorial visualization.*
- **M7 — Scale up with an open VLA.** LoRA/full fine-tune X-VLA (or SmolVLA) on the same suite;
  compare few-shot adaptation against our small model. Language-conditioned demos.
- **M8 — Showcase polish.** README with per-milestone GIFs, full demo video, results write-up,
  optional project page.

Every milestone ends with a pushed demo video/GIF in the README.

## 6. Demo video specification

Formats that earn technical credibility (derived from the field survey, `research.md` §5):
- **One brain, many bodies** — identical weights, N arms, one grid, one take.
- **Learning race** — split screen, pretrained-adapted vs from-scratch, live success counter.
- **Counted repetitions** — on-screen counters beat claimed percentages.
- **Perturbation reel** — objects shoved mid-episode, on camera, uncut.
- **Failure reel** — short and honest; buys more credibility than it costs.
- **Data-engine shot** — MimicGen fanning one demo across five arms.

Non-negotiable norms: label real-time vs sped-up on every clip; prefer uncut takes; state the demo
count and seed behind every number shown.

## 7. Hardware & constraints

**DGX Spark**: GB10, 128 GB unified memory, ARM (aarch64), sm_121, ~273 GB/s bandwidth.
- Memory is the advantage — it fits full fine-tunes that normally need an A100-80GB.
- Bandwidth is the constraint — roughly 7× below A100/H100, so wall-clock scales sharply with
  parameter count. Sub-1B is where research loops stay fast.
- ARM is the risk — prefer NVIDIA NGC containers over generic pip wheels; verify aarch64 builds
  early for anything with compiled CUDA extensions (SAPIEN, flash-attn, jaxlib).
- Pure MuJoCo is the safe path and runs natively.

## 8. Success criteria

1. A cross-embodiment pretrained policy adapts to a held-out arm with **≥5–10× fewer demos** (or
   GPU-hours) than from-scratch at matched success — with the pretrained-on-one-arm control
   included, so the claim is about *multi-embodiment* pretraining specifically.
2. The visual/kinematic gap decomposition produces a clear, defensible finding.
3. README demo videos good enough to lead a job-application portfolio.
4. `research.md` deep enough to answer interview questions on any major paper in the field.

## 9. Conventions

Code style and repo rules: `coding_rule.md`. Daily logs: `progress/YYYY-MM-DD.md`.
Research notes: `research.md`. Commit per change and push.
