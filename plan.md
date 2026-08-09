# Generalist Robotics — Project Plan

**Repo:** https://github.com/KiwooShin/generalist_robotics
**Started:** 2026-08-09

## 1. The paradigm

Train **one policy model across many robot embodiments**, then adapt it to a **new, unseen robot** with a fraction of the data/compute needed to train from scratch.

```
                 ┌─────────────────────────────┐
  Robot A demos ─┤                             │
  Robot B demos ─┤   Cross-embodiment          │──► fine-tune with few demos ──► Robot X
  Robot C demos ─┤   pre-training              │        (new robot, hours)
  Robot D demos ─┤   (one generalist policy)   │
                 └─────────────────────────────┘
       vs. baseline: train Robot X from scratch (much more data + time)
```

This is the central bet of Physical Intelligence (π0/π0.5), DeepMind Gemini Robotics 1.5
(Motion Transfer), NVIDIA GR00T, Skild, and Generalist AI. The project reproduces the
core phenomenon — **positive transfer across embodiments + fast adaptation** — at a scale
that fits one DGX Spark, with headline demo videos.

## 2. Two deliverables

1. **Showcase** — fancy demo videos (README GIFs + full videos per milestone) proving the
   paradigm works: same brain driving different robots, and a new robot learned in hours.
   Target audience: frontier AI labs & robotics companies (research engineer roles).
2. **Learning** — `research.md`: a living, deep survey of generalist-robotics research,
   maintained as papers are read and revisited.

## 3. Research exploration pipeline (sub-agents)

Five parallel research agents, each owning a slice; results synthesized into `research.md`:

| Agent | Scope |
|---|---|
| 1 | Physical Intelligence lineage: π0, FAST, Hi Robot, π0.5, knowledge insulation, π*0.6, openpi repo practicalities |
| 2 | DeepMind lineage: RT-1/2, RT-X/OXE, RoboCat, ALOHA Unleashed, Gemini Robotics 1.0 → On-Device → 1.5 (Motion Transfer) |
| 3 | Open cross-embodiment models: Octo, OpenVLA(-OFT), CrossFormer, HPT, RDT-1B, GR00T N1/N1.5, SmolVLA, LAPA/UniVLA + morphology-aware architectures |
| 4 | Industry landscape: Generalist AI (GEN-0), Skild, Figure Helix, 1X, TRI/BD LBMs, Covariant, AgiBot… + what makes demo videos impressive |
| 5 | Datasets/benchmarks/sim (OXE, DROID, LIBERO, SimplerEnv, RoboCasa, ManiSkill, MuJoCo Menagerie, LeRobot) + new-robot adaptation techniques |

Ongoing: when a new relevant paper appears, add an entry to `research.md` (same template).

## 4. Build roadmap (draft — to be locked after research synthesis)

- **M0 — Survey & design lock.** `research.md` v1; pick simulator, robot set, model family,
  and the exact headline experiment.
- **M1 — Multi-embodiment task suite.** One simulator (likely MuJoCo-based), 4–6 arms from
  MuJoCo Menagerie (e.g. Franka, UR5e, xArm7, Kinova, SO-101, WidowX), shared tabletop
  tasks (reach / push / lift / pick-place / drawer), unified obs+action interface,
  scripted-expert or teleop data generation. Demo: grid video of all robots doing all tasks.
- **M2 — Per-robot baselines (from scratch).** Small transformer policy (ACT/diffusion-style)
  per robot; establish success rates and the data/compute cost curve of "from scratch."
- **M3 — Cross-embodiment pre-training.** One generalist policy across all training robots
  (embodiment conditioning: per-robot stem/head or embodiment token). Show no regression vs
  per-robot baselines + any positive transfer. Demo: "one brain, many bodies" montage.
- **M4 — Headline: fast adaptation to held-out robot.** Fine-tune pretrained model on the
  held-out arm with {10, 25, 50, 100} demos vs from-scratch. Deliver success-vs-demos and
  success-vs-GPU-hours curves. Demo: split-screen learning race + animated curves.
- **M5 — Scale up with an open VLA.** LoRA fine-tune an open foundation model (candidates:
  π0.5/openpi, GR00T N1.5, SmolVLA — pick after research) on the same suite; compare
  few-shot adaptation vs our small model. Language-conditioned demos.
- **M6 — Showcase polish.** README with GIFs per milestone, full demo video, write-up of
  results; optionally a project page.

Each milestone ends with a pushed demo video/GIF in the README (standing preference).

## 5. Demo video concepts (showcase priority)

- **"One brain, many bodies"**: same language command, N robots executing simultaneously in a grid.
- **Learning race**: split screen — pretrained-adapted vs from-scratch on the new robot at
  equal wall-clock/demos, with live success counters.
- **Data-efficiency animation**: success-vs-demos curves drawing themselves; the gap is the story.
- **"New robot in an hour"**: timeline video — demos collected → fine-tune → deployment reel.
- Norms from the field: disclose real-time vs sped-up, prefer uncut takes, include a failure reel.

## 6. Hardware & constraints

- **DGX Spark**: GB10, 128 GB unified memory, ARM CPU, ~1 PFLOP FP4 (modest BF16 throughput,
  ~273 GB/s bandwidth). Implications:
  - From-scratch models: ≤~200M params comfortably.
  - Open VLAs (2–3B, e.g. π0.5 / GR00T / SmolVLA-class): LoRA fine-tune feasible in memory;
    throughput is the constraint — plan runs in hours-to-days, not minutes.
  - Sim data generation: CPU/GPU headless MuJoCo, cheap.
- Simulation-first (no real robot assumed); real-to-sim credibility via SimplerEnv-style evals
  if applicable.

## 7. Success criteria

1. A pretrained cross-embodiment policy adapts to a held-out robot with **≥5–10× fewer demos**
   (or GPU-hours) than from-scratch at matched success rate — clearly visualized.
2. README demo videos good enough to lead a job-application portfolio.
3. `research.md` deep enough to answer interview questions on any major generalist-robotics paper.
