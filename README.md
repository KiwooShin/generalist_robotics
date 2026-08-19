# generalist_robotics

**Walking a locomotion policy from one robot into another.** Instead of transferring a policy
across the gap between two differently-shaped robots in one jump, deform the body *continuously* —
and fine-tune only at the points where the policy actually falls over.

![A locomotion policy carried into a body twice its size](media/morphology_continuation.gif)

*Berkeley Humanoid grown to 2× size while walking. Floor squares are 1.00 m; the posts mark the
start hip height (0.515 m) and the target (1.030 m). Sped up; see the full video for timing.*

This is **numerical continuation** (homotopy) applied to policies, and simultaneously a self-paced
curriculum over morphology where "does it still walk" is the pacing signal.

## The result

A policy trained only on the base Berkeley Humanoid (160.6 M steps, reward 26.46, 0.449 m/s) was
walked to a robot **twice its size** — mass ×8, torque ×16 — along the dynamic-similarity manifold:

| | fine-tune cost | outcome |
|---|---|---|
| **On the similarity manifold** (torque ×16) | **6.55 M steps — 4.1% of training from scratch** | reached 2× size; 4 of 5 waypoints needed no fine-tuning at all |
| **Off the manifold** (torque ×8, underpowered) | 118 M steps — 73% of from-scratch | **failed**, stalled at α=0.55 after three backtracks |

Walking speed rose 0.448 → 0.594 m/s where dynamic similarity predicts ×√2 = 0.634, and the
**Froude number stayed ≈0.035–0.040 across the whole sweep** — the invariant that lets a single
viability threshold hold at every body size.

The controlled pair is the point: the same geometric path is nearly free when the actuators scale
with the body, and a wall when they do not. The off-manifold failure is a *survival* collapse
(0.24–0.68) while speed holds at ~0.48 m/s — the underpowered giant walks fine, then falls.

## Videos

| | |
|---|---|
| `media/morphology_continuation.mp4` | 55 s, 1920×1080 @ 60 fps — the hero run, with the fine-tuning beat at α=0.475 |
| `media/manifold_vs_underpowered.mp4` | 42 s, 1920×1080 @ 60 fps — success against failure, side by side |

The mp4s are ~80 MB and ~60 MB and are **not committed**; regenerate them with
`PYTHONPATH=src python -m generalist_robotics.viz.movie`.

## Documents

- **[plan.md](plan.md)** — the paradigm, milestones, and what must be beaten
- **[project.md](project.md)** — how this direction was chosen, and the eight that were refuted
- **[research.md](research.md)** — living survey of generalist robotics (~1,500 lines, citations verified)
- **[supplement_research.md](supplement_research.md)** — method deep dives
- **[coding_rule.md](coding_rule.md)** — repo conventions · **[progress/](progress/)** — daily logs

## Layout

```
src/generalist_robotics/
  morphology/   scale a MuJoCo model in size, mass and torque (similarity-exact)
  envs/         Playground locomotion envs carrying the morph into MJX
  evaluation/   rollout statistics and the Froude-based viability test
  training/     PPO, checkpointing, and warm-started fine-tuning
  continuation/ the predictor-corrector walk through morphology space
  viz/          rendering, HUD, and video encoding
  runtime/      GPU memory guards (read the warning below)
  manipulation/ deferred robosuite work
```

## Setup

```bash
conda create -n genrobo python=3.11 -y && conda activate genrobo
pip install -r requirements.txt
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s src -p "test_*.py"
```

Headless rendering needs `MUJOCO_GL=egl`. Set `GENROBO_SKIP_SLOW_TESTS=1` to skip the long paths.

## GPU memory on unified-memory hardware (read before running anything)

The DGX Spark shares one memory pool between CPU and GPU. JAX preallocates **75% of "GPU"
memory**, so a single process reserves **116.5 GiB of the 121 GiB total**. Two concurrent JAX
processes exhaust the machine and then block in D-state inside the NVIDIA driver, where the kernel
OOM killer cannot reap them — the host wedges and needs a hard reboot. This happened on
2026-08-19.

`import generalist_robotics` applies the guard automatically:

| | virtual | JAX device limit |
|---|---|---|
| JAX default | 116.5 GiB | 91.3 GiB |
| with guard | **22.1 GiB** | **30.4 GiB** |

**Never run two GPU jobs at once.** `runtime.gpu.gpu_lock()` enforces it with an advisory lock.

## Hardware

NVIDIA DGX Spark: GB10, 128 GB unified memory, aarch64, sm_121. Measured here: MJX trains the
Berkeley Humanoid at **~148k env-steps/s**, so a full 150 M-step run takes **~22 minutes**.
