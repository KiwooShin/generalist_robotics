# generalist_robotics

**One policy, many robots.** Pre-train a manipulation policy across several robot arms, then
adapt it to a *new, unseen* arm with a fraction of the demonstrations and compute that training
from scratch would need.

This is the central bet of Physical Intelligence (π0.5, π0.7), Google DeepMind (Gemini Robotics
1.5's Motion Transfer), NVIDIA GR00T, Skild, and Generalist AI. This project reproduces the core
phenomenon — positive transfer across embodiments plus fast adaptation — at a scale that fits a
single DGX Spark, and adds a measurement the large labs cannot easily make.

- **[plan.md](plan.md)** — paradigm, locked stack, roadmap M0–M8, demo specification
- **[research.md](research.md)** — living survey of generalist-robotics research (~1,500 lines)
- **[coding_rule.md](coding_rule.md)** — repo conventions
- **[progress/](progress/)** — daily logs

## Status

**M0 (survey and design lock) — complete.** The simulation stack is verified running on the
target hardware: 6 arms × 6 tasks all construct and step on the DGX Spark's aarch64 GB10.

## The experiment

```
  Panda ──┐
  Sawyer ─┤   cross-embodiment          ┌─► UR5e   (near transfer: same class, 6 DoF)
  IIWA ───┤   pre-training  ────────────┤
  Kinova3 ┘   (one policy)              └─► Jaco   (far transfer)

  measured against: training the new arm from scratch
```

The headline result will be a **success-vs-demonstrations** curve on the held-out arms, with the
control most demos skip: *pretrained on one arm* alongside *pretrained on four*, so the claim is
about multi-embodiment pretraining specifically rather than pretraining in general.

The distinctive contribution is a **decomposition of the embodiment gap**. In simulation we can
render one arm's appearance while executing another's kinematics — a factorial no real-robot lab
can run — which separates *the robot looks different* from *the robot moves different* as causes
of transfer failure.

## Why the same task runs on every arm

robosuite tasks are robot-agnostic, and the operational-space controller exposes a delta
end-effector action space of identical width regardless of joint count. Verified here:

| Arm | Joints (DoF) | Action width |
|---|---|---|
| Panda, Sawyer, IIWA, Kinova3, Jaco | 7 | 7 |
| UR5e | 6 | 7 |

That uniformity is exactly the "delta-EEF is transfer-friendly" convention the Open X-Embodiment
literature relies on — and making the harder joint-space setting work is one of the planned
ablations.

## Setup

```bash
conda create -n genrobo python=3.11 -y
conda activate genrobo
pip install -r requirements.txt
```

Headless rendering needs `MUJOCO_GL=egl` (the `osmesa` backend is unavailable on this machine).

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m unittest discover -s tests -v
```

**MuJoCo must stay pinned to 3.3.7** — robosuite 1.5.2 calls `MjData.qM`, which was removed in
MuJoCo 3.11.

## Hardware

NVIDIA DGX Spark: GB10, 128 GB unified memory, aarch64, sm_121, ~273 GB/s bandwidth. Memory is
generous enough for full fine-tunes that normally require an A100-80GB; bandwidth is the binding
constraint, so sub-1B models are where research iteration stays fast.
