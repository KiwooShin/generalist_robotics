# generalist_robotics

**One policy, many robots.** Pre-train a manipulation policy across several robot arms, then
adapt it to a *new, unseen* arm with a fraction of the demonstrations and compute that training
from scratch would need.

This is the central bet of Physical Intelligence (π0.5, π0.7), Google DeepMind (Gemini Robotics
1.5's Motion Transfer), NVIDIA GR00T, Skild, and Generalist AI. This project reproduces the core
phenomenon — positive transfer across embodiments plus fast adaptation — at a scale that fits a
single DGX Spark, and adds a measurement the large labs cannot easily make.

- **[plan.md](plan.md)** — paradigm, proposed stack, roadmap M0–M8, demo specification
- **[research.md](research.md)** — living survey of generalist-robotics research (~1,500 lines)
- **[supplement_research.md](supplement_research.md)** — method deep dives (how the algorithms work)
- **[project.md](project.md)** — which direction to actually take, and why
- **[coding_rule.md](coding_rule.md)** — repo conventions
- **[progress/](progress/)** — daily logs

## Status

**M0 — survey complete, direction under review.** The research survey is written and the
simulation stack is verified on the target hardware (6 arms × 6 tasks all construct and step on
the DGX Spark's aarch64 GB10). The experiment design below is a recommendation drawn from the
survey; implementation has not started.

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
MUJOCO_GL=egl PYTHONPATH=src python -m unittest discover -s src -p "test_*.py"
```

**MuJoCo must stay pinned to 3.3.7** — robosuite 1.5.2 calls `MjData.qM`, which was removed in
MuJoCo 3.11.

## Reading the research survey

`research.md` renders as a styled page with a section rail and verification badges:

```bash
python tools/build_research_page.py        # regenerate after editing research.md
python tools/serve.py --host 127.0.0.1
# open http://127.0.0.1:8765/research_page.html
```

**To read it from another machine on the tailnet** (e.g. a laptop, while the repo lives on the
Spark), bind to the Tailscale address instead of loopback:

```bash
python tools/serve.py          # binds to the tailnet address by default
# then browse to http://spark-ddbc:8765/research_page.html
```

That keeps the page on the private tailnet rather than exposing it to the LAN or the internet.

Stop it with Ctrl-C, or from another shell `pkill -f "[t]ools/serve.py"` — the bracket keeps the
pattern from matching (and killing) the shell running the command. GitHub also renders
[research.md](research.md) directly, tables and all.


## GPU memory on unified-memory hardware (read before running anything)

The DGX Spark shares one pool of memory between CPU and GPU. JAX preallocates **75% of "GPU"
memory** by default, which on this machine means a single process reserves **116.5 GiB of the
121 GiB total**. Two concurrent JAX processes exhaust the machine, then block in D-state inside
the NVIDIA driver where the kernel OOM killer cannot reap them — so the box wedges and needs a
hard reboot rather than merely losing the job. This happened on 2026-08-19 (three concurrent
processes; `total_vm` of 369 GiB and 146 GiB in the OOM report).

`import generalist_robotics` applies the guard automatically:

| | virtual | JAX device limit |
|---|---|---|
| JAX default | 116.5 GiB | 91.3 GiB |
| with guard | **22.1 GiB** | **30.4 GiB** |

Set the same values in a shell that imports `jax` directly, before the import:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
```

**Never run two GPU jobs at once.** `generalist_robotics.runtime.gpu.gpu_lock()` enforces this
with an advisory file lock; wrap any training or sweep entry point in it.

## Hardware

NVIDIA DGX Spark: GB10, 128 GB unified memory, aarch64, sm_121, ~273 GB/s bandwidth. Memory is
generous enough for full fine-tunes that normally require an A100-80GB; bandwidth is the binding
constraint, so sub-1B models are where research iteration stays fast.
