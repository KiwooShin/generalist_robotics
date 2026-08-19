# Generalist Robotics — Project Plan

**Repo:** https://github.com/KiwooShin/generalist_robotics
**Direction locked:** 2026-08-09 — see [project.md](project.md) for how it was chosen.

> The earlier manipulation-focused plan is archived at
> [archive/plan_manipulation_superseded.md](archive/plan_manipulation_superseded.md).

## Goal

Transfer a legged locomotion policy between robots of **different shape and different forces** by
walking it along a *continuous path through morphology space*, fine-tuning only when it fails —
rather than transferring in one jump.

This is **numerical continuation** (homotopy) applied to policies, and simultaneously a self-paced
curriculum over morphology where "does it still walk" is the pacing signal.

Explicitly a learning-and-demo project. The publishable angle is secondary; the artifact,
the understanding, and the demo videos are the point.

**Out of scope this version:** probing/system-identification, sensor-space continuation, language
interfaces. **Deferred:** hand-only manipulation policy with continuation over finger geometry.

## The result worth aiming at

> Find a target morphology where **direct RL training fails but continuation from an ancestor
> succeeds.**

That reframes continuation from "a cheaper route" to "a route to otherwise-unreachable policies",
and needs no cluster.

## Verified stack (hands-on, on the DGX Spark)

| Component | Status |
|---|---|
| JAX + CUDA on aarch64 | ✅ `jax 0.10.2`, backend `gpu`, `CudaDevice(id=0)` |
| MJX throughput | ✅ 4.6M steps/s @ 4,096 envs |
| MuJoCo Warp | ✅ `cuda:0`, GB10, sm_121 |
| MuJoCo Playground | ✅ 19 locomotion envs |

⚠️ PyPI distribution is **`playground`**, imported as `mujoco_playground`.

## Robot ladder (measured from the models)

| Robot | env | actuators | mass (kg) | height (m) |
|---|---|---:|---:|---:|
| Robotis OP3 | `Op3Joystick` | 20 | 3.1 | 0.244 |
| **Berkeley Humanoid** | `BerkeleyHumanoidJoystickFlatTerrain` | **12** | 16.1 | 0.515 |
| Booster T1 | `T1JoystickFlatTerrain` | 23 | 31.6 | 0.665 |
| Unitree G1 | `G1JoystickFlatTerrain` | 29 | 33.3 | 0.785 |
| Unitree H1 | `H1JoystickGaitTracking` | 19 | 51.4 | 0.970 |
| Apptronik Apollo | `ApolloJoystickFlatTerrain` | 32 | 80.9 | 1.080 |

Berkeley Humanoid is the starting robot: fewest actuators, cheapest to train.

## Milestones

### M1 — Self-morphing continuation (fixed topology) ← **current**

Parametric morphing of one robot: independent **size**, **mass** and **torque** scales. Train a
baseline policy, then map where it survives and walk it through morphology space.

Deliverables:
1. `morphology.py` — apply (size, mass, torque) scaling to a MuJoCo model, inertia-consistent.
2. `evaluation.py` — roll out a policy on a morphed model, return survival/velocity/distance.
3. `training.py` — PPO training and fine-tuning on a morphed env, with checkpointing.
4. `continuation.py` — step along a morphology path, test, fine-tune on failure, log the cost.
5. `viability.py` — 2-D sweep (size × torque) producing the **viability map**.
6. Demo: morph timelapse video + viability map figure.

**Headline artifact — the viability map.** Physics predicts a *ridge of cheap transfer* along the
dynamic-similarity manifold (torque ∝ k⁴, time ∝ √k). Confirming that ridge is a result that
predicts a physical law rather than merely reporting a success rate.

Baselines M1 must report: from-scratch RL at the target, one-jump fine-tune, and **domain
randomization spanning the whole interval** (the honest baseline, most likely to win). Report total
env-steps and wall-clock **along the entire path**, not just the final step.

### M2 — Cross-robot continuation
Berkeley → T1 → G1 → H1 → Apollo. Needs DoF changes, handled by annealing joint stiffness (a joint
locked at high stiffness is effectively absent; annealing it down grows a DoF continuously).

### M3 — Topology: growing legs
2 → 3 → 4 legs. Add a limb with near-zero mass and locked joints, anneal mass up and stiffness
down. Biped→quadruped *must* cross a gait bifurcation, making this both the best demo and the most
interesting science.

### M4 — Showcase
Morph timelapse, growing-leg reel, the ladder, animated viability map, failure reel. Norms: label
real-time vs sped-up, prefer uncut takes, state seeds and step counts.

## Conventions

Code style: [coding_rule.md](coding_rule.md). Daily logs: `progress/YYYY-MM-DD.md`. Research:
[research.md](research.md), [supplement_research.md](supplement_research.md). Commit per change.
