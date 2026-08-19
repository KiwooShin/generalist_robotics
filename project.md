# Project Direction — finding the unexplored gap

> **Purpose.** `research.md` maps what the field has done. `supplement_research.md` explains how
> the methods work. **This file decides what *we* do** — which direction is both genuinely
> unexplored and executable on one DGX Spark.
>
> Run completed 2026-08-09: 15 agents, 3 phases, ~1.7M tokens, 1,098 tool calls, 0 failures.

## Headline

**Nothing proposed was virgin territory. All eight candidate directions came back
`PARTLY_ANTICIPATED` — zero survived untouched, zero were outright refuted.** Every one has a
narrower claim that does survive, and in most cases the surviving claim is sharper and more
defensible than the original.

That is the correct outcome, and more valuable than a clean "gap found." The refutation agents
surfaced **eight papers, five of them from 2026, that the literature mapping had missed** —
including one posted 2.5 weeks ago that directly implements a benchmark we were about to propose.
Discovering that now costs an evening. Discovering it after three months of implementation, or
from a reviewer, costs the project.

## Method

Deliberately adversarial, because "nobody has done X" is the easiest claim in research to get
wrong.

1. **Map** (6 agents) — probe-then-act lineage; in-context and training-free VLA adaptation;
   active exploration and optimal experiment design; 2026 SOTA in new-embodiment adaptation;
   language grounding in robot capability; benchmarks and adjacent fields (adaptive control,
   classical calibration, sim-to-real system ID). Each mined limitations and future-work
   sections, where the field advertises its own open problems.
2. **Synthesize** (1 agent, high effort) — propose 6–8 directions, each stated precisely enough
   that a single paper could refute it, with at least two not depending on the candidate idea.
3. **Refute** (8 agents, one per direction) — job is to *find the paper that already did it*,
   searching under different terminology and adjacent fields, minimum six phrasings each. A
   direction survives only if a determined adversary fails.

## The candidate idea, and what happened to it

**The "understanding phase" (working title: Handshake).** Probe a new robot with commands, observe
realized motion, calibrate command→behaviour, then drive it with a shared policy — no per-robot
training.

**Verdict: the skeleton is a well-established recombination.** Probe → infer embodiment latent →
condition a shared policy is RMA + PEARL + Hardware-Conditioned Policies. Two 2026 papers occupy
most of the territory:

- **SPACE** ([arXiv 2606.24049](https://arxiv.org/abs/2606.24049)) already does the *passive/random*
  version for manipulation: a 500-step random probe (M=10 × K=50) and an ordinary-least-squares
  linear fit. This is the understanding phase, implemented, with the naive probe strategy.
- **Active Embodiment Identification with RL for Legged Robots** (Bohlinger & Peters,
  [arXiv 2605.08020](https://arxiv.org/abs/2605.08020), May 2026) already does the *active*
  version — a policy rewarded for "information-seeking movements that excite the robot's
  dynamics." Locomotion only, four morphologies.

**But Bohlinger & Peters explicitly do not close the loop.** Their stated future work is
*"feeding embodiment predictions back into closed-loop control."* They identify; they never show
that the identification helps a task policy. They also report an unsolved tension: their learned
prober "maintains a stable standing pose with small and medium movements," which "hinders the
identification of parameters that define the maximum capabilities of the motors" — the
safety-versus-informativeness tradeoff in active probing is openly open.

**And the VLM-issues-the-probes framing is the weakest part, not the strongest.** No agent found
support for it being necessary; the metric-command reliability question turns out to already have
prior art (below). Treat the VLM as an optional interface, not the contribution.

### The premise's most serious threat

**Embodiment Scaling Laws in Robot Locomotion** ([arXiv 2505.05753](https://arxiv.org/abs/2505.05753),
CoRL 2025) trains on ~1,000 procedurally generated embodiments and finds that **scaling the number
of training embodiments generalizes far better than scaling data per embodiment**, with zero-shot
transfer to Go2 and H1. The field's current answer to "new robot" is: *train on enough bodies that
you never need to probe.* Any probing work must argue why probing beats simply adding morphologies
to pretraining — or better, measure the exchange rate between them.

## The eight directions and their verdicts

| # | Direction | Verdict | What killed the broad version |
|---|---|---|---|
| G1 | **Interface obfuscation** — transfer to an arm whose action convention is unknown | PARTLY | **ActionShift** (engrXiv, 2026-07-23) benchmarks 6 of 7 scramble axes with the exact collapse-and-recover result, posted 2.5 weeks ago |
| G2 | **Probe-efficiency frontier** — success vs probe budget *k* on a held-out arm | PARTLY | SPACE fixes the budget by fiat (500 steps); ASID, ICWM, Poke-and-Strike likewise. **The budget axis itself survives** |
| G3 | **Embodiment scaling laws for manipulation** | PARTLY | GET-Zero ([2407.15002](https://arxiv.org/abs/2407.15002)) already plots held-out success vs number of training embodiments; [2511.01177](https://arxiv.org/abs/2511.01177) publishes a dexterous-manipulation scaling curve |
| G4 | **Forgetting Pareto** — cost to old arms of onboarding a new one | PARTLY | ACE-Brain-0 and SMPL report related numbers; XEWorld ([2608.05799](https://arxiv.org/abs/2608.05799)) shows it for world models |
| G5 | **Identifiable × policy-relevant 2×2** | PARTLY | **Poke and Strike** reports both axes on a KUKA arm |
| G6 | **Contact probes for self-identification** | PARTLY | Five lines of work do this; closest is *Exciting Contact Modes* (Sathyanarayan & Abraham, Yale) — a manipulator presses a surface to identify its own inertia |
| G7 | **OnboardBench** — adaptation cost as the primary metric | PARTLY | Time-to-threshold is canonical (Taylor & Stone 2009); Jaquier et al. (IJRR 2024) asked robotics to adopt it; HIL-SERL, LIBERO, RoboTwin 2.0, RoboCat all report variants |
| G8 | **Is language a calibratable actuator interface?** | PARTLY | **HumanCLAW** ([arXiv 2607.27180](https://arxiv.org/abs/2607.27180), Jul 2026) already builds the commanded→realized gain construct with per-command variance for a frozen VLM |

### Method caveat, disclosed

Two refutation agents (G3, G4) hit an exhausted WebSearch quota and fell back to the arXiv API,
HuggingFace semantic search, and direct full-text reads. Their arXiv/cs.RO coverage is good;
**ICRA/T-RO-only non-arXiv work is under-covered for those two directions**. Re-check G3 and G4
against IEEE Xplore before relying on them.

## Recommendation

Three surviving claims compose into one project that is defensible against everything the
refutation phase found.

### Close the loop that locomotion left open — and put a budget axis on it

**The claim, stated to survive review:**

> Active, information-seeking probing has been demonstrated for legged embodiment identification
> but never closed back into control (Bohlinger & Peters 2026, stated future work). Passive random
> probing has been closed into control for manipulation but at a budget fixed by fiat (SPACE 2026,
> 500 steps). **No published work reports downstream manipulation task success on a
> morphologically distinct held-out arm as a function of the deployment-time probe budget k, nor
> compares actively-selected against random and passive-history probe strategies on that axis.**

**Why this is the right pick.** It inherits its own motivation from two 2026 papers rather than
asserting a gap — you are answering a question the field wrote down. The budget axis is the piece
that survived refutation most cleanly. And it converts the headline metric from *success vs
demonstrations* (where you would compete with Octo's ~100 and Gemini On-Device 2's <200) to
**success vs probe actions**, which nobody owns.

**The regime that makes it non-trivial.** Verified on this machine: robosuite's operational-space
controller already gives a uniform 7-dim delta-EEF action across all six arms despite UR5e having
6 joints. *"Move left 0.1 m" already means the same thing on every arm* — so probing in the clean
regime rediscovers what inverse kinematics gives analytically, and the experiment is vacuous.

The fix is to **scramble the interface**: axis permutation, sign flips, per-axis gain, base-frame
rotation, actuation latency, control modality. ActionShift did this for a *fixed* arm; the
surviving combination is **scrambled interface × held-out morphology**, which nobody has run.

**Day-1 de-risking measurement (~4 hours, do this before anything else).** On all six arms,
command a grid of unit delta-EEF actions across sampled workspace poses, log realized
displacement, and fit per-arm affine maps. Report the R² of a single global map. **If one global
map explains nearly all the variance, the clean regime is confirmed vacuous and the scrambled
regime is mandatory** — that single number decides the project's shape and costs an afternoon.

**What makes it a paper rather than a demo:** the scoped G5 analysis — for the arm's *own*
interface parameters, cross per-parameter identifiability (Fisher information from probes) against
per-parameter policy sensitivity. Some parameters are easy to identify and irrelevant; some matter
and are invisible in free space. That 2×2 is the intellectual content, and it tells you which
probes are worth spending budget on.

**Optional third leg** if time allows: contact probes (G6, scoped). Free-space motion cannot reveal
payload, friction, or stiffness. Adding a compliant fixture and comparing free-space against
contact probes at matched budget is a clean ablation, and the prior art (Yale's contact-mode work)
is on classical inertia identification, not on conditioning a learned policy.

### What to drop

- **The VLM as prober.** Not load-bearing, and HumanCLAW already owns the metric-command
  reliability construct. Keep language as an optional interface, not the claim.
- **OnboardBench as a headline** (G7). Adaptation cost is canonical, not novel. Report the metric;
  don't sell a benchmark.
- **Standalone embodiment scaling for manipulation** (G3) — already published twice. It survives
  only as the *exchange rate* question: how many training morphologies equal one probe budget?
  Interesting, but it needs the procedural arm generator, which is a project of its own.

## Decision

*(yours — the recommendation above is what the evidence supports, not a commitment)*

## Reading queue from this run

Papers that ambushed the proposal and should be read before committing:

- Bohlinger & Peters, *Active Embodiment Identification with RL for Legged Robots*,
  [arXiv 2605.08020](https://arxiv.org/abs/2605.08020) — **read first**
- SPACE, [arXiv 2606.24049](https://arxiv.org/abs/2606.24049)
- ActionShift, [engrXiv 7688](https://doi.org/10.31224/7688) — scrambled-interface benchmark
- HumanCLAW, [arXiv 2607.27180](https://arxiv.org/abs/2607.27180)
- *Embodiment Scaling Laws in Robot Locomotion*, [arXiv 2505.05753](https://arxiv.org/abs/2505.05753)
- Hardware Conditioned Policies, [arXiv 1811.09864](https://arxiv.org/abs/1811.09864) — HCP-I is
  the direct ancestor; our framing is "replace HCP-I's per-robot backprop with an amortized
  encoder fed by a deliberate probe sequence"
- *Poke and Strike* — task-informed exploration, reports identification and policy sensitivity
- DexFormer, [arXiv 2602.08278](https://arxiv.org/abs/2602.08278) — passive morphology inference
  for hands
- XEWorld, [arXiv 2608.05799](https://arxiv.org/abs/2608.05799) — embodiment forgetting

---

# Direction change (2026-08-09) — morphology continuation

> Supersedes the probe-based "Handshake" idea above. That analysis is kept because its baselines
> and refuted claims still apply.

## The idea

Several humanoids, similar in gross shape but not identical — different heights, link lengths,
masses, torque limits, actuator dynamics, joint ranges. Train a locomotion policy by RL on robot A
only. Direct deployment of A's policy on robot B is assumed to fail.

Rather than jumping A → B, build a **continuous path through morphology space**. Take a small step
(make A slightly taller, nudge masses and torques toward B). Test whether the policy still
locomotes. If yes, step again. If not, fine-tune briefly — cheap, because the morphology barely
moved since the last working point. Repeat until the morphology *is* robot B, carrying a
continuously adapted policy with it. **The policy is walked from A to B through morphology space.**

This is **numerical continuation** (homotopy): track a solution as a parameter is deformed, seeding
each solve from the last. It is simultaneously a **self-paced curriculum over morphology**, with
"does it still walk" as the pacing signal. Both framings bring machinery — predictor-corrector
schemes, adaptive step size, arc-length parameterization — and a known failure mode that is
scientifically interesting (bifurcation, below).

## Extension: the sensor suite is also a continuation axis

Not only shape. **Sensor placement and sensor characteristics** are part of the path:

- IMU position and orientation on the body (pelvis → torso → head)
- Joint encoder resolution, noise, bias, drift
- Observation latency and update rate
- Force/torque and contact sensor placement
- Camera pose, field of view, resolution

This axis may be *stronger* than pure morphology, for three reasons.

**It escapes the dynamic-similarity trap.** Geometric scaling has an exact similarity theory
(below), so parts of the morphology path may be trivially transferable. Sensor relocation has no
such theory — moving an IMU changes the observation function with no compensating rescaling, so
the policy genuinely has to adapt.

**It is where policies are most brittle.** A locomotion policy reads gravity projection and angular
rate from a specific body frame. Move that frame and every observation is a different function of
the same state.

**It makes topological changes continuous** — the key technical trick. Adding or removing a sensor
is a discrete change in observation dimension, which a homotopy cannot cross. But it can be made
continuous two ways: anneal the sensor's **noise** from effectively infinite (uninformative, i.e.
absent) down to its real value, or anneal a multiplicative **gate** from 0 to 1. The same trick
applies to degrees of freedom: a joint locked with very high stiffness is effectively absent, and
annealing stiffness down **continuously grows a new DoF**. That converts changes in robot topology
— different joint counts, different sensor suites — into paths a continuation method can follow.

## Why the framing is strong, and where it breaks

**The physics problem to confront first.** "Slightly taller" is underspecified. Scale length by k
and dynamic similarity requires mass ~ k³, gravitational torque ~ k⁴, time ~ √k, with the Froude
number v²/gL matched for gaits to correspond. A robot that grows with unchanged motors becomes
weaker relative to its own weight. So morphology space has structure: paths **along** the
dynamically-similar manifold may transfer almost free (and would make the method look good for an
uninteresting reason), while paths **off** it — different torque density, actuator bandwidth, mass
distribution — are where continuation earns its keep. Which directions are cheap and which are
expensive, and what the cheapest A→B path is, is a geodesic question nobody appears to own.

**Bifurcation is the interesting failure.** Continuation methods fail where the solution branch
folds or splits; the robotics analogue is a morphology change forcing a qualitatively different
gait — exactly the Froude-number-driven walk/trot/gallop transitions in biomechanics. Expect points
where small morphology steps demand large policy changes. Detecting them is measurable (contact
sequence, gait phase discontinuity) and predicting them from morphology parameters would be a real
result.

**The baseline that must be beaten.** Embodiment Scaling Laws (arXiv 2505.05753, CoRL 2025) trains
on ~1,000 procedural bodies and transfers zero-shot to Go2 and H1, finding morphology *count* beats
data per morphology. If plain randomization spanning A and B works, a careful path is wasted. The
headline metric must be **total compute to a working B policy**, against: from-scratch on B, direct
fine-tune A→B in one jump, and broad randomized pretraining.

*(literature verdict, baselines and refuted directions pending — run in progress)*
