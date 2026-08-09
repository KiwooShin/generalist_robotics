# Generalist Robotics — Research Notes

> Living survey of cross-embodiment / generalist robot policy research.
> Core question: **can a policy pretrained on many robots adapt to a new robot far faster
> than training from scratch — and how is that best achieved?**
>
> Entry template: TL;DR / Key idea / Architecture & data / Results / Relevance to fast
> cross-embodiment adaptation / Links.
>
> **Verification legend:** ✅ = links and claims independently re-checked · ⏳ = summarized
> from one pass, pending verification. Anything post-2026-02 is treated as high-risk until
> verified.

## 0. Field map (read this first)

*(to be written after all sections land: one page connecting the lineages, the shared bets,
and the open problems)*

## 1. Physical Intelligence lineage ⏳

Founded early 2024 by Karol Hausman, Sergey Levine, Chelsea Finn, Brian Ichter, and Lachy
Groom. Funding: ~$70M seed → $400M (Nov 2024, $2.4B) → $600M Series B (Nov 2025, $5.6B,
led by CapitalG) → reportedly raising ~$1B at ~$11B (2026). The company site moved from
`physicalintelligence.company` to **pi.website** (old URLs 308-redirect).

**Lineage at a glance:** π0 (Oct 2024) → FAST / π0-FAST (Jan 2025) → openpi open-source
(Feb 2025) → Hi Robot (Feb 2025) → π0.5 (Apr 2025) → Knowledge Insulation (May 2025) →
Real-Time Chunking (Jun 2025) → π*0.6 / Recap (Nov 2025) → Human-to-Robot Transfer
(Dec 2025) → MEM (Mar 2026) → RLT (Mar 2026) → π0.7 (Apr 2026). As of Aug 2026 π0.7 is the
newest model; there is **no "π1"**.

### π0 (Physical Intelligence, Oct 2024)
- **TL;DR**: The first PI generalist VLA — a pre-trained 3B VLM plus a flow-matching "action
  expert" emitting 50 Hz continuous action chunks for dexterous, cross-embodiment control.
- **Key idea**: Follow the LLM playbook for robots — pre-train one policy on a large,
  diverse cross-embodiment mixture, then post-train on high-quality task data for hard
  skills. Rather than discretizing actions, a separate flow-matching head generates
  continuous action chunks, enabling precision that token binning cannot deliver.
- **Architecture & data**: PaliGemma 3B VLM backbone + ~300M action expert (~3.3B total)
  attending to the backbone's KV cache; flow matching produces 50-step chunks at up to
  50 Hz. Trained on VLM web pre-training + Open X-Embodiment + PI's proprietary ~10,000+
  hours across ~8 platforms (UR5e single/bimanual, Franka, Trossen/ALOHA-style bimanual,
  mobile manipulators).
- **Results**: Substantially beat OpenVLA (7B, discretized — near 0 on PI's hard tasks) and
  Octo (93M, diffusion); ~0.97 on easy table bussing, 1.0 on shirt folding. Post-trained π0
  does multi-minute laundry folding from a dryer, table bussing with emergent stacking, and
  cardboard box assembly.
- **Relevance**: Founding artifact of the paradigm — a cross-embodiment base explicitly
  designed for per-task/per-platform fine-tuning; PI's results show pre-train + fine-tune
  beats from-scratch on every task.
- **Links**: [blog](https://www.pi.website/blog/pi0) · [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)

### FAST action tokenizer & π0-FAST (Physical Intelligence, Jan 2025)
- **TL;DR**: A DCT+BPE action tokenizer that makes autoregressive VLAs viable for
  high-frequency dexterous control, training ~5× faster than π0.
- **Key idea**: Naive per-dimension binning fails at high frequency because consecutive
  tokens are highly correlated (weak learning signal). FAST applies a Discrete Cosine
  Transform to each action chunk — concentrating signal in low frequencies, as in JPEG/MP3 —
  then Byte-Pair Encoding, giving ~10× compression (300–500 raw dims → 30–60 tokens) so a
  standard autoregressive VLM can predict actions as tokens.
- **Architecture & data**: Same PaliGemma-based backbone, purely autoregressive (no flow
  head). "FAST+" is a universal tokenizer trained on ~1M real robot action sequences across
  many embodiments, released with a ~3-line API.
- **Results**: Matches π0 on dexterous tasks with ~5× less training compute. Trained on
  DROID it produced the first generalist policy running simple manipulation zero-shot in
  entirely new scenes at UC Berkeley, Stanford, and UW — though autoregressive decoding
  makes inference slower than π0's.
- **Relevance**: FAST+ is an explicitly *embodiment-agnostic action interface* — a drop-in
  tokenizer for a new robot — and the 5× training-cost cut directly lowers adaptation cost.
- **Links**: [blog](https://www.pi.website/research/fast) · [arXiv 2501.09747](https://arxiv.org/abs/2501.09747) · [tokenizer](https://huggingface.co/physical-intelligence/fast)

### openpi — open-sourcing π0 (Physical Intelligence, Feb 2025; major update Sep 2025)
- **TL;DR**: Apache-2.0 repo with weights and fine-tuning code for π0, π0-FAST, and (since
  Sep 2025) π0.5 — JAX reference implementation plus a newer partial PyTorch port.
- **Key idea**: Release the pre-trained cross-embodiment base checkpoints and the full
  fine-tuning stack (LeRobot-format datasets in, policy out) so external labs can adapt PI's
  generalists to their own robots.
- **Checkpoints**: base — `pi0_base`, `pi0_fast_base`, `pi05_base`; fine-tuned —
  `pi0_droid` / `pi0_fast_droid`, `pi05_droid` (knowledge-insulated, fast inference, good
  language following), `pi05_libero` (SOTA on LIBERO), `pi0_aloha_towel` /
  `_tupperware` / `_pen_uncap`. Deployment uses a websocket policy-server/client split.
- **Practical specs**: inference >8 GB VRAM; **LoRA fine-tune >22.5 GB**; **full fine-tune
  >70 GB**. JAX has mixed precision, FSDP, LoRA, all model types; the PyTorch port (Sep 2025)
  is validated on LIBERO but lacks π0-FAST, LoRA, mixed precision, FSDP, EMA, and needs a
  pinned/patched `transformers==4.53.2`. π0.5 is supported only via its flow-matching head
  (the discrete/KI training pipeline is not in the repo). Tested only on Ubuntu 22.04 x86.
- **Relevance**: This is the concrete tooling for "pre-train on many robots, fine-tune on
  yours." PI recommends **1–20 h of demos** for a new task/platform, LoRA for small budgets,
  with config examples for ALOHA, DROID, and LIBERO.
- **Links**: [repo](https://github.com/Physical-Intelligence/openpi) · [blog](https://www.pi.website/blog/openpi)

### Hi Robot (Physical Intelligence, Feb 2025)
- **TL;DR**: Hierarchical "System 2 / System 1" stack — a high-level VLM reasons over
  open-ended prompts and live user feedback, issuing atomic subtask commands to a low-level π0.
- **Key idea**: Flat VLAs follow short commands but choke on compound or interactively
  amended instructions ("make me a veggie sandwich, no pickles… actually that's not trash").
  A separate high-level VLM takes prompt + images, decomposes the job into steps π0
  understands, and grounds real-time user interjections visually before re-planning.
- **Architecture & data**: High-level VLM (PaliGemma family) + π0 as the low-level policy;
  trained largely on *synthetic* data — robot observations paired with hypothetical prompts
  and simulated user corrections.
- **Results**: Across table bussing, sandwich making, grocery shopping — instruction-following
  accuracy 76% vs 36% (flat VLA), and vs a GPT-4o-as-planner baseline (~30% accuracy / 64%
  progress); task progress 81% vs 44% (flat). It beat the much larger GPT-4o orchestrator
  because the high-level VLM is trained in-domain.
- **Relevance**: Orthogonal to embodiment transfer, but it shows the reasoning layer is
  decoupled from the motor layer — you can swap/fine-tune the low-level policy for a new
  robot while reusing the semantic layer. This hierarchy was later folded into π0.5 itself.
- **Links**: [blog](https://www.pi.website/research/hirobot) · [arXiv 2502.19417](https://arxiv.org/abs/2502.19417)

### π0.5 (Physical Intelligence, Apr 2025)
- **TL;DR**: A single VLA co-trained on heterogeneous data (multi-environment robot data,
  cross-embodiment data, web VQA, verbal coaching, subtask labels) that cleans entirely
  unseen homes — PI's open-world generalization milestone.
- **Key idea**: Generalization comes from *data diversity, not just scale*. At inference the
  same model runs hierarchically: it first decodes a high-level semantic subtask in language,
  then generates the 50-step (1 s) low-level action chunk for it — internalizing Hi Robot's
  hierarchy in one network.
- **Architecture & data**: π0-style backbone + action expert, but unified: discrete token
  decoding (language/subtasks, FAST-style) *and* continuous flow matching (actions). Mixture
  includes ~400 h mobile-manipulation data across ~100 real environments plus cross-embodiment
  and web streams.
- **Results**: Multi-minute tasks (put away dishes, make beds, wipe spills, tidy bedrooms) in
  never-seen homes. Ablations: web data drives out-of-distribution object handling, and
  **other-robot data is critical in all conditions**; scaling over number of training
  environments approaches a baseline trained on the test home itself at ~100 environments.
- **Relevance**: The strongest evidence in the lineage that cross-embodiment data materially
  improves a *target* embodiment's generalization. The ablation where removing other robots'
  data hurts everywhere is the empirical core of "pre-train broadly, adapt cheaply."
- **Links**: [blog](https://www.pi.website/blog/pi05) · [arXiv 2504.16054](https://arxiv.org/abs/2504.16054)

### Knowledge Insulation for VLAs (Physical Intelligence, May 2025)
- **TL;DR**: Stop-gradient the flow-matching expert from the VLM backbone while training the
  backbone on FAST discrete tokens + web data — ~7.5× faster training at π0-level inference
  speed, with better generalization.
- **Key idea**: Gradients from a continuous flow action expert *corrupt* the VLM's
  web-pretrained representations, slowing training and degrading language following. Fix:
  (1) block those gradients; (2) still teach the backbone motor representations via
  discretized (π0-FAST) action tokens; (3) co-train on web VL data and robot planning data to
  preserve semantics. The flow expert learns to read the insulated representations for fast
  continuous decoding at inference.
- **Architecture & data**: Same dual-head architecture; the discrete-token branch is a
  training-time scaffold only. This is the training recipe behind π0.5 (+KI) and the released
  `pi05_droid` checkpoint.
- **Results**: ~7.5× training speedup over vanilla π0 training; inference as fast as π0
  (avoiding π0-FAST's slow autoregressive decode); better OOD behavior on unseen
  objects/environments; highest task-completion rates across evaluated benchmarks.
- **Relevance**: Directly cuts the compute cost of pre-training *and* fine-tuning, and
  preserves general knowledge during adaptation — exactly what you want when fine-tuning a
  generalist to a new robot without catastrophic forgetting.
- **Links**: [blog](https://www.pi.website/research/knowledge_insulation) · [arXiv 2505.23705](https://arxiv.org/abs/2505.23705)

### Real-Time Chunking (RTC) (Physical Intelligence, Jun 2025)
- **TL;DR**: A training-free inference algorithm that generates the next action chunk while
  the current one executes — freezing committed actions and "inpainting" the rest — making
  any flow/diffusion VLA robust to inference latency.
- **Key idea**: Chunked policies pause or jerk at chunk boundaries under real latency. RTC
  treats hand-off as inpainting: actions guaranteed to execute are frozen, the remainder is
  re-generated conditioned on them, asynchronously.
- **Architecture & data**: Pure inference-time method for any diffusion/flow VLA (π0, π0.5),
  no retraining. Evaluated on a 12-task dynamic Kinetix benchmark + 6 real bimanual tasks.
- **Results**: Uniquely robust to injected inference delays; improves throughput and enables
  precision tasks like lighting a match under significant latency.
- **Relevance**: Indirect — decouples deployment quality from inference latency, which matters
  when serving a fine-tuned model from a remote or modest GPU (the standard openpi websocket
  deployment mode).
- **Links**: [blog](https://www.pi.website/research/real_time_chunking) · [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)

### π*0.6 & Recap (Physical Intelligence, Nov 2025)
- **TL;DR**: A 5B VLA trained with **Recap** (RL with Experience and Corrections via
  Advantage-conditioned Policies) that learns from its own deployments — demos → teleop
  corrections → autonomous RL — reaching all-day real-world reliability.
- **Key idea**: Imitation-only policies fail because small errors compound outside the demo
  distribution. Recap mimics human skill acquisition in three stages: supervised fine-tuning
  on demonstrations; "coaching" (expert teleop interventions showing recovery when the robot
  errs); and autonomous practice scored by a learned value function. Rather than discarding
  bad data, the policy is *conditioned on the advantage* (change in predicted value), so all
  experience becomes training signal; at inference it is prompted for high-advantage behavior.
- **Architecture & data**: π0.6 base = evolution of π0.5 (5B VLM + action expert) plus a
  learned value function for credit assignment; trained on demos + interventions + thousands
  of autonomous episodes per task.
- **Results**: On the hardest tasks Recap **more than doubles throughput and cuts failure
  rates by 2×+**. π*0.6 made espresso drinks continuously 5:30 am–11:30 pm, folded 50 novel
  laundry items in a new home for hours uninterrupted, and assembled/labeled 59
  chocolate-packaging boxes in a real factory at >90% success.
- **Relevance**: Extends the paradigm past fine-tuning — after adapting a generalist to a new
  robot with demos, the policy keeps improving from its own on-robot experience. The
  adaptation loop becomes demos → corrections → autonomous RL. Cost: task-specific RL
  specialists, which π0.7 later removes.
- **Links**: [blog](https://www.pi.website/blog/pistar06) · [arXiv 2511.14759](https://arxiv.org/abs/2511.14759)

### Emergence of Human-to-Robot Transfer in VLAs (Physical Intelligence, Dec 2025)
- **TL;DR**: Co-fine-tuning π0.5 on egocentric human video (actions = 3D hand poses, treated
  as just another embodiment) roughly doubles generalization — and the benefit *grows with
  pre-training scale*.
- **Key idea**: No special transfer machinery. Human first-person video is ingested as one
  more embodiment in the cross-embodiment mixture; larger/more diverse robot pre-training
  aligns human and robot visual representations, so human data transfers "for free."
- **Architecture & data**: π0.5 base; co-fine-tuning on mixed human egocentric video +
  relevant robot data.
- **Results**: ~2× improvement across four generalization tasks (bussing, spice organization,
  dresser tidying, egg sorting); the gain from human data increases monotonically with the
  scale of the pre-trained model.
- **Relevance**: Shows the cross-embodiment abstraction stretches to *human bodies* — the
  cheapest data source — and that the scale of the pre-trained base is what unlocks cheap
  adaptation data.
- **Links**: [blog](https://www.pi.website/research/human_to_robot)

### MEM — Long and Short-Term Memory for VLAs (Physical Intelligence, Mar 2026)
- **TL;DR**: Multi-scale Embodied Memory adds compressed video short-term memory plus
  language-based long-term memory to π0.5, enabling 15+-minute kitchen tasks and in-context
  adaptation after failures.
- **Key idea**: Long-horizon tasks require remembering progress and out-of-view object
  locations. Short-term: a video encoder (interleaved spatial/temporal attention) compresses
  recent frames into few tokens. Long-term: the model *reasons about what to remember*,
  storing events as natural-language summaries; a chain-of-thought-like loop selects subtasks
  at low frequency while actions run at high frequency.
- **Architecture & data**: Built on π0.5; the MEM video-history encoder later becomes a
  component of π0.7.
- **Results**: Completes grilled-cheese preparation, 15-minute recipe ingredient retrieval,
  and full kitchen cleanup; substantially beats memory-free baselines on six tasks; shows
  in-context strategy changes after failures (regrasping chopsticks differently, trying the
  fridge door the other way) without retraining.
- **Relevance**: Indirect — memory is embodiment-agnostic capability infrastructure — but its
  in-context adaptation hints at test-time adaptation reducing the need for fine-tuning.
- **Links**: [blog](https://www.pi.website/research/memory)

### RLT — Precise Manipulation with Efficient Online RL (Physical Intelligence, Mar 2026)
- **TL;DR**: Freeze the VLA, bottleneck its internals into a compressed "RL token," and train
  tiny actor-critic heads online on-robot — 3× speedups on precision phases from as little as
  15 minutes of robot data.
- **Key idea**: For contact-rich precision (insertion, fastening), full-model RL is too slow.
  RLT trains an encoder-decoder bottleneck producing an RL token from the frozen VLA's
  representations; small actor-critic nets do sample-efficient online RL on that token,
  predicting action-chunk *edits* to the VLA's proposals, with regularization and
  reference-action dropout, plus optional human interventions.
- **Architecture & data**: Frozen π-family VLA + lightweight RL heads; ~15 min of real-world
  data, ~2 h total adaptation.
- **Results**: On screwdriver alignment, zip-tie fastening, ethernet and power-cord insertion:
  up to 3× speedup on the most precise stages. On ethernet insertion the final policy (median
  66 timesteps) beats human teleoperation (146) and the base VLA (228). Complements Recap:
  Recap = large-scale RL for long-horizon reliability, RLT = rapid refinement of hard
  sub-phases.
- **Relevance**: Very direct — a template for *hours-scale* adaptation of a new platform's
  hard skills without touching the generalist's weights: cheap per-robot specialization
  layered on a frozen cross-embodiment base.
- **Links**: [blog](https://www.pi.website/research/rlt)

### π0.7 (Physical Intelligence, Apr 2026) — newest model as of Aug 2026
- **TL;DR**: A 5B *steerable* generalist (4B VLM + MEM video-history encoder + 860M action
  expert) that matches task-specific RL-fine-tuned specialists out of the box, with emergent
  compositional and cross-embodiment generalization.
- **Key idea**: Condition the policy not only on *what* to do but *how*: prompts carry visual
  subgoal images (from a world model), speed/quality metadata, and control-modality labels.
  This "diverse context" disambiguates heterogeneous training data, so even suboptimal
  autonomous trajectories (annotated with quality metadata) become usable signal — effectively
  distilling many π*0.6 RL specialists into one generalist via strategy metadata, ending
  per-task RL fine-tuning.
- **Architecture & data**: 4B VLM backbone + MEM-style video history encoder + 860M action
  expert. A high-level policy generates language subtasks and a world model produces visual
  subgoals. Trained on multi-embodiment robot demos, human video, and autonomous RL episodes.
- **Results**: Single model matches or exceeds π*0.6 RL specialists on laundry folding,
  espresso making, and box building. **Zero-shot cross-embodiment laundry folding on a
  bimanual UR5e with no task data for that robot**, matching first-try expert teleoperators;
  unseen-kitchen appliance operation via skill recombination; vegetable peeling and glass
  cleaning with no fine-tuning. No open-source release announced.
- **Relevance**: The strongest claim yet that the endpoint of "pre-train on many robots" is
  *zero-shot* new-embodiment competence — and that where fine-tuning is still needed, steering
  via prompts/subgoals/metadata can substitute for weight updates.
- **Links**: [blog](https://www.pi.website/blog/pi07) · [arXiv 2604.15483](https://arxiv.org/abs/2604.15483)

**Other 2025–2026 posts (non-model):** "Moravec's Paradox and the Robot Olympics"
(Dec 2025) — essay plus a fine-tuning showcase on very hard manipulation tasks;
"The Physical Intelligence Layer" (Feb 2026) — PI positions its models as a foundation layer
for partner robotics companies, signaling a partner/API business model rather than open
weights for newer models.

### Practical takeaways for fine-tuning PI models on a DGX Spark

1. **What you can actually get weights for**: only π0, π0-FAST, and π0.5 (base +
   DROID/ALOHA/LIBERO variants) via openpi (Apache 2.0). π*0.6 and π0.7 are not released;
   π0.5 in openpi exposes only the flow-matching head. **`pi05_base` is the best open
   starting point** for a new-robot fine-tune; `pi0_base` + LoRA if memory-constrained.
2. **Memory is the Spark's advantage; bandwidth is its weakness.** openpi requires >8 GB
   (inference), >22.5 GB (LoRA), >70 GB (full fine-tune). The Spark's 128 GB unified memory
   fits even **full fine-tuning**, which normally needs an A100-80GB. But GB10's ~273 GB/s
   memory bandwidth is roughly 7× below an A100/H100 — expect full fine-tunes to run
   overnight-to-days for typical 20k–30k-step runs. LoRA is the pragmatic sweet spot.
3. **ARM/CUDA is the main risk.** openpi is tested only on Ubuntu 22.04 x86_64; the Spark is
   aarch64 + CUDA 13 + sm_121. Every wheel (jaxlib, torch, flash-attn) must exist for ARM64
   and target sm_121. Practice from the community: use NVIDIA NGC containers built for
   CUDA 13/sm_121 rather than generic pip wheels; set `TORCH_CUDA_ARCH_LIST` for cc 12.1 when
   compiling extensions; expect some fused kernels (flash-attn variants) to lack aarch64
   builds and fall back to slower paths. openpi's `uv`-pinned dependency set will likely need
   manual re-pinning inside an NGC base image.
4. **PyTorch vs JAX on Spark**: PyTorch is the easier ARM path (NVIDIA ships Spark-ready torch
   containers; an Isaac-GR00T-on-Spark fine-tuning precedent exists), but openpi's PyTorch
   port **has no LoRA** — so PyTorch-on-Spark means a full bf16 fine-tune (fits, but slow).
   JAX has aarch64 CUDA support via NVIDIA's JAX containers, but openpi's pinned jaxlib may
   predate sm_121. Alternative: HuggingFace **LeRobot** hosts community PyTorch ports of
   π0/π0.5 with its own fine-tuning pipeline — easier to bend onto Spark, at some fidelity risk.
5. **Deployment fits the Spark well**: openpi's websocket policy-server design lets the Spark
   serve the fine-tuned policy while a light client on the robot streams observations/actions;
   add RTC-style asynchronous chunking if chunk-boundary latency appears.
6. **Recipe guidance**: 1–20 h of demos is PI's stated range for a new task/platform; convert
   data to LeRobot format; don't over-train narrow (π0.5 ablations argue for preserving the
   pre-trained mixture's benefits). If post-fine-tune reliability matters, the RLT pattern
   (frozen VLA + tiny online-RL heads, ~15 min robot data) is the cheapest published route to
   specialist precision — conceptually reproducible at Spark scale, though its code is unreleased.

## 2. Google DeepMind lineage ⏳

**Lineage at a glance:** RT-1 (Dec 2022) → RoboCat (Jun 2023) → RT-2 (Jul 2023) → RT-X /
Open X-Embodiment (Oct 2023) → RT-Trajectory / SARA-RT / AutoRT (Dec 2023–Jan 2024) →
ALOHA Unleashed (Oct 2024) → Gemini Robotics 1.0 + ER (Mar 2025) → On-Device + SDK
(Jun 2025) → Gemini Robotics 1.5 + ER 1.5 / **Motion Transfer** (Sep 2025) → ER 1.6
(Apr 2026) → **Gemini Robotics 2 family** (Jul 2026). Note RoboCat precedes RT-2.

### RT-1: Robotics Transformer (Google Research, Dec 2022)
- **TL;DR**: First demonstration that one transformer trained on large-scale multi-task real
  robot data yields a generalizing "robotics foundation model" — and can absorb data from a
  *different* robot.
- **Key idea**: Collect a massive multi-task teleop dataset on a fleet of identical robots and
  train one language-conditioned transformer over it. Capacity + data diversity, not
  algorithmic novelty, drives generalization.
- **Architecture & data**: ImageNet-pretrained EfficientNet-B3, language via FiLM, TokenLearner
  token compression, decoder-only Transformer (~35M) emitting discretized actions (11 dims:
  7 arm, 3 base, 1 mode-switch) at 3 Hz. 130k+ episodes, 700+ instructions, from 13 Everyday
  Robots mobile manipulators over 17 months.
- **Results**: 97% on seen instructions (+25% over BC-Z, +32% over Gato); 76% on unseen
  instructions; 83% under distractors, 59% under new backgrounds. **Cross-robot absorption:
  mixing Kuka bin-picking data nearly doubled bin-picking success on Everyday Robots from
  22% → 39% with no loss on original tasks.**
- **Relevance**: The seed of the paradigm — first concrete evidence that heterogeneous robot
  data is an asset, not a contaminant.
- **Links**: [arXiv 2212.06817](https://arxiv.org/abs/2212.06817) · [project](https://robotics-transformer1.github.io/)

### RoboCat (Google DeepMind, Jun 2023)
- **TL;DR**: A self-improving, multi-embodiment goal-conditioned agent that adapts to new tasks
  and **new robot arms** from 100–1,000 demonstrations, then generates its own training data.
- **Key idea**: One decision transformer trained across many arms, action spaces, and control
  modes, with tasks specified by *goal images*. After fine-tuning to a new task/embodiment it
  practices autonomously (~10,000 rollouts/task) and the self-generated data folds back into
  the next generalist round — a flywheel where each generation adapts faster.
- **Architecture & data**: Gato-style autoregressive decision transformer (1.18B) with a frozen
  VQ-GAN image tokenizer; per-embodiment proprioception/action dimensionality handled natively
  in the token sequence. Embodiments: sim Sawyer 7-DoF, sim Panda 7-DoF, real Sawyer 5-DoF,
  real Panda 7-DoF, and — **held out entirely** — a real KUKA 14-DoF arm with a proprietary
  three-finger hand. Millions of trajectories, 253 tasks.
- **Results**: New tasks from as few as 100 demos; onboarded the unseen KUKA three-finger
  embodiment (genuinely different action space) from 1,000 human demos collected in hours.
  Self-improvement raised unseen-task success **36% → 74%** across generations with no
  additional human data per task.
- **Relevance**: The most direct DeepMind precedent for this project's paradigm — it quantified
  the demo budget (100–1,000), showed the budget shrinks as the pre-training mixture grows, and
  onboarded a morphologically different gripper. Goal-image conditioning is the main respect in
  which it was superseded (by language).
- **Links**: [arXiv 2306.11706](https://arxiv.org/abs/2306.11706) · [blog](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)

### RT-2: Vision-Language-Action Models (Google DeepMind, Jul 2023)
- **TL;DR**: Coined the VLA recipe — cast robot actions as text tokens inside a pretrained VLM
  so web-scale knowledge transfers directly into control.
- **Key idea**: Take a large VLM (PaLI-X or PaLM-E), express discretized actions as text token
  strings, and *co-fine-tune* on robot trajectories mixed with the original web-scale
  vision-language data. The robot inherits semantic generalization never present in robot data.
- **Architecture & data**: PaLI-X (55B, also 5B) and PaLM-E (12B), fine-tuned on the RT-1 robot
  dataset co-mixed with internet VQA/captioning; actions as text-token bins, executed closed-loop.
- **Results**: ~6,000 real trials; roughly **2× generalization improvement** over RT-1 on unseen
  conditions (≈62% vs ≈32%). Emergent: "pick the smallest object", "move the banana to 2+1",
  improvised-tool reasoning ("pick the rock as a hammer") via chain-of-thought.
- **Relevance**: Established that most of a policy's "understanding" can come from non-robot
  data — which is *why* later models can afford tiny per-embodiment datasets: the
  embodiment-specific fine-tune only teaches motor grounding, not perception or semantics.
- **Links**: [arXiv 2307.15818](https://arxiv.org/abs/2307.15818) · [project](https://robotics-transformer2.github.io/)

### RT-X / Open X-Embodiment (GDM + 21→34 institutions, Oct 2023)
- **TL;DR**: The field-defining demonstration that pooling data from 22 robots into one policy
  produces **positive transfer** — the shared model beats each lab's own specialist on that
  lab's own robot.
- **Key idea**: Standardize 60 heterogeneous datasets into a common format (RLDS: images +
  language + end-effector actions) and train existing architectures on the union. If one
  "X-robot" policy beats per-robot policies, robotics can consolidate around pretrained
  generalist backbones the way NLP/vision did.
- **Architecture & data**: OXE dataset — 1M+ real trajectories, **22 embodiments** (single arm,
  bimanual, quadruped), 60 datasets, 527 skills / 160,266 tasks. Models: RT-1-X (35M) and
  RT-2-X (55B) trained on the pooled mixture with a coarsely unified end-effector action space,
  **no per-robot heads**.
- **Results**: RT-1-X beat each lab's specialist by **+50% mean success** across six academic
  labs in the small-data regime. RT-2-X showed **3× improvement on emergent-skill evaluations** —
  skills present only in *other* robots' data (e.g. spatial prepositions "on" vs "near")
  transferred to the evaluation platform. Dataset + RT-1-X checkpoint released openly; OXE
  became the substrate for the entire open VLA ecosystem.
- **Relevance**: The foundational empirical result for this project — cross-embodiment data is
  synergistic, and the gains are largest exactly where you have least data. OXE is public and
  directly inheritable.
- **Links**: [arXiv 2310.08864](https://arxiv.org/abs/2310.08864) · [project + dataset](https://robotics-transformer-x.github.io/)

### RT-Trajectory, SARA-RT, AutoRT (Google DeepMind, Dec 2023 – Jan 2024)
- **TL;DR**: Trajectory-sketch conditioning for task generalization, linear-attention
  up-training for faster VLAs, and LLM/VLM-orchestrated fleet data collection.
- **Key idea**: RT-Trajectory replaces language conditioning with coarse 2D trajectory sketches
  overlaid on the image — a *motion-centric, embodiment-agnostic* task representation. SARA-RT
  up-trains quadratic-attention VLAs into linear-attention models to cut inference cost. AutoRT
  uses a VLM to describe scenes and an LLM (governed by a "Robot Constitution") to propose tasks,
  autonomously steering a fleet to collect diverse data.
- **Results**: RT-Trajectory **63% on 41 unseen tasks vs 29% for RT-2**. SARA-RT-2: 10.6% more
  accurate and 14% faster than RT-2. AutoRT: 20 robots simultaneously (52 total) over 7 months,
  77,000 episodes across 6,650 unique tasks.
- **Relevance**: RT-Trajectory matters most here — representing tasks as *motions in image
  space* rather than language decouples skill from embodiment, a direct conceptual ancestor of
  Gemini 1.5's Motion Transfer, and a cheap trick a small project can copy.
- **Links**: [RT-Trajectory 2311.01977](https://arxiv.org/abs/2311.01977) · [SARA-RT 2312.01990](https://arxiv.org/abs/2312.01990) · [AutoRT 2401.12963](https://arxiv.org/abs/2401.12963)

### ALOHA Unleashed (Google DeepMind, Oct 2024)
- **TL;DR**: Scaled teleop data + diffusion policies crack genuinely dexterous bimanual tasks
  (shoelace tying, shirt hanging) on the low-cost ALOHA 2 platform.
- **Key idea**: Pure imitation scales further than expected on contact-rich deformable-object
  tasks given (a) thousands of high-quality demos per task from a cheap bimanual teleop rig and
  (b) an expressive generative policy class (diffusion) instead of deterministic regression.
- **Architecture & data**: Transformer trained with Diffusion Policy on ALOHA 2 (two 6-DoF arms,
  parallel-jaw grippers). 26,000+ demos across 5 real tasks (8,658 for shirt hanging, 5,133 for
  lace tying) plus 2,000+ demos on 3 sim tasks.
- **Results**: 25–95% per-task success with separate per-task models — ~75% shirt hanging
  (including an unseen shirt), ~70% gear insertion at millimeter tolerance, lace tying. Clear
  success-vs-demo-count scaling (shirt: 75% → 30% when cut to 25% of data).
- **Relevance**: Not cross-embodiment itself, but strategically decisive: it made ALOHA 2
  DeepMind's data engine, and that corpus became the *source* embodiment whose skills Gemini
  Robotics 1.5 later transfers to Franka and Apollo. Also calibrates "hard task" demo counts
  (5–10k) against which the 50–200-demo adaptation numbers below should be read.
- **Links**: [arXiv 2410.13126](https://arxiv.org/abs/2410.13126) · [project](https://aloha-unleashed.github.io/)

### Gemini Robotics 1.0 + Gemini Robotics-ER (Google DeepMind, Mar 2025)
- **TL;DR**: First Gemini-2.0-based VLA — cloud backbone + on-robot decoder — doubling prior
  SOTA VLA generalization, learning new tasks from ~100 demos, and adapting to new embodiments
  including a humanoid.
- **Key idea**: Split the model into a large cloud-hosted VLA backbone and a small local action
  decoder, so frontier-model semantics coexist with real-time control. Ship a companion
  pure-reasoning model (ER) exposing embodied primitives — pointing, 3D boxes, grasp/trajectory
  prediction, multi-view correspondence — for zero-shot robot programming.
- **Architecture & data**: Built on Gemini 2.0; backbone query→response <160 ms, end-to-end
  observation→action-chunk ≈250 ms, action chunking yields an **effective 50 Hz control rate**.
  Trained on a large ALOHA 2 teleop corpus plus web-scale multimodal data. Introduced the ERQA
  embodied-reasoning benchmark.
- **Results**: "More than doubles performance on a comprehensive generalization benchmark
  compared to other SOTA VLAs." Specialization to new short-horizon tasks from **as few as 100
  demonstrations**. Adapted from ALOHA 2 to bi-arm Franka and the Apptronik Apollo humanoid.
  ER achieved 2–3× the success of raw Gemini 2.0 on end-to-end perceive-plan-code control.
- **Relevance**: First flagship to make new-embodiment onboarding an explicit product claim with
  a concrete ~100-demo budget and a demonstrated ALOHA→Franka/Apollo pathway.
- **Links**: [arXiv 2503.20020](https://arxiv.org/abs/2503.20020) · [blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)

### Gemini Robotics On-Device + SDK (Google DeepMind, Jun 2025)
- **TL;DR**: A distilled VLA running entirely on the robot that nearly matches the cloud
  flagship — and DeepMind's first *fine-tunable* release, adapting to new tasks and embodiments
  with 50–100 demonstrations.
- **Key idea**: Low-latency offline autonomy plus developer-facing adaptation: expose the model
  through an SDK (with MuJoCo simulation for evaluation) so external teams port it to their own
  hardware on a small demo budget.
- **Results**: On visual/semantic/behavioral generalization suites, On-Device scores 0.52–0.74
  vs the flagship's 0.60–0.75. **Task adaptation from 50–100 demonstrations.** Externally
  adapted from ALOHA to bi-arm Franka FR3 (instruction following, garment folding) and to the
  Apollo humanoid — a substantially different morphology — with the same recipe.
- **Relevance**: The paradigm productized: multi-robot-pretrained checkpoint + documented
  ~50–100-demo fine-tuning path + sim-based eval loop. The closest industrial template for this
  project; the demo budget, SDK workflow, and MuJoCo-in-the-loop evaluation are all imitable.
- **Links**: [blog](https://deepmind.google/discover/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

### Gemini Robotics 1.5 + ER 1.5 (Google DeepMind, Sep 2025) — **Motion Transfer**
- **TL;DR**: A single multi-embodiment VLA checkpoint controlling ALOHA 2, bi-arm Franka, and
  Apollo out of the box, whose "Motion Transfer" recipe makes skills collected on one robot
  execute **zero-shot** on another — paired with an ER model that orchestrates it agentically.
- **Key idea**: Three innovations. (1) **Motion Transfer (MT)** — an architecture and training
  recipe learning a *unified representation of motion and physical-interaction effects* across
  heterogeneous robot data, aligning embodiments so skills become portable rather than siloed.
  (2) **Embodied Thinking** — the VLA interleaves natural-language reasoning traces with actions
  (decomposing "sort clothes" into primitive motion language before acting), buying multi-step
  robustness, implicit success detection, and self-correction. (3) **Agentic split** — ER 1.5
  (VLM orchestrator: planning, tool use, success detection) calls GR 1.5 (VLA action model) as
  a tool.
- **Architecture & data**: Both inherit Gemini's multimodal backbone. GR 1.5 is a Thinking VLA
  emitting reasoning traces then actions. Robot data: thousands of diverse tasks on ALOHA 2,
  bi-arm Franka, and Apollo, plus internet text/image/video. Notably, **>90% of development
  evaluation episodes ran in a visually and physically aligned MuJoCo simulation**, with
  verified sim-to-real rank consistency.
- **Results** — the key cross-embodiment evidence:
  - *One checkpoint, three form factors*: controls ALOHA, bi-arm Franka, and Apollo **without
    robot-specific post-training**; near-80% on ALOHA generalization suites; beats GR 1.0 and
    On-Device across instruction/action/visual/task generalization on all platforms.
  - *Zero-shot skill transfer*: on a dedicated cross-embodiment benchmark each robot is tested
    on tasks whose data exists **only** on another robot ("unhang the tape", "close the pear
    organizer", "slide open the wardrobe door"). ALOHA performs Franka-only tasks and vice
    versa; Apollo performs ALOHA-only skills despite a much wider embodiment gap. **Success
    rates** are reported, not just progress scores.
  - *Ablations*: single-embodiment training performs poorly on the transfer benchmark;
    multi-embodiment data without MT helps; **MT amplifies the gain further**. The benefit
    profile depends on target-robot data volume — for data-rich ALOHA raw cross-embodiment data
    adds little but MT still extracts transfer; for moderate-data Franka both help; for the
    data-scarce humanoid extra embodiment data gives the biggest absolute boost while MT's
    alignment effect is weakest (largest embodiment gap). A nuanced scaling law for transfer.
  - *Thinking*: thought traces lift multi-step progress across all three embodiments
    (~0.26–0.60 off vs ~0.55–0.67 on), largest jump on the humanoid.
  - *ER 1.5*: SOTA on an aggregate of 15 embodied-reasoning benchmarks, ahead of Gemini 2.5
    Flash and GPT-5 (thinking).
- **Relevance**: The strongest industrial validation of the paradigm to date — beyond OXE's
  "pooled data helps" to **mechanistic transfer**, where an under-resourced embodiment inherits
  *completed skills* from data-rich siblings. Directly transferable to this project: the
  benchmark design (test each robot on tasks seen only by another), the ablation structure
  (single-emb vs multi-emb vs multi-emb + alignment), and the sim-heavy evaluation loop.
- **Links**: [arXiv 2510.03342](https://arxiv.org/abs/2510.03342) · [blog](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)

### Gemini Robotics-ER 1.6 (Google DeepMind, Apr 2026)
- **TL;DR**: Incremental but publicly available upgrade of the embodied-reasoning brain — better
  spatial logic, multi-view reasoning, industrial instrument reading, stricter safety
  instruction following.
- **Key idea**: Harden the orchestrator layer as a product: read analog gauges and sight glasses,
  reason across multiple camera views, call tools (Search, custom functions, VLAs). Driven by
  the Boston Dynamics partnership (autonomous industrial inspection with Spot).
- **Relevance**: Indirect but useful — an ER-style orchestrator is inherently cross-embodiment
  (it emits language subgoals and points, not motor commands), so a small project can pair a
  frozen off-the-shelf ER model with a tiny per-robot action policy and fine-tune only the latter.
- **Links**: [blog](https://deepmind.google/blog/gemini-robotics-er-1-6/)

### Gemini Robotics 2 family (Google DeepMind, Jul 2026)
- **TL;DR**: Three models — GR 2 (VLA), GR-ER 2, GR On-Device 2 — bringing whole-body humanoid
  control, multi-fingered dexterity, multi-robot collaboration, and **few-hour / <200-example
  adaptation to new bi-arm embodiments**.
- **Key idea**: Extend the 1.5 agentic stack three ways: (1) **whole-body intelligence** — one
  VLA coordinates locomotion + manipulation on Apptronik Apollo 2 (walk, crouch, bend, grasp in
  one instruction); (2) **multi-robot collaboration** — ER 2 gives heterogeneous robots
  (Apollo 2 + Franka F3 Duo) shared semantic understanding for workflow handoffs; (3) **cheap
  onboarding** — On-Device 2 uses motion-transfer techniques for rapid adaptation to
  community-scale hardware.
- **Architecture & data**: GR 2 drives full humanoids and bi-arm robots with multi-fingered
  hands (SharpaWave, Inspire) and grippers. ER 2 reportedly based on Gemini 3.5 Flash (128k
  context) with video understanding for continuous progress tracking, native tool orchestration,
  and "think while acting" concurrency. On-Device 2 builds on GR 1.5 technology plus Gemma
  on-device models. Embodiments span Apollo 2, Franka Duo, Dexmate, Trossen, and the **low-cost
  open-source SO-101 arm**. Safety: ASIMOV-Agentic benchmark released on HuggingFace (CC-BY-4.0).
- **Results**: Apollo 2 dexterity — unscrew bulb 92%, pick-from-shelf 76.3% (Inspire hands), tie
  trash bag 44%, ziplock 40%; gripper tasks — precise insertion 89.6% (Franka Duo), general
  pick-and-place 74.2%. **Cross-embodiment adaptation (headline for this project)**: On-Device 2
  adapts to new bi-arm embodiments "in a few hours, typically with fewer than 200 examples" —
  on **SO-101, success jumps 6.7% → 53.3%** (vs 0.0% → 6.7% for On-Device 1); on Dexmate,
  24.4% → 75.6% (vs 13.3% → 33.3%).
- **Relevance**: **The single most relevant datapoint in the lineage for this project** — a
  hobbyist-grade arm going from near-zero to >50% success with <200 demos and hours of compute,
  purely because the base model's motion-transfer pretraining improved between generations
  (the identical recipe topped out at 6.7% with On-Device 1). Adaptation efficiency is chiefly
  a property of the *pre-trained prior*, not the fine-tuning procedure.
- **Links**: [blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/) · [ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)

### Key lessons from the DeepMind lineage for a small-scale project

- **Positive transfer is robust and largest where you have least data.** RT-1's Kuka experiment
  (22%→39%), RT-1-X (+50% in small-data labs), GR 1.5's ablations all agree: pooled multi-robot
  data helps most on the under-resourced embodiment — exactly the situation of a "new robot."
- **The demo budget for onboarding a new embodiment has collapsed ~10× per generation and is now
  ≲200.** RoboCat 1,000 (2023) → Gemini Robotics ~100 for new tasks (Mar 2025) → On-Device
  50–100 (Jun 2025) → On-Device 2 <200 for a whole new bi-arm platform (2026). Budget hundreds,
  not tens of thousands, per embodiment; reserve ALOHA-Unleashed-scale collection (5–10k) only
  for a genuinely dexterous flagship skill.
- **Adaptation efficiency lives in the pre-trained prior, not the fine-tuning trick.** The same
  <200-demo recipe gave 6.7% with On-Device 1 and 53.3% with On-Device 2. If few-shot adaptation
  is failing, fix the base model or data mixture, not the fine-tuning method.
- **Alignment beats concatenation: use a motion-centric shared representation.** Naive
  multi-embodiment co-training helps, but an explicit alignment recipe (Motion Transfer)
  amplifies transfer into zero-shot *task success*. RT-Trajectory suggests a cheap version:
  represent skills as trajectories/motions in image space. Note MT's limit — alignment weakens
  as the embodiment gap widens.
- **Split the stack: embodiment-agnostic brain + small embodied action model.** The ER↔VLA split
  means perception, planning, success detection, and recovery come from a frozen general model,
  reducing per-embodiment learning to short-horizon visuomotor grounding.
- **Evaluate in aligned simulation and benchmark transfer explicitly.** >90% of GR 1.5's eval
  episodes ran in a rank-consistent MuJoCo replica; its benchmark tests each robot on tasks
  whose data exists only on *another* robot, reporting both progress and success. Both are
  directly reproducible at small scale and turn "does transfer work?" into a measurable claim.
- **Actions-as-tokens + web-scale co-training buys semantic generalization you don't collect.**
  Keep vision-language data in the mixture (or start from a VLM/VLA); robot demos then only
  teach the mapping into your action space.
- **Thinking traces and self-improvement are cheap multipliers.** Interleaved language reasoning
  improves long-horizon success and gives implicit success detection with no extra robot data;
  RoboCat-style autonomous practice (36%→74%) can substitute for human demos once a seed policy
  exists — both fit a compute-poor setting.

## 3. Open cross-embodiment models

*(agent running — Octo, OpenVLA/-OFT, CrossFormer, HPT, RDT-1B, GR00T N1/N1.5, SmolVLA,
LAPA, UniVLA, …)*

## 4. Morphology-aware architectures & adaptation techniques

*(agent running — embodiment tokens, per-robot stems/heads, unified action spaces, LoRA
practice, normalization gotchas, few-shot adaptation numbers across the field)*

## 5. Industry landscape ⏳

*Everyone except Physical Intelligence (§1) and DeepMind (§2). Note where claims are
technically documented vs marketing — the gap is large and informative.*

### Generalist AI — GEN-0 / GEN-1 ("Harmonic Reasoning")
- **What they claim**: Pete Florence's startup (he co-created RT-2 and PaLM-E at DeepMind)
  claims genuine **robot-data scaling laws**: downstream post-training performance is a
  predictable power law in pretraining data and model size. GEN-1 (Apr 2026) is claimed as
  "the first general physical AI model to cross the threshold of commercial viability" — 99%
  success where prior models hit 64%, ~3× faster execution.
- **What's published**: The most substantive non-paper technical disclosure in the field. The
  GEN-0 post (Nov 2025) reads like a compressed tech report: a **10B+ parameter embodied
  foundation model trained on 270,000+ hours of real dexterous manipulation** (growing
  10,000 hrs/week from homes, bakeries, laundromats, warehouses across 1,000s of sites), with
  actual scaling-law curves, validation-loss and reverse-KL analyses, blind A/B real-robot
  evals across 16 task sets, and strict train/post-train data separation. Two headline claims:
  (1) **"Harmonic Reasoning"** — trained on asynchronous continuous-time interleaved streams of
  sensing and acting tokens, letting the model "think and act simultaneously" without a
  System-1/System-2 split; (2) a **phase transition / "ossification"** observation — 1B models
  plateau and cannot absorb the data firehose while ~7B+ models keep internalizing it, claimed
  as the first ossification observation in robotics. GEN-1 adds **500,000+ hours of
  pretraining data, none of it robot data** (low-cost wearables on humans), plus ~1 hour of
  robot fine-tuning data per production task. Not peer-reviewed; curves self-reported;
  architecture withheld. Raised $400M Series B at $2B (Jun 2026).
- **Cross-embodiment angle**: Strong but manipulation-centric — transfer across 6-DoF, 7-DoF
  and 16+DoF semi-humanoid platforms. The bigger bet is **embodiment-free pretraining**: since
  GEN-1's corpus is human wearable data, the robot embodiment only enters at fine-tuning,
  arguably the cleanest "one brain, many bodies" story in industry.
- **Demo style**: The current benchmark for credibility-first demos — **counted consecutive
  cycles** (1,800 consecutive block-packing cycles, 200+ vacuum-servicing cycles at 99%, 86
  consecutive t-shirt folds), explicit **"1× speed, not sped up"** labels, speed comparisons
  vs their own prior model, long uncut single-take runs, and a t-SNE data-browser flythrough
  communicating corpus scale visually. They also admit failures ("not all tasks hit these rates").
- **Links**: [GEN-0](https://generalistai.com/blog/nov-04-2025-GEN-0) · [GEN-1](https://generalistai.com/blog/gen-1)

### Skild AI — Skild Brain ("omni-bodied")
- **What they claim**: One shared brain that is **omni-bodied** — it can wake up in a body it
  has never seen (quadruped, humanoid, tabletop arm, mobile manipulator) and control it
  zero-shot or with minimal post-training. Raised $1.4B at $14B+ led by SoftBank (2026).
- **What's published**: Thin relative to the claims — narrative blogs, no papers or curves. A
  **hierarchical architecture** (low-frequency high-level manipulation/navigation policy →
  high-frequency low-level joint/torque policy) trained on simulation at scale + internet human
  video + targeted real robot data. The omni-bodied post claims training across **~100,000
  procedurally varied robot bodies over ~1,000 simulated years**, with test robots explicitly
  held out. Founders Deepak Pathak and Abhinav Gupta (CMU) have strong paper trails, but
  Skild-branded quantitative evals are essentially absent — the biggest claim-to-evidence gap
  among well-funded players.
- **Cross-embodiment angle**: The most extreme in industry — embodiment *randomization* as the
  core training principle, so the model infers its own morphology online from proprioception.
- **Demo style**: **Adversarial robustness stunts** rather than chores — a quadruped's calf is
  amputated and it re-learns a gait in ~7–8 s; joints locked and it walks on three legs; wheels
  jammed → switches from rolling to walking; stilts beyond training range. Memorable and
  directly visualizes generalization. Lesson: *demonstrate the property you claim (adaptivity)
  via visible perturbation, on camera, in one take.*
- **Links**: [general-purpose brain](https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain) · [omni-bodied](https://www.skild.ai/blogs/omni-bodied)

### Figure AI — Helix → Helix 02
- **What they claim**: Helix (Feb 2025) was the first VLA controlling a full humanoid upper body
  at 200 Hz; **Helix 02 (Jan 2026)** extends to full-body loco-manipulation — "a single neural
  system controlling the full body directly from pixels."
- **What's published**: Detailed blogs, no papers. Helix: dual-system VLA — **S2, a 7B VLM at
  7–9 Hz** for scene/language understanding; **S1, an 80M transformer at 200 Hz** outputting a
  35-DoF upper-body action space; trained on **~500 hours** of teleop with VLM auto-labeled
  language; runs on dual embedded low-power GPUs onboard. Helix 02 moves to a **three-tier
  hierarchy**: S0 (10M whole-body controller at **1 kHz**, trained on 1,000+ hours of human
  motion across 200k+ simulated environments), S1 (200 Hz, full-body joint targets from all
  sensors including palm cameras and 3-gram-sensitive fingertip tactile), S2 (semantic
  reasoning). Evals absent — no success rates, no N.
- **Cross-embodiment angle**: Deliberately none — the vertical-integration counterexample. One
  brain, one body, co-designed with hardware. Their scaling story is fleet data from their own
  robots, not cross-embodiment transfer.
- **Demo style**: Defines the "quiet competence" genre — the Helix 02 flagship is a **four-minute,
  uncut, end-to-end dishwasher unload/reload across a full kitchen: 61 loco-manipulation
  actions, no resets, no interventions** — plus two robots with *identical weights* wordlessly
  handing groceries to each other. Techniques worth copying: same-weights multi-robot
  collaboration (proves generality more than any single skill), conceptual-command tests on
  camera ("pick up the desert item" → toy cactus), and one continuous shot as the hero asset
  instead of a montage.
- **Links**: [Helix](https://www.figure.ai/news/helix) · [Helix 02](https://www.figure.ai/news/helix-02)

### 1X Technologies — NEO, Redwood, World Model Lab
- **What they claim**: NEO is the first consumer home humanoid ($20k / $499-mo, shipping 2026)
  running Redwood, an onboard "home brain," with a **teleop-to-autonomy flywheel** where human
  "Expert Mode" operators fill capability gaps while generating training data.
- **What's published**: Mid-level blog detail. Redwood: a **160M-parameter vision-language
  transformer + diffusion-policy action decoder** fusing language embeddings, ViT tokens, and
  proprioception; outputs arm/hand *and* locomotion/pelvis commands simultaneously (whole-body
  multi-contact behaviors like bracing against a wall to open a heavy door); trained on teleop +
  autonomous episodes from EVE and NEO, **including failure rollouts used to supervise
  prediction heads**; runs fully onboard at **~5 Hz**. Separately, the 1X World Model (Jan 2026)
  is a learned action-conditioned video simulator used for policy evaluation, with a dedicated
  World Model Lab (Jun 2026).
- **Cross-embodiment angle**: Two embodiments (EVE wheeled → NEO humanoid) in one corpus, but
  the story is vertical.
- **Demo style — a cautionary tale**: The NEO launch videos (cinematic home chores) went viral,
  then journalists demonstrated nearly everything was teleoperated — in a WSJ session ~100% of
  the work was Expert Mode — plus privacy blowback over remote operators seeing inside homes.
  **Undisclosed or ambiguous teleop is now the fastest way to torch credibility**; the field's
  press literacy has caught up.
- **Links**: [Redwood](https://www.1x.tech/discover/redwood-ai)

### NVIDIA — Isaac GR00T platform + Cosmos (ecosystem angle)
- **What they claim**: Not one brain but the **picks-and-shovels for everyone's brain** —
  open(ish) GR00T N-series models, Cosmos world foundation models for synthetic data, Isaac
  Sim/Isaac Lab + the Newton physics engine, and three-computer deployment (DGX train /
  Omniverse simulate / Jetson Thor run).
- **What's published**: GR00T N1 paper and Cosmos tech reports (models covered in §3). At GTC
  March 2026: GR00T N1.7 to early access with commercial licensing, **GR00T N2 previewed**
  (built on DreamZero research, claimed >2× success on new tasks in new environments vs leading
  alternatives), **Cosmos 3** announced as "the first world foundation model unifying synthetic
  world generation, vision reasoning and action simulation," Isaac Lab 3.0 on Newton 1.0.
- **Cross-embodiment angle**: Structural — GR00T is explicitly cross-embodiment and openly
  fine-tunable, making NVIDIA the default brain for the long tail of hardware companies. The
  deeper strategy is that *everyone* trains on NVIDIA sim and silicon (adopters include 1X,
  AgiBot, Agility, Boston Dynamics, Figure, FANUC, ABB, KUKA; even Generalist AI and Skild use
  NVIDIA infra).
- **Demo style**: Ecosystem-sizzle keynote montages — effective for platform positioning, not a
  model for a research portfolio. The useful artifact style is their developer content:
  reproducible fine-tuning notebooks and sim-to-real walkthroughs.
- **Links**: [GTC 2026](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world) · [Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)

### Tesla — Optimus
- **What they claim**: One end-to-end neural stack shared with FSD, trained via human
  demonstration, video, simulation, and factory fleet data; Optimus V3 with a 22-DoF/50-actuator
  hand entering production in 2026.
- **What's published**: **Essentially nothing** — no papers, no technical blog, no eval
  protocol. Evidence is X posts, keynote segments, and earnings calls. Musk acknowledged on the
  Q4 2025 call that factory Optimus units are "still very much in the R&D phase," and V3
  production slipped to late summer 2026. The 2024 "We, Robot" bartender robots were later
  confirmed teleoperated.
- **Demo style**: High-production spectacle (kung fu, dancing, running) optimized for virality,
  with autonomy asserted in a caption rather than demonstrated by format. **Anti-pattern for a
  research portfolio.**

### Toyota Research Institute + Boston Dynamics — Large Behavior Models
- **What they claim**: Diffusion-based multitask "Large Behavior Models" as the route to
  general-purpose robots; the Oct 2024 partnership put TRI's LBMs on Atlas.
- **What's published**: **The most scientifically rigorous output in this survey.** TRI's
  arXiv 2507.05331, "A Careful Examination of Large Behavior Models for Multitask Dexterous
  Manipulation" (Jul 2025, ~80 authors), is a real evaluation paper: diffusion-policy LBMs on
  large sim+real corpora with a statistically principled pipeline (blind A/B, confidence
  intervals, thousands of rollouts), with the honest headline that **multitask pretraining
  makes policies more successful and robust and cuts new-task data needs** — plus candid nulls
  where scaling didn't help. The Atlas LBM blog (Aug 2025) gives real specs: **450M-param
  Diffusion Transformer with flow matching; 30 Hz observations (stereo + proprioception +
  language); 48-step (1.6 s) action chunks over a ~50-DoF full-body action space**; VR teleop
  built on the MPC controller with haptics and foot trackers; follow-ups showed 1.5–2× inference
  speedup without degradation.
- **Cross-embodiment angle**: Moderate — the LBM recipe transferred from bimanual station arms
  to Atlas, with sim+real co-training across platforms, but no "any body zero-shot" claim.
  Deliberate scientific conservatism.
- **Demo style**: The "Spot Workshop" Atlas video is the reference for **honest long-horizon
  demo craft** — one uncut sequence of sequential language-prompted subtasks (parts sorting,
  barstool flip, tire manipulation, tablecloth spreading), with mid-run perturbations (engineers
  shoving parts, closing lids) shown on camera, and explicit disclosure when clips are sped up.
  Their blogs discuss failures. This is what research-credible marketing looks like.
- **Links**: [arXiv 2507.05331](https://arxiv.org/abs/2507.05331) · [project](https://toyotaresearchinstitute.github.io/lbm1/) · [BD blog](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)

### Covariant → Amazon — RFM-1 (the cautionary consolidation)
- **What they claim**: RFM-1 (Mar 2024) — an 8B multimodal "robotics foundation model,"
  video-prediction world model + language interface, grounded in warehouse pick data.
- **What happened**: Solid technical blog, no paper; then an Aug 2024 **reverse-acquihire** —
  Amazon hired founders Pieter Abbeel, Peter Chen, Rocky Duan plus ~25% of staff and
  non-exclusively licensed the models. The tech resurfaced in Amazon's Blue Jay multi-arm
  system (Oct 2025). Lesson: at ~750,000 robots, Amazon's data advantage dwarfs any startup's.
- **Demo style**: RFM-1's most effective asset was **action-conditioned video prediction
  side-by-sides** (model imagines the outcome of a pick vs reality) — a world-model demo format
  now widely copied.

### Chinese ecosystem — AgiBot, Unitree, Galaxea, Xiaomi (brief)
Far more *open* than US peers on average.
- **AgiBot (Zhiyuan)**: GO-1 (Mar 2025) introduced the ViLLA (Vision-Language-Latent-Action)
  framework with MoE, trained on **AgiBot World** — an open dataset of 1M+ trajectories from a
  100-robot data factory. GO-2 (Apr 2026) adds "action chain-of-thought" + asynchronous
  dual-system, reporting LIBERO 98.5%, sim-to-real 82.9%. Shipped >5,100 humanoids in 2025.
- **Unitree**: hardware-first, but open-sourced **UnifoLM-WMA-0** (Sep 2025) — a
  world-model-action architecture spanning multiple embodiments, usable as an interactive
  simulator or as a policy-enhancing future-predictor; weights on HuggingFace, fine-tuned on
  Open-X + five open Unitree datasets.
- **Galaxea**: published an actual paper — Galaxea Open-World Dataset + G0 dual-system VLA
  (arXiv 2509.00576) — with a **three-stage curriculum (cross-embodiment pretrain →
  single-embodiment pretrain → task post-train)**, open code, and zero-shot numbers (82.5%
  DROID, 98.9% LIBERO). This curriculum is a directly copyable recipe for this project.
- **Cadence**: 13 new embodied models in June 2026 alone; Xiaomi open-sourced
  Xiaomi-Robotics-1 (Aug 2026), pretrained on 100k+ hours of UMI data + 10k hours
  cross-embodiment.
- **Pattern**: cross-embodiment pretraining is standard practice and open weights + open
  datasets are the competitive weapon (commoditize the brain, sell the hardware).

### Dyna Robotics — DYNA-1 (task-first commercial wedge)
- **What they claim**: "First commercial-ready robot foundation model" — master one economically
  real task at a time (napkin folding for restaurants) at production reliability.
- **What's published**: A decent research blog (no paper); the key disclosed idea is a
  **foundation reward model in the loop** — automatic segmentation, progress estimation, and
  subtask labeling of streaming robot experience enabling autonomous data collection. Headline:
  **24-hour uninterrupted run, 850+ napkins, 99.4% success, zero interventions, ~60% of human
  throughput**; week-by-week improvement (failing at 5 min in week 1 → 24 h in week 6);
  transfer evidence (cup-filling learned with 0.7% of the flagship task's data).
- **Demo style**: Invented the **"24-hour livestream" demo genre** — continuous 1× footage of a
  full day of operation plus model-perspective camera views. An endurance format converts
  "reliability" from a claimed number into a watchable fact.
- **Links**: [DYNA-1](https://www.dyna.co/research/dyna-1)

### Sunday Robotics — Memo + ACT-1
- **What they claim**: Founded by **Tony Zhao and Cheng Chi** (first authors of ALOHA/ACT and
  Diffusion Policy/UMI), Sunday exited stealth Nov 2025 with Memo, a wheeled home robot powered
  by **ACT-1**, a "zero robot data" skill foundation model trained purely on human
  demonstrations from a **$200 Skill Capture Glove** (~2,000 gloves in 500+ homes, ~10M
  household episodes) — sidestepping $20k teleop rigs. Raised $165M Series B at $1.15B.
- **Cross-embodiment angle**: An interesting inversion — human hand → robot gripper *is* the
  embodiment gap, solved in the data pipeline rather than by multi-robot pretraining. Same
  philosophical family as Generalist's wearable-first GEN-1.
- **Demo style**: Long-horizon "table-to-dishwasher" run with 33 distinct dexterous
  interactions, in-home (not lab), unseen environments. **The glove itself is a demo asset** —
  showing the *data engine*, not just the robot, is now a differentiating move.

### What actually impresses in demo videos

Synthesized across the field — the formats that earn technical credibility rather than views.
**This is the spec for this project's showcase videos.**

- **One uncut, long-horizon take beats any montage.** Figure's 4-minute dishwasher run (61
  actions, no cuts, no resets) and Atlas's "Spot Workshop" sequence are the most-cited demos of
  the year precisely because they are single continuous shots.
- **Explicit speed disclosure is a norm and a differentiator.** Generalist stamps "1×, not sped
  up"; Boston Dynamics labels 1.5–2× clips. Undisclosed speedup reads as hiding something.
- **Counted, consecutive repetitions turn reliability into a visual.** "1,800 consecutive
  cycles," "86 folds in a row," "850 napkins in 24 h with zero interventions." An on-screen
  counter is worth more than a claimed success percentage.
- **Endurance formats (24-hour livestream) are the strongest robustness claim.**
- **On-camera perturbations prove closed-loop control.** Shoving objects mid-task, amputating a
  limb, moving items while the robot works — the cheapest way to show the policy isn't replaying
  a trajectory.
- **Teleop ambiguity is fatal.** State per-clip what is autonomous, what is teleoperated, and
  what the human prompt was.
- **Same weights, multiple robots / multiple sites is the generality proof.** Identical
  checkpoints on two robots (Figure); zero-shot at an unseen customer site (Dyna);
  held-out-embodiment tests (Skild). ← *directly the core demo of this project*
- **Show the data engine, not just the robot.** t-SNE corpus flythroughs, the $200 glove, the
  teleop rig — demos of the *pipeline* signal scalability to technical audiences.
- **Model-perspective and world-model views add scientific texture.** Policy-camera views,
  predicted-vs-real video side-by-sides.
- **Admitting failure buys credibility.** Generalist's "not all tasks hit these rates" and TRI's
  published nulls are repeatedly cited as reasons to trust the rest. A short failure reel costs
  little and disarms skeptics.

### Hiring-signal takeaways

What these companies visibly reward in robotics research engineers.

- **Data-engine engineering is the scarcest skill.** Collection hardware, auto-labeling,
  segmentation/reward models, dataloaders at scale — more than architecture novelty.
- **Rigorous evaluation is a differentiator, not overhead.** TRI built blind A/B, statistically
  powered real-robot evals; Generalist leads with eval methodology. Being the person who can
  *prove* a policy works (CIs, held-out sites, blind protocols) is a visible signal everywhere.
- **VLA / diffusion / flow-matching fluency is table stakes.** The convergent recipe (VLM
  backbone + high-rate action expert, flow/diffusion action heads, action chunking) appears
  everywhere. Hands-on training *and deployment* end-to-end matters more than paper count.
- **Hierarchy and real-time systems knowledge is back.** S0/S1/S2 stacks at 1 kHz/200 Hz/7 Hz,
  onboard 5 Hz inference on embedded GPUs, async dual-systems — people who can budget latency,
  quantize, and split models across embedded compute are prized.
- **Whole-body control + learning crossover.** The 2026 frontier is loco-manipulation under one
  policy; people bridging MPC/legged control with imitation/RL are the rarest hires.
- **Sim-at-scale and world models are a parallel track.** Massive procedural sim, sim-to-real,
  action-conditioned video models are a distinct hireable specialty.
- **Demonstrated demo craft is itself a signal.** Engineers who can produce a credible uncut,
  counted, perturbation-tested demo of their own work are shipping exactly the artifact these
  teams ship. A portfolio with honest failure analysis mirrors the highest-status industrial
  communication style.
- **Track record of open, replicated recipes gets funded.** Sunday ($1.15B on ALOHA/Diffusion
  Policy pedigree) and Galaxea (paper + open weights) show that shipping reproducible methods
  the community adopts is the strongest individual credential in this market.

## 6. Datasets, benchmarks, simulators

*(agent running — OXE, DROID, BridgeData V2, AgiBot World, LIBERO, SimplerEnv, RoboCasa,
ManiSkill3, MuJoCo Menagerie/Playground, LeRobot ecosystem)*

## 7. Synthesis — implications for this project

*(what we adopt, what we test, experimental design)*

## 8. Reading queue

*(papers spotted but not yet summarized)*
