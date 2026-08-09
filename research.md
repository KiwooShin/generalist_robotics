# Generalist Robotics — Research Notes

> Living survey of cross-embodiment / generalist robot policy research.
> Core question: **can a policy pretrained on many robots adapt to a new robot far faster
> than training from scratch — and how is that best achieved?**
>
> Entry template: TL;DR / Key idea / Architecture & data / Results / Relevance to fast
> cross-embodiment adaptation / Links.
>
> **Verification status (2026-08-09).** 59 high-risk claims — chiefly anything dated after
> early 2026 — were put through an adversarial fact-check against primary sources:
> **48 confirmed, 9 partly wrong (corrected in place), 2 refuted (removed)**. Corrections are
> marked 🔧 and refutations ❌ where they matter. Two findings changed the project plan:
> ManiSkill3 **cannot be installed on this DGX Spark** (no aarch64 wheels for SAPIEN or mplib),
> and its supposed "official cross-embodiment demo set" **does not exist**. Claims not marked
> ✅ or 🔧 come from a single research pass and should be re-checked before being quoted.

## 0. Field map (read this first)

**The one-paragraph version.** Between 2022 and 2026 robot learning repeated the NLP playbook:
first single-task policies, then one multi-task transformer (RT-1), then actions-as-tokens inside
a web-pretrained VLM (RT-2), then pooled multi-robot data that provably helps every robot in the
pool (RT-X / Open X-Embodiment). By 2025 the architecture had converged industry-wide. By 2026
the frontier is no longer "can one model drive many bodies" — it demonstrably can — but **how
cheaply a new body joins the club**, and that cost has fallen from ~1,000 demonstrations to under
200 in three years.

**The converged architecture.** Nearly every serious 2025–26 system is the same shape: a
pretrained VLM backbone (semantics, language, web knowledge) at low frequency, plus a small
**action expert** — flow-matching or diffusion — emitting **chunks** of continuous actions at
high frequency. π0/π0.5, GR00T N1.x, Gemini Robotics, RDT, SmolVLA, X-VLA, TRI/Boston Dynamics
LBM, Figure's Helix, 1X's Redwood all instantiate it. The disagreements are about (a) how the
action expert is coupled to the backbone (gradient flow — see Knowledge Insulation), (b) discrete
tokens vs continuous chunks (FAST vs flow), and (c) how many tiers the hierarchy has (Figure's
S0/S1/S2 at 1 kHz/200 Hz/7 Hz is the extreme).

**Three competing answers to the embodiment gap.** This is the axis that matters for this project:

1. **Pool and scale** — throw all robots' data in and let capacity sort it out. Evidence: RT-X
   (+50% on small-data labs), π0.5 (removing other robots' data hurts *everywhere*), AgiBot GO-1.
   Simple, works, but leaves transfer implicit.
2. **Align explicitly** — learn a shared representation in which embodiments are commensurable.
   Evidence: Gemini Robotics 1.5's **Motion Transfer** (its ablation shows naive multi-embodiment
   co-training helps, and alignment on top amplifies it into *zero-shot task success*), HPT's
   stem/trunk/head, X-VLA's soft prompts, UniAct/LAPA/UniVLA's latent action codebooks. This is
   where the interesting research is.
3. **Erase the gap in the data or the interface** — make the embodiment irrelevant before the
   policy sees it. Evidence: RDT2 and Sunday's ACT-1 (standardize on a gripper-relative action
   space captured from *humans*), Generalist's GEN-1 (pretraining corpus contains no robot data
   at all), Mirage/Shadow/RoVi-Aug (repaint the robot in the pixels), FAST+ (universal action
   tokenizer). Often the cheapest, but bounded by the interface's expressiveness — e.g.
   gripper-relative schemes only cover parallel-jaw arms.

**The trend line that defines the paradigm.** Demos needed to onboard a new robot: RoboCat 2023 →
1,000; Gemini Robotics Mar 2025 → ~100 for new tasks; On-Device Jun 2025 → 50–100; On-Device 2
Jul 2026 → **<200 for an entirely new bi-arm platform, in a few hours**. The most important
finding attached to that trend is a negative one: the *same* <200-demo recipe produced 6.7%
success with On-Device 1 and 53.3% with On-Device 2 on the same SO-101 arm. **Adaptation
efficiency is a property of the pre-trained prior, not of the fine-tuning procedure.** If
few-shot adaptation underperforms, fix the base model or the data mixture — not the optimizer.

**Where the data is coming from next: humans.** Four independent groups converged in 2025–26 on
egocentric human video/wearables as the cross-embodiment substrate — PI's human-to-robot transfer
(≈2× generalization, and the benefit *grows with pre-training scale*), NVIDIA's ego-video
pretraining for GR00T N1.7, Generalist's GEN-1 (500k+ hours, no robot data), Sunday's $200
capture glove. The embodiment gap is quietly being reframed from a modeling problem into a
data-interface problem.

**What is still unsolved** (and therefore where a small project can say something new):
- **Morphology encoding is a locomotion technology.** URDF/kinematic-graph conditioning is mature
  in legged RL (MetaMorph, URMA, GET-Zero, Body Transformer) and almost absent from manipulation
  VLAs, which mostly use opaque embodiment IDs or per-robot heads. That gap is a legitimate
  research angle.
- **Alignment degrades exactly where you need it.** Gemini 1.5's ablations show Motion Transfer's
  benefit is *weakest* for the humanoid — the largest embodiment gap — which is the opposite of
  what the marketing implies.
- **Evaluation rigor is rare.** TRI's LBM paper (blind A/B, confidence intervals, thousands of
  rollouts, published nulls) is the outlier, not the norm; most releases are self-reported with
  no error bars. Being rigorous is cheap differentiation.
- **The standard benchmarks can't test the paradigm.** LIBERO is single-arm and saturated at
  ~98%. Cross-embodiment claims need a suite where the *same* task runs on several arms with one
  held out — which is why §6.2 matters more than any leaderboard.

## 1. Physical Intelligence lineage ✅

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

## 2. Google DeepMind lineage ✅

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
- **Results** 🔧: New tasks from as few as 100 demos; onboarded the previously unseen KUKA
  14-DoF three-finger embodiment from **1,000 teleoperated demos collected in hours, reaching
  86% success on gear insertion**. Across successive RoboCat generations — each trained on
  progressively broader experience including self-generated data — average success on held-out
  tasks *after fine-tuning on 500 demos* rose **36% → 74%**. (The 36/74 figures are per-generation
  at a fixed 500-demo budget, not a within-run self-improvement curve.)
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

## 3. Open cross-embodiment models ✅

*The models we could actually download and train. Ranked feasibility for DGX Spark at the
end of §7.*

### Octo (UC Berkeley, May 2024)
- **TL;DR**: Small open transformer generalist pre-trained on 800K OXE trajectories,
  explicitly designed for cheap fine-tuning to new robots with new observation/action spaces.
- **Key idea**: Tokenize arbitrary observation streams (multiple cameras, proprioception,
  language/goal-image tasks) into a shared transformer sequence, then attach lightweight
  "readout" tokens whose embeddings feed a diffusion action head. Because inputs and outputs
  attach via small encoders/heads, **a new robot needs only new tokenizers/heads plus trunk
  fine-tuning — no architectural surgery.**
- **Architecture & data**: Octo-Small (27M) and Octo-Base (93M); ViT-style trunk, language via
  T5, diffusion action head predicting chunks; 800K trajectories from 25 OXE datasets.
- **Results**: Beat RT-1-X and matched/beat RT-2-X (55B) zero-shot across institutions;
  fine-tunes to entirely new embodiments (new action spaces, new sensors) with **~100
  demonstrations in under 5 hours on a single consumer GPU**, beating from-scratch and prior
  transfer baselines by ~20%.
- **Relevance**: The canonical small-scale demonstration of this project's paradigm, and its
  fine-tuning protocol is trivially within Spark budget (full fine-tune, not just LoRA).
  Absolute performance is now below 2025–26 VLAs, but it is an excellent fast-iteration baseline.
- **Links**: [arXiv 2405.12213](https://arxiv.org/abs/2405.12213) · [code](https://github.com/rail-berkeley/octo) · [project](https://octo-models.github.io/)

### OpenVLA (Stanford/Berkeley/TRI/GDM, Jun 2024)
- **TL;DR**: The reference open 7B VLA — Llama-2-based, trained on 970K OXE episodes, with a
  well-documented LoRA fine-tuning path for new robots.
- **Key idea**: Fine-tune a Prismatic VLM to emit robot actions as discretized tokens (256 bins
  per dimension) in the LLM vocabulary, inheriting web-scale visual-semantic priors. New-robot
  adaptation is parameter-efficient fine-tuning of the same next-token objective.
- **Architecture & data**: 7B; fused SigLIP + DINOv2 vision encoders, Llama-2-7B backbone;
  autoregressive discrete action tokens; 970K OXE episodes.
- **Results**: Beat RT-2-X (55B) by 16.5% absolute across 29 tasks with 7× fewer params. LoRA
  (~1.4% of params) matches full fine-tuning on new-robot setups in **10–15 h on a single
  A100**; 4-bit inference fits ~7 GB.
- **Relevance**: Proven pretrain→LoRA workflow with the largest community ecosystem (most
  2025–26 papers ablate against it). On Spark: LoRA fits easily in 128 GB, but 7B autoregressive
  training is compute-bound — expect multi-day fine-tunes; 3 Hz native inference is slow for
  control without OFT-style changes.
- **Links**: [arXiv 2406.09246](https://arxiv.org/abs/2406.09246) · [code](https://github.com/openvla/openvla)

### OpenVLA-OFT (Stanford, Feb 2025)
- **TL;DR**: An "Optimized Fine-Tuning" recipe — parallel decoding + action chunking +
  continuous actions + L1 loss — that turns OpenVLA into a near-SOTA, 26× faster policy.
  **The recipe matters more than the model.**
- **Key idea**: Systematically ablates VLA adaptation design choices and finds that replacing
  autoregressive discrete tokens with single-forward-pass parallel decoding of continuous action
  chunks under an L1 regression loss dramatically improves both success and latency. OFT+ adds
  FiLM language conditioning for language-sensitive bimanual tasks.
- **Results**: LIBERO average **76.5% → 97.1%**, 26× action-generation throughput; strong
  bimanual ALOHA results outperforming π0 and RDT-1B fine-tuned on the same data.
- **Relevance**: Current best practice for adapting *any* pretrained VLA to a new robot, and it
  transfers directly to smaller backbones. **Adopt this action-head recipe regardless of which
  backbone this project picks** — it is backbone-agnostic and Spark-friendly.
- **Links**: [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) · [project](https://openvla-oft.github.io/)

### CrossFormer (UC Berkeley, Aug 2024)
- **TL;DR**: One 130M transformer controlling manipulation arms, bimanual systems, wheeled
  robots, quadrupeds, and quadcopters — **without aligning observation/action spaces**.
- **Key idea**: Extends Octo to maximally heterogeneous embodiments: variable-length observation
  token sequences per embodiment plus separate action readout heads per action-space class,
  trained jointly. Shows **no negative transfer** vs specialist policies despite 4 orders of
  magnitude difference in control frequency/DoF across the data.
- **Architecture & data**: ~130M; per-embodiment tokenizers and action heads; 900K trajectories
  spanning 20 embodiments (OXE manipulation, GNM navigation, locomotion, aviation).
- **Results**: Matches embodiment-specific specialists on all evaluated platforms; substantially
  outperforms prior aligned-action-space cross-embodiment methods.
- **Relevance**: Directly validates "single trunk + per-embodiment I/O heads" across radically
  different morphologies — the same structural bet as this project, at a size (130M) where full
  fine-tuning on a Spark is easy. Caveat: no polished few-shot new-embodiment protocol; JAX code.
- **Links**: [arXiv 2408.11812](https://arxiv.org/abs/2408.11812) · [project](https://crossformer-model.github.io/)

### HPT — Heterogeneous Pre-trained Transformers (MIT CSAIL / Meta, Sep 2024)
- **TL;DR**: Embodiment-specific **stems** tokenize each robot's proprioception + vision into a
  fixed number of tokens feeding a shared pre-trained **trunk** — the cleanest architectural
  instantiation of "pre-train the trunk, swap stems for new robots."
- **Key idea**: Every embodiment gets a small stem (cross-attention tokenizers mapping arbitrary
  camera/proprio configurations to ~16 tokens each); a large shared transformer trunk learns
  task- and embodiment-agnostic representations; small per-task heads decode actions. **Adapting
  to a new robot = instantiate a fresh stem + head and train them (optionally with the trunk),
  transferring the trunk weights.**
- **Architecture & data**: Scaled to 1B+ params and 50+ datasets (real teleop, simulation, human
  video); standard transformer trunk; MLP heads outputting normalized continuous actions.
- **Results**: Scaling laws hold across model/data size; the pre-trained trunk improves
  fine-tuned success by **>20%** on unseen tasks vs from-scratch and vs Octo/RT baselines
  (NeurIPS 2024).
- **Relevance**: **Architecturally the closest published match to this project's paradigm.** All
  sizes (3M–1B) released; even HPT-XL full-fine-tunes comfortably on a Spark. Caveats: proprio +
  vision only (no language), research-grade code, absolute task performance trails modern VLAs —
  best treated as the *architectural blueprint to reimplement* with a modern trunk.
- **Links**: [arXiv 2409.20537](https://arxiv.org/abs/2409.20537) · [code](https://github.com/liruiw/HPT) · [project](https://liruiw.github.io/hpt/)

### RDT-1B (Tsinghua, Oct 2024)
- **TL;DR**: 1.2B diffusion transformer for bimanual manipulation with a unified 128-dim
  "physically interpretable" action space covering heterogeneous robots.
- **Key idea**: Handle cross-embodiment heterogeneity by embedding every robot's action into a
  **fixed 128-slot vector where each slot has fixed physical meaning** (joint positions, EEF
  pose, gripper), masking unused slots. A DiT denoises action chunks conditioned on SigLIP
  vision + T5 language. Enables cross-robot pretraining *without* per-dataset normalization.
- **Architecture & data**: 1.2B DiT; pre-trained on 46 datasets / 1M+ trajectories, fine-tuned
  on 6K self-collected ALOHA bimanual episodes.
- **Results**: SOTA bimanual manipulation; zero-shot generalization to unseen objects/scenes;
  **few-shot (1–5 demo) learning of new skills**; ICLR 2025.
- **Relevance**: The unified-slot action space is the reference design for proprioception/action
  padding schemes and a pragmatic alternative to per-robot heads. At 1.2B, full fine-tuning fits
  a Spark easily.
- **Links**: [arXiv 2410.07864](https://arxiv.org/abs/2410.07864) · [code](https://github.com/thu-ml/RoboticsDiffusionTransformer)

### RDT2 (Tsinghua, Sep 2025 / ICML 2026)
- **TL;DR**: First open foundation model claiming **zero-shot deployment on unseen embodiments**
  for open-vocabulary pick/place/wipe tasks, trained by scaling gripper-standardized UMI human data.
- **Key idea**: Sidestep embodiment heterogeneity entirely — collect 10K+ hours of handheld-UMI
  human manipulation in a standardized *gripper-relative EEF* action space that any two-finger
  arm can execute. Two variants: RDT2-VQ (Qwen2.5-VL-7B + Residual-VQ action tokens, strong
  instruction following) and RDT2-FM (flow matching, low latency).
- **Relevance**: Complementary to this project's paradigm — instead of fine-tuning per robot, it
  standardizes the action interface so "new robot" needs only calibration. Limited to
  parallel-jaw gripper embodiments; ~8B means LoRA-only and slow on Spark.
- **Links**: [project](https://rdt-robotics.github.io/rdt2/) · [code](https://github.com/thu-ml/RDT2)

### GR00T N1 → N1.5 → N1.7 (NVIDIA, Mar 2025 → 2026)
- **TL;DR**: The open humanoid/manipulator foundation model line with the most
  production-ready new-embodiment registration pipeline — and NVIDIA explicitly targets DGX
  Spark as the fine-tuning appliance.
- **Key idea**: Dual-system VLA — System 2 (VLM, ~10 Hz) interprets scene and instruction;
  System 1 (flow-matching DiT action expert, ~120 Hz chunks) denoises continuous actions
  conditioned on VLM features and **per-embodiment state/action projectors**. Cross-embodiment
  is handled by embodiment-tagged input/output projectors around a shared core. N1.5 **freezes**
  the upgraded VLM (Eagle 2.5) to preserve web grounding, and adds FLARE future-latent alignment.
- **Architecture & data** 🔧: N1 ~2.2B (Eagle-2 + DiT); N1.5 ~3B (frozen Eagle 2.5 + DiT).
  **N1.6 and N1.7 are distinct releases and must not be conflated.** N1.6 (announced ~Sep 2025,
  documented Jan 2026) is a 3B model on a Cosmos-Reason-2B variant with native-resolution
  support — no EgoScale, no Action Cascade. **N1.7** (Early Access **17 Apr 2026**) is the 3B
  model with the **Cosmos-Reason2-2B** backbone, **"EgoScale" pretraining on 20,854 hours of
  human egocentric video**, and the **"Action Cascade"** dual-system architecture (System 2
  Cosmos-Reason2-2B + System 1 32-layer DiT). Data pyramid: web/human video, DexMimicGen
  synthetic, real humanoid teleop.
- **Results**: N1.5 lifted language following 46.6% → **93.3%** on GR-1 and improved low-data
  post-training efficiency; 0% → 15% zero-shot novel-object handling. Post-training demo counts
  reported at **30 / 100 / 300 demos per task** (45% avg at 100 demos vs 33.4% for Diffusion
  Policy). Integrated into LeRobot v0.4.0 for turnkey post-training. A GR00T N2 built on
  world-model research was previewed in 2026.
- **Relevance**: The `EmbodimentTag` + new-head registration workflow **is** "pre-train across
  robots, rapidly fine-tune to a new one," with mature tooling. Community fine-tunes run on
  single 4090/A6000-class GPUs (~25 GB), so Spark is comfortable. 🔧 On the frequently cited
  Spark datapoint: a **third-party** benchmark (Classmethod/DevelopersIO) reports fine-tuning
  GR00T **N1.6** on a DGX Spark in **5 h 47 m at 90.8/128 GB** — but the write-up specifies
  neither method nor dataset and does *not* claim to follow NVIDIA's official playbook (which
  targets a DGX Station GB300 on LIBERO-Spatial in ~45 min). Treat it as an unattributed
  third-party figure, not a reproducible baseline.
- **Links**: [arXiv 2503.14734](https://arxiv.org/abs/2503.14734) · [code](https://github.com/NVIDIA/Isaac-GR00T) · [N1.5 weights](https://huggingface.co/nvidia/GR00T-N1.5-3B)

### SmolVLA (HuggingFace LeRobot, Jun 2025)
- **TL;DR**: 450M-param community-data VLA that matches much larger models on SO-100/LIBERO-class
  tasks and trains on a single consumer GPU.
- **Key idea**: Compact SmolVLM-2 backbone (with early-layer skipping) + flow-matching action
  expert using interleaved cross/self-attention; pre-trained **purely on crowd-sourced LeRobot
  community datasets** — 🔧 the paper trains on **481** datasets (~22.9K episodes, ~10.6M
  frames); the HuggingFace launch blog's "487" refers to a curated pool, so cite whichever you
  mean. Proof that curated small heterogeneous data + small models is viable. Asynchronous
  inference decouples perception from actuation for 2× throughput.
- **Results**: Matches or outperforms ACT and larger VLAs on LIBERO, Meta-World, and real
  SO-100/SO-101 tasks; **trains on one consumer GPU**, inference possible on CPU.
- **Relevance**: The most Spark-friendly modern VLA — full fine-tuning (not just LoRA) in hours,
  fast iteration loops, native LeRobot tooling for adding a new embodiment. Pretraining
  distribution is hobbyist-arm-skewed, so transfer to dissimilar morphologies leans on the
  fine-tune.
- **Links**: [arXiv 2506.01844](https://arxiv.org/abs/2506.01844) · [blog](https://huggingface.co/blog/smolvla) · [weights](https://huggingface.co/lerobot/smolvla_base)

### X-VLA (Tsinghua AIR et al., Oct 2025 / ICLR 2026)
- **TL;DR**: A 0.9B flow-matching transformer where per-embodiment **soft prompts** (learnable
  embedding sets per data source) handle heterogeneity — SOTA small cross-embodiment VLA.
- **Key idea**: Instead of per-robot stems/heads, keep one clean transformer encoder and give
  each embodiment/data-domain a small set of **learnable prompt tokens** conditioning the whole
  network. **Adaptation to a new robot = initialize a new soft prompt (tiny parameter count) +
  fine-tune** — both cheap and strong.
- **Architecture & data**: 0.9B; standard transformer encoders + flow matching; pre-trained on
  ~290K episodes across 7 platforms (single-arm to bimanual).
- **Results**: LIBERO 98.1%, CALVIN 4.43, SimplerEnv WidowX 95.8%, Google Robot 83.5%; 1st place
  in the AgiBot World Challenge @ IROS 2025. Apache-2.0; 8 per-embodiment fine-tuned checkpoints
  + LoRA adapters released; LeRobot integration.
- **Relevance**: **Arguably the best current fit for this project** — sub-1B (full fine-tune
  fits Spark memory *and* compute), explicit embodiment-conditioning mechanism, documented
  fine-tune recipes, permissive license.
- **Links**: [arXiv 2510.10274](https://arxiv.org/abs/2510.10274) · [code](https://github.com/2toinf/X-VLA)

### LAPA — Latent Action Pretraining from Videos (KAIST/UW/MSR/NVIDIA/AI2, Oct 2024)
- **TL;DR**: Pre-train a VLA on **actionless video** by learning VQ-VAE latent actions between
  frames, then map latents to real actions with a small robot dataset — 30× cheaper pretraining.
- **Key idea**: (1) a VQ-VAE quantizes inter-frame dynamics into discrete latent actions; (2) a
  VLM is pre-trained to predict latent actions from observation + instruction (works on human
  video); (3) small-scale fine-tuning grounds latents into the target robot's action space.
  **The embodiment gap is absorbed by the latent interface.**
- **Results**: +6.2% over the ground-truth-action equivalent trained on the same data across
  cross-task/cross-environment evals; **~30× pretraining compute efficiency**. CoRL 2024
  LangRob best paper.
- **Relevance**: The cheapest route to a cross-embodiment prior when target robots have little
  labeled data; the grounding stage *is* a rapid fine-tune. The released 7B model is heavy for
  Spark — the method applied to a smaller backbone is the interesting part.
- **Links**: [arXiv 2410.11758](https://arxiv.org/abs/2410.11758)

### UniVLA (OpenDriveLab, May 2025)
- **TL;DR**: Learns **task-centric** latent actions from cross-embodiment video (filtering out
  embodiment-specific motion), reaching SOTA at 1/20 of OpenVLA's pretraining compute.
- **Key idea**: A latent action model with DINOv2 features and language conditioning separates
  task-relevant dynamics from embodiment/camera nuisance factors, so one latent policy learns
  from arbitrary embodiments and viewpoints (including human video); per-robot decoding heads
  ground latents into actions.
- **Results**: SOTA on LIBERO, CALVIN, SimplerEnv, R2R navigation and real manipulation; beats
  OpenVLA with **<1/20 pretraining compute (~960 A100-h) and 1/10 downstream data**; RSS 2025.
- **Relevance**: New-embodiment adaptation = train a small action decoder + brief fine-tune,
  with most knowledge in the frozen latent planner — a strong match for Spark budgets.
- **Links**: [arXiv 2505.06111](https://arxiv.org/abs/2505.06111) · [code](https://github.com/OpenDriveLab/UniVLA)

### GO-1 / ViLLA (AgiBot, Mar 2025)
- **TL;DR**: Vision-Language-Latent-Action generalist: VLM + MoE where a latent planner learns
  from cross-embodiment/human data and an action expert learns from 1M+ real trajectories.
- **Key idea**: Latent action tokens bridge image-text and low-level control; the latent planner
  absorbs actionless heterogeneous data while the action expert specializes on AgiBot World teleop.
- **Results**: +32% average success over prior SOTA (46% → 78%); the latent planner ablates as
  the key contributor. Power-law scaling with trajectory count (r = 0.97).
- **Relevance**: Good pretrained prior with full-stack tooling, but data/license are
  non-commercial (CC BY-NC-SA 4.0) and AgiBot-hardware-centric.
- **Links**: [arXiv 2503.06669](https://arxiv.org/abs/2503.06669) · [code](https://github.com/OpenDriveLab/AgiBot-World)

### SpatialVLA (Shanghai AI Lab / IPEC, Jan 2025)
- **TL;DR**: 4B PaliGemma2-based VLA with Ego3D position encoding and **adaptive action grids**
  that re-discretize for each new robot — spatial structure as the transfer vehicle.
- **Key idea**: Inject 3D context (depth-lifted ego coordinates) into visual tokens and
  discretize actions on statistics-adaptive spatial grids; **adapting to a new robot re-fits the
  grids to the target's action distribution** — a principled, very cheap cross-embodiment interface.
- **Results**: Strong zero-shot SimplerEnv/WidowX results; superior spatial-prompt understanding
  across 7 scenarios, 16 real tasks, 48 sim setups; inference in 8.5 GB; RSS 2025.
- **Links**: [arXiv 2501.15830](https://arxiv.org/abs/2501.15830) · [code](https://github.com/SpatialVLA/SpatialVLA)

### RoboVLM (ByteDance et al., Dec 2024)
- **TL;DR**: A systematic design-space study ("what matters in building VLAs") plus a unified
  codebase supporting 8 VLM backbones and arbitrary architecture combinations.
- **Key finding relevant here**: it directly answers *"does cross-embodiment pretraining speed
  up new-robot post-training?"* — **yes, mainly in low-data regimes.** Also finds policy-head
  continuous actions beat interleaved discrete tokens in their setting. Nature Machine
  Intelligence (2025).
- **Relevance**: Use as an evidence base for design choices rather than as a checkpoint.
- **Links**: [arXiv 2412.14058](https://arxiv.org/abs/2412.14058) · [project](https://robovlms.github.io/)

### Other notable open releases (brief)
- **MolmoAct / MolmoAct2 (Ai2, Aug 2025 / 2026)** — fully open (weights + code + data +
  tokenizer) "Action Reasoning Models" reasoning in 3D: depth tokens → editable 2D visual traces
  → actions. MolmoAct2 grafts a flow-matching expert onto the discrete-token VLM and reports
  beating π0.5. Releases include a large open bimanual teleop dataset. Most *transparent*
  full-stack release; 7B means LoRA-only on Spark.
- **WALL-OSS / Wall-OSS-0.5 (X Square Robot, 2025/2026)** — ~4B Qwen2.5-VL-based, shared
  attention with task-routed FFNs to mitigate VLM forgetting during action training.
- **LingBot-VLA 2.0 (Ant Group, 2026)** — ~6B, one checkpoint reportedly driving 20+ robot
  configurations across many brands; widest embodiment coverage in an open checkpoint. Apache-2.0.
- **Xiaomi-Robotics-0/1 (2026)** — open cross-embodiment VLAs engineered for real-time execution
  (async-execution training, ~80 ms latency).
- ⚠️ The four entries above are 2026-era and least verified; treat numbers as provisional.

## 4. Morphology-aware architectures & adaptation techniques ✅

*The mechanism layer: how do you actually make one network serve many bodies, and how do you
move it to a new one cheaply?*

### 4.1 Embodiment conditioning — four families

| Family | Mechanism | Exemplars | New-robot cost |
|---|---|---|---|
| **Modular I/O** | per-embodiment stems/tokenizers + heads around a shared trunk | HPT, CrossFormer, Octo, GR00T (`EmbodimentTag` projectors) | train a new stem+head; trunk transfers |
| **Conditioning token** | a learned embodiment ID/prompt conditions one monolithic net | X-VLA (soft prompts), embodiment-ID tokens | initialize a new prompt (tiny) + fine-tune |
| **Unified action space** | every robot's action padded into fixed physically-meaningful slots | RDT-1B (128-dim), FAST tokens, delta-EEF convention | no new params; needs a mapping |
| **Learned latent actions** | shared latent action codebook, per-robot decoder | UniAct, LAPA, UniVLA, GO-1 ViLLA | train a small decoder head only |

**Verdict from the evidence**: modular stems (HPT) and soft prompts (X-VLA) have the strongest
published results per parameter; latent-action codebooks (UniAct) have the *cheapest* adaptation
step (decoder head only, extractor frozen). A monolithic model with no embodiment conditioning
at all is the weakest option but the strongest baseline to beat.

### 4.2 Action-space unification — the core design choice
- **Delta end-effector (delta-EEF) + gripper**: what OXE-era models coerce everything to.
  Maximizes transfer because EEF motions are near-embodiment-invariant — but it *hides*
  morphology and breaks when arms lack IK reach parity. **The easy setting.**
- **Joint space**: the honest hard setting — DoF mismatch requires padding/masking. RDT-1B's
  128 fixed physically-interpretable slots is the reference scheme.
- **FAST tokens (PI, 2025)**: DCT per action dimension → quantize → BPE; ~10× compression;
  makes autoregressive VLAs work on high-frequency dexterous data where per-dim binning fails.
- **Flow/diffusion heads**: continuous chunked actions, no tokenization loss — **the dominant
  head type in 2026** (π0, RDT, GR00T, SmolVLA, X-VLA).
- **For this project**: delta-EEF vs RDT-style padded joint-space is the cleanest ablation axis,
  and the contrast is itself a defensible small-scale finding.

### 4.3 Morphology encoding (mostly locomotion literature — a gap for manipulation)
- **MetaMorph (Stanford, 2022)**: serialize the kinematic tree into per-limb tokens; one RL
  policy across 100 UNIMAL robots; combinatorial generalization and fast transfer to unseen
  variants. [arXiv 2203.11931](https://arxiv.org/abs/2203.11931)
- **AnyMorph (Intel/Mila, 2022)**: transfers to unseen morphologies **without any morphology
  description** — learns per-sensor/actuator embeddings from trajectory data. Attractive when
  URDFs don't align cleanly. [arXiv 2206.12279](https://arxiv.org/abs/2206.12279)
- **URMA — "One Policy to Run Them All" (TU Darmstadt, 2024)**: attention encoder over per-joint
  observation sets + universal morphology decoder; one locomotion policy across quadrupeds,
  humanoids, and a hexapod, transferring zero/few-shot to unseen robots in sim *and real*.
  The strongest locomotion-side instantiation of this paradigm; trains on one workstation GPU.
  [arXiv 2409.06366](https://arxiv.org/abs/2409.06366)
- **Body Transformer (Berkeley, 2024)**: masked attention restricted to the robot's body graph —
  each sensor/actuator node attends to itself and its neighbors. A drop-in structural inductive
  bias. [arXiv 2408.06316](https://arxiv.org/abs/2408.06316)
- **GET-Zero (Stanford, 2024)**: graph-attention conditioned on URDF connectivity enables
  **zero-shot** control of modified hardware (removed joints, extended links), +20% on unseen
  variants. [arXiv 2407.15002](https://arxiv.org/abs/2407.15002)
- **Gap worth noting in a write-up**: URDF/graph morphology conditioning is well developed in
  locomotion RL and largely *unexplored* in manipulation VLAs. That gap is a legitimate research
  angle for this project.

### 4.4 Visual embodiment bridging — transfer without touching the policy
- **Mirage (Berkeley, RSS 2024)**: zero-shot transfer by **cross-painting** — mask the target
  robot in the image and render the source robot at the same EEF pose via URDF, so the policy
  only ever *sees* the robot it was trained on. No fine-tuning at all. Needs URDFs, camera
  calibration, similar workspaces, two-jaw grippers. [arXiv 2402.19249](https://arxiv.org/abs/2402.19249)
- **RoVi-Aug (Berkeley, CoRL 2024 oral)**: the *training-time* version — diffusion-based
  augmentation repainting demos with different arms and viewpoints, so the policy is simply
  trained to be embodiment/viewpoint invariant. Up to ~30% improvement when fine-tuning on target
  data. [arXiv 2409.03403](https://arxiv.org/abs/2409.03403)
- **Shadow (Stanford, CoRL 2024)**: replace the robot in *both* training and test images with a
  composite segmentation-mask "shadow" of source+target — cheap, robust to calibration noise, no
  generative model needed. **The simplest visual-bridging trick to implement in a MuJoCo
  pipeline; negligible compute.** [arXiv 2503.00774](https://arxiv.org/abs/2503.00774)
- **OXE-AugE (Berkeley AUTOLab, Dec 2025)**: RoVi-Aug at dataset scale — repaints OXE with 9
  different arms/grippers, tripling it to 4.4M trajectories, improving performance on augmented
  robots, on *unseen* robots, and on original robots under distribution shift. The sim analogue
  (swap the arm model, regenerate the same task) is exactly this project.

### 4.5 Fine-tuning practice — the 2026 standard recipe
1. **LoRA on all linear layers** (OpenVLA official: rank 32, all-linear, no quantization —
   matched full fine-tuning quality on one consumer GPU).
2. **Always re-initialize the action head**, and the action expert if diffusion/flow.
3. **Vision encoder**: freezing is common, but OpenVLA found *unfreezing* helps. Consensus 2026:
   LoRA the VLM backbone, retrain the action head.
4. **Adopt the OFT recipe** (parallel decoding, action chunking, continuous actions, L1 loss) —
   backbone-agnostic, and it is what closed the LIBERO gap.
5. ⚠️ **The single most common silent failure — normalization statistics.** OXE stats are
   computed *per sub-dataset* (q01/q99 quantiles stored in checkpoint `norm_stats`); at inference
   you must pass the matching `unnorm_key`. **For a new embodiment you must compute fresh stats
   from your fine-tuning set** — reusing source-robot stats silently produces garbage actions.
   Log the stats used in every eval run.

### 4.6 "N demos to adapt to a new robot" — the field's datapoints

The single most useful table in this document: **everything clusters at 50–300 demos** for a new
arm on a known task family when starting from a strong generalist. That is the number this
project should reproduce and try to beat, and the axis its curves should sweep.

| Model / method | Demos for new robot | Setting | Notes |
|---|---|---|---|
| **Mirage** (Berkeley 2024) | **0** (zero-shot) | Franka↔UR5, gripper swaps | image cross-painting; needs calibration + URDF |
| **RDT-1B** (2024) | 1–5 | new skills | few-shot skill acquisition |
| **RoboTwin 2.0** (2025) | 10 real + synthetic pretrain | real bimanual | +367% relative over 10-demo-only |
| **OpenVLA** (2024) | 10–150 per task | new tasks, incl. embodiments absent from pretraining | LoRA r=32 |
| **GR00T N1** (NVIDIA 2025) | **30 / 100 / 300 per task** | RoboCasa, DexMimicGen, GR-1 | 45% avg at 100 demos vs 33.4% Diffusion Policy |
| **Octo** (Berkeley 2024) | **~100** | new embodiments, new obs/action spaces | <5 h on one consumer GPU; +52% over next baseline |
| **Gemini Robotics** (2025) | ~100 | new short-horizon tasks | — |
| **Gemini On-Device** (2025) | **50–100** | new tasks; ALOHA→Franka→Apollo | first DeepMind fine-tunable VLA |
| **Gemini On-Device 2** (2026) | **<200, a few hours** | whole new bi-arm platform | SO-101 6.7% → 53.3% |
| **RoboCat** (DeepMind 2023) | **100–1,000** | new tasks *and* unseen embodiments | + self-improvement loop |
| **π0 / openpi** (PI 2024–25) | **1–20 hours of data** | new tasks/platforms | ≈ low-hundreds to few-thousand episodes |

## 5. Industry landscape ✅

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
- **Demo style — a cautionary tale** 🔧: The NEO launch videos (cinematic home chores) went
  viral, then journalists found that essentially all complex tasks in hands-on sessions were
  remotely operated — in a WSJ session ~100% was Expert Mode — plus privacy blowback over remote
  operators seeing inside homes. **Precision matters here**: 1X *did* disclose teleoperation at
  the 28 Oct 2025 launch — "Expert Mode" was a headline element, and the CEO framed it as
  deliberately not over-promising. The criticism was about the *degree* of reliance and the lack
  of **per-clip labeling** of which segments were autonomous. The lesson for demo-makers is
  therefore sharper than "don't hide teleop": **label autonomy shot by shot**, because a blanket
  disclosure buried in a launch post does not protect you.
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
- **What they claim** 🔧: Founded by **Tony Zhao and Cheng Chi** (first authors of ALOHA/ACT and
  Diffusion Policy/UMI), Sunday exited stealth **19 Nov 2025** with Memo, a wheeled home robot
  powered by **ACT-1**, a "zero robot data" skill foundation model trained purely on human
  demonstrations captured with a low-cost **Skill Capture Glove** — sidestepping $20k teleop
  rigs. Funding is two rounds, not one: **$35M** from Benchmark and Conviction at the stealth
  exit, then a **$165M Series B at $1.15B led by Coatue in March 2026**. The widely repeated
  "$200 glove" figure is press estimate (reports range $200–400); Sunday has not published a price.
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

## 6. Datasets, benchmarks, simulators ✅

### 6.1 Datasets

**Open X-Embodiment (OXE)** (GDM + 34 labs, 2023, still the substrate in 2026) — 1M+ real
trajectories, 22 embodiments, 527 skills, 60 source datasets converted to RLDS. **Pitfalls that
this project's unified interface must solve:** (1) *action-space inconsistency* — the same action
vector means different things per sub-dataset (relative vs absolute, different EEF frames,
different gripper conventions); most consumers coerce to delta-EEF + gripper and drop the rest.
(2) *camera conventions vary* (pose relative to robot, intrinsics, which camera is "primary").
(3) *normalization statistics are per-sub-dataset* and must be matched at inference.
(4) *control-frequency mismatch*. Also note the effective diversity is lower than advertised —
the top four robot types account for **over 85%** of the real data.
[arXiv 2310.08864](https://arxiv.org/abs/2310.08864)

**DROID** (Stanford/Berkeley + 13 institutions, 2024) — 76k teleoperated Franka trajectories,
~350 h, 86 tasks, **564 scenes in 52 buildings on 3 continents**, 50 operators, 12 months, on a
standardized rig (Franka Panda + Robotiq, 2 external Zed 2 stereo + wrist Zed Mini, Quest 2
teleop), every episode with calibrated multi-view RGB-D + language. Single embodiment, so not a
transfer dataset — but it is the reference for what a *clean per-embodiment data standard* looks
like, and DROID-pretrained checkpoints are good adaptation sources.
[arXiv 2403.12945](https://arxiv.org/abs/2403.12945)

**BridgeData V2** (Berkeley RAIL, 2023) — 60,096 trajectories on a $3k WidowX 250 across 24
environments; the standard low-cost-arm pretraining corpus and the substrate of SimplerEnv's
WidowX eval. Republished in LeRobot v3 format.
[arXiv 2308.12952](https://arxiv.org/abs/2308.12952)

**AgiBot World** (AgiBot + OpenDriveLab, 2025) — **1,001,552 trajectories, 2,976 hours, 217
tasks, 87 skills, 106 scenes**, from a fleet of 100 identical dual-arm mobile manipulators.
Policies pretrained on it beat OXE-pretrained ones by ~30% in- and out-of-distribution — useful
evidence that *homogeneous* scale can beat heterogeneous aggregation for a target platform.
[arXiv 2503.06669](https://arxiv.org/abs/2503.06669)

**RoboMIND** (X-Humanoid, 2024; 2.0 in Dec 2025) 🔧 — **the best deliberately multi-embodiment
teleop dataset**: 107,517 trajectories, 479 tasks, 96 object classes, collected under one unified
protocol so embodiment is the controlled variable. Corrected per-embodiment counts: Franka Emika
Panda **26,856**, UR5e **25,170**, Tien Kung humanoid **15,187**, AgileX Cobot Magic V2.0
**10,269** — the four *real* embodiments total **77,482**; the headline 107k figure additionally
includes **30,035 simulation** trajectories. RSS 2025. **The closest existing real dataset to
this project's sim design** — worth citing as real-world corroboration.
[arXiv 2412.13877](https://arxiv.org/abs/2412.13877) · RoboMIND 2.0: [arXiv 2512.24653](https://arxiv.org/abs/2512.24653)

**LeRobot ecosystem** (HuggingFace, 2024–2026) — **the de-facto open tooling stack**: the
`LeRobotDataset` v3 format (episodes packed into Parquet + MP4 video chunks with relational
metadata, streamable from the Hub without full download), training code, a policy zoo (ACT,
Diffusion Policy, π0, π0-FAST, SmolVLA, GR00T), and the $100-class SO-100/SO-101 5-DoF
3D-printable arms that spawned 1,000+ community datasets. NVIDIA, Physical Intelligence, and
Google all publish LeRobot-format checkpoints and tutorials.
**→ This project's data pipeline should emit LeRobotDataset v3 regardless of simulator** — it
makes every mainstream policy trainable on our data with zero conversion work, and it is the
format reviewers expect in 2026.
[format docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) · [code](https://github.com/huggingface/lerobot)

### 6.2 Which benchmarks actually support *the same tasks across multiple embodiments*

This is the decisive question for the project. Answer:

| Benchmark | Same tasks across N embodiments? | Arms | Demo generation | Engine |
|---|---|---|---|---|
| **robosuite + MimicGen** ✅ | **Yes, by design** — one-argument robot swap; MimicGen ships a 4-arm × same-tasks subset | 8 embodiments: 6 single-arm (Panda, Sawyer, UR5e, Kinova3, IIWA, Jaco) + 2 bimanual (Baxter, ALOHA) | MimicGen from ~10 source demos | **MuJoCo** |
| **ManiSkill3** ❌ | Tasks accept an embodiment arg, but there is **no official cross-embodiment demo set** | Panda, xArm6, WidowX AI, SO-100, + | motion-planning scripted experts, RL, teleop | SAPIEN/PhysX (GPU) — **will not install on aarch64** |
| **RoboTwin 2.0** | Yes — 50 bimanual tasks × 5 embodiments | 5 dual-arm platforms | MLLM code-gen experts | SAPIEN |
| **AnyBody** | Yes — reach/push × 18 morphologies | 18 procedural + real-based | RL | Isaac Sim |
| **RoboCasa** | Partial — several form factors | mobile manipulators, humanoids | MimicGen | MuJoCo |
| **LIBERO / Meta-World / SimplerEnv** | **No** — fixed embodiment | 1–2 | — | MuJoCo / SAPIEN |

**robosuite + robomimic + MimicGen** (ARISE Initiative / NVIDIA, 2020–2026) ✅ — MuJoCo-based, and
**every task is robot-agnostic**: the same Lift/Stack/PickPlace/NutAssembly/Door/ToolHang
environment runs on every supported robot with a one-argument change; reward functions,
observation spaces, and controllers adapt automatically. **Independently verified on this
machine: 48/48 (6 tasks × 8 embodiments) construct, reset, and step on robosuite 1.5.2.**
🔧 Two caveats the marketing glosses: **ALOHA ships in the separate `robosuite_models` package**,
not core robosuite (two pip installs), and Baxter and ALOHA are **bimanual** (dual-arm), so the
honest phrasing is "8 robot embodiments, 6 single-arm + 2 bimanual." The Kinova Gen3 API name is
`Kinova3`. Note also that constructing and stepping is not the same as *solving* — a per-arm
reachability audit is still required.
MimicGen transforms *object-relative EEF trajectory segments*, which are embodiment-independent —
hence its released dataset includes a **"robot transfer" subset of ~16,000 generated demos across
4 arms (Panda, Sawyer, IIWA, UR5e) on the same tasks, generated from Panda-only source demos**.
MimicGen overall: <200 human demos → 50k+ generated demos across 18 tasks. RoboCasa found
MimicGen-generated data *beats* human data at equal cost (47.6% vs 28.8%).
[robosuite](https://robosuite.ai/) · [MimicGen arXiv 2310.17596](https://arxiv.org/abs/2310.17596)

**ManiSkill3** (UCSD Hao Su Lab, 2024–2026) — GPU-parallelized (up to 30k+ FPS with rendering),
20+ embodiments, tasks that accept an embodiment argument, SimplerEnv integrated, 10–1000× faster
with 2–3× less GPU memory than comparable platforms. **Two claims about it did not survive
verification:**
- ❌ **There is no official cross-embodiment demo set.** No "6 tasks × 4 arms × 100 episodes"
  dataset exists; demonstrations are distributed per-task by environment ID. Third-party work
  (MOTIF) *generated* its own cross-embodiment suite inside ManiSkill. Additional trap: there is
  no standard-gripper xArm7 agent — only `xarm7_ability` (xArm7 + Ability dexterous hand) — so
  the claimed 4-arm parallel-jaw set is not even constructible.
- ❌ **It cannot be installed on this DGX Spark.** PyPI has **zero aarch64 wheels for SAPIEN**
  across all 24 releases, and `mplib` has none either, so `pip install mani_skill` fails outright
  on aarch64. Unofficial aarch64 SAPIEN wheels on GitHub Releases both fail to import on
  GB10/glibc 2.39. **Treat ManiSkill3 as literature, not infrastructure, for this project.**
[arXiv 2410.00425](https://arxiv.org/abs/2410.00425)

**MuJoCo Menagerie + Playground** (GDM, 2022–2026) — Menagerie is curated, quality-graded MJCF
models of essentially every arm you'd want (Panda, FR3, UR5e, UR10e, xArm7, Lite6, KUKA iiwa14,
Sawyer, Unitree Z1, ViperX 300, **SO-100/SO-101**, ALOHA 2, plus grippers and dexterous hands).
Playground adds MJX/Warp GPU-accelerated environments with Madrona batch rendering and the
train-in-minutes-on-one-GPU recipe. Building a custom suite on Menagerie gives maximum control
and runs natively on the Spark's aarch64 + CUDA via JAX — at the cost of writing your own tasks,
randomization, and demo-generation plumbing that robosuite/ManiSkill already ship.
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie) · [Playground](https://github.com/google-deepmind/mujoco_playground)

**LIBERO** (UT Austin, 2023) — 130 language-conditioned Franka tasks in robosuite/MuJoCo; the
universal VLA fine-tuning benchmark. **Single-embodiment and now saturated** (OpenVLA 76.5% →
OpenVLA-OFT 97.1%; community SOTA ~98%), though on LIBERO-OOD variants all tested VLAs fall
below 21%. Good for validating fine-tuning code against published numbers; **useless for
cross-embodiment claims**. [arXiv 2306.03310](https://arxiv.org/abs/2306.03310)

**SimplerEnv** (UCSD/Berkeley, CoRL 2024) — real-to-sim *evaluation* environments reproducing
Google Robot and WidowX+Bridge real setups, with "visual matching" and "variant aggregation"
protocols that correlate sim scores with real rollouts. Now ported into ManiSkill3 with GPU
parallelism. **A model for this project's eval protocol** (fixed visual-matching scenes, pose
sweeps) rather than a training ground. [code](https://github.com/simpler-env/SimplerEnv)

**AnyBody** (Princeton VL, 2025) — purpose-built for *morphology generalization*: 18 robot
variations (8 procedurally generated + 10 real-robot-based) on reach/push tasks, with
**interpolation / extrapolation / composition test splits**. Its split taxonomy is the cleanest
existing formalization of "held-out embodiment" evaluation axes — **borrow it for our eval
protocol even though we won't use Isaac Sim.** [arXiv 2505.14986](https://arxiv.org/abs/2505.14986)

**RoboCasa** (UT Austin / NVIDIA, 2024) — 120 kitchen scenes, 2,500+ objects, 100 tasks, 100k+
MimicGen trajectories, multiple form factors. MuJoCo-native but targets mobile manipulation in
large scenes — heavier than needed for a controlled N-arms study. [arXiv 2406.02523](https://arxiv.org/abs/2406.02523)

**Isaac Lab / Isaac Sim** (NVIDIA) — GPU-parallel, **buildable natively on aarch64/DGX Spark**
(ARM publishes a DGX Spark Isaac learning path), with mimic-style data generation and teleop.
But its manipulation tasks are robot-specific configs, not a robot-agnostic suite — making 5 arms
share one task suite is more custom work than robosuite or ManiSkill.

**Genesis** — impressive throughput claims (43M FPS on simplified state-based scenes) but the
manipulation benchmarking layer (standard task suites, demo datasets, eval protocols) is still
thin, and aarch64 support is unverified. **Watch, don't build on.**

### 6.3 Sim data generation
**MimicGen-family demo multiplication is the proven route**: collect ~10 source demos once,
regenerate thousands per arm automatically, because object-relative EEF segment transforms are
embodiment-independent. DexMimicGen extended this to bimanual/dexterous (60 human demos → 21k+,
ICRA 2025). RoboTwin 2.0 uses MLLM-written scripted experts with sim-in-the-loop refinement.
ManiSkill3 ships motion-planning scripted experts. Teleop-in-sim is the fallback for tasks
automation can't reach.

## 7. Synthesis — implications for this project

### 7.1 What the field says we should build

The literature converges on a protocol this project can execute at small scale with high fidelity:

1. Train one policy on **N robots sharing a task suite**, hold out one or two arms entirely.
2. Adapt to the held-out arm on a **demo budget swept from 0 to ~300** — the range where the
   entire field's results live (§4.6).
3. Report **success vs demos** and **success vs GPU-hours**, against a from-scratch baseline.
4. Include the control that most demos skip: **pretrained-on-one-arm** as well as
   pretrained-on-N-arms, so the claim is "multi-embodiment pretraining helps," not merely
   "pretraining helps."
5. Test each robot on tasks whose data exists **only on another robot** (Gemini 1.5's benchmark
   design), reporting completion, not just progress.

### 7.2 Locked design decisions (M0)

| Decision | Choice | Why |
|---|---|---|
| **Simulator** | **robosuite 1.5.2 + MimicGen + robomimic** (MuJoCo) ✅ verified on the Spark | Tasks are robot-agnostic *by construction* — one-argument robot swap. MimicGen has already demonstrated exactly our protocol (Panda-source demos regenerated for Sawyer/IIWA/UR5e). Pure MuJoCo runs natively on aarch64. **Pin `mujoco==3.3.7`** — robosuite 1.5.2 calls `MjData.qM`, removed in MuJoCo 3.11. Headless rendering needs `MUJOCO_GL=egl`. |
| **Scale-up option** | ❌ **ManiSkill3 is ruled out** | No aarch64 wheels exist for SAPIEN or mplib, and unofficial builds fail to import on GB10. If GPU-parallel rollouts become the bottleneck, the fallback is **MuJoCo Playground / MJX** (JAX, aarch64-native), not ManiSkill. |
| **Training arms** | Panda, Sawyer, IIWA, Kinova Gen3 | All in robosuite, same parallel-jaw gripper class, so gripper effects don't confound morphology effects. |
| **Held-out (near)** | UR5e | Same class, similar DoF/reach → "interpolation" in AnyBody's taxonomy. |
| **Held-out (far)** | SO-101 (imported from MuJoCo Menagerie) | 5-DoF, short reach, hobbyist-grade → "extrapolation." Also the exact arm Gemini On-Device 2 reports on, giving an external reference point. |
| **Tasks** | 6–8 robosuite tasks across the precision spectrum: Lift, Stack, PickPlace-Can, NutAssembly-Square, Door, ToolHang | Verify reachability per arm before locking; SO-101's workspace will force scaled-down scene variants — document the scaling. |
| **Data route** | ~10 scripted/human source demos per task → MimicGen-regenerate ~1,000 per (task, arm) | ≈30–40k trajectories, a weekend of CPU generation. Held-out arm gets small adaptation sets of {5, 10, 25, 50, 100, 300}. |
| **Data format** | **LeRobotDataset v3** | Makes π0, GR00T, SmolVLA, X-VLA, ACT, Diffusion Policy all trainable on our data with zero conversion, and it's what reviewers expect in 2026. |
| **Action head** | OpenVLA-OFT recipe: parallel decoding, action chunking, continuous actions, L1 (or flow) | Backbone-agnostic, Spark-friendly, and it is what closed the LIBERO gap. |
| **Policy A (ours)** | Small transformer, HPT-style per-embodiment stems + shared trunk, ~50–200M | Full training on Spark; the architecture whose thesis we are testing. |
| **Policy B (pretrained)** | **X-VLA-0.9B** primary, **SmolVLA-450M** fallback, GR00T N1.5 if we want NVIDIA's supported pipeline | Sub-1B fully fine-tunes on Spark; X-VLA's soft-prompt conditioning is the cleanest published new-embodiment mechanism. |

### 7.3 The distinctive contribution — decomposing the embodiment gap

A pure reproduction is a weak portfolio piece. The thing simulation makes possible that the real
world does not: **separating the visual gap from the kinematic gap.** In sim we can render arm A's
appearance while executing arm B's kinematics, and vice versa — a factorial 2×2 that no real-robot
lab can run. Combined with Shadow-style robot masking (§4.4, nearly free to implement in MuJoCo),
this yields a clean answer to a question the field currently hand-waves: *when a policy fails on a
new robot, is it because the robot looks different or because it moves differently?*

Secondary angles, in order of cost-effectiveness:
- **Interface ablation**: delta-EEF (transfer-friendly) vs RDT-style padded joint-space
  (transfer-hostile). A defensible small-scale finding on its own.
- **Conditioning ablation**: no conditioning vs embodiment ID token vs per-arm stems vs soft
  prompts — the four families of §4.1 on one controlled suite. Nobody has published this
  head-to-head at matched scale.
- **Morphology encoding for manipulation** — the gap identified in §0. Feeding a kinematic-graph
  embedding (URMA/GET-Zero style) into a manipulation policy is genuinely underexplored.

### 7.4 Evaluation protocol

- **≥50 rollouts** per (task, arm, condition) with randomized object poses; **3 training seeds**;
  report mean ± standard error. Blind/scripted scoring where possible.
- **Fix the camera rig and scene identically across arms** so embodiment is the only variable
  (SimplerEnv's visual-matching discipline).
- **Headline metric**: *demos-to-X%* — demos needed to reach 80% of the 300-demo ceiling — plus
  GPU-hours on the Spark as a second x-axis. Spark's ~273 GB/s bandwidth makes wall-clock a
  genuinely interesting practical metric, not an apology.
- **Report AnyBody-style splits separately**: UR5e (interpolation) and SO-101 (extrapolation) get
  their own numbers, never a blended "transfer" figure.
- **Frame results against the field's cluster**, not against LIBERO: Octo ~100, GR00T 30/100/300,
  Gemini On-Device 2 <200. The defensible claim is *"our sim protocol reproduces the industry's
  adaptation-demo curve and isolates why it works."*

### 7.5 Pitfalls to design against (each has burned published work)

1. **Normalization statistics** — recompute action/proprio q01/q99 per embodiment *and* per
   adaptation set; never reuse the pretraining `unnorm_key` for the held-out arm. Log the stats
   used in every eval run. This is the single most common silent failure in cross-embodiment
   fine-tuning.
2. **Action-space aliasing** — confirm every arm's controller uses the same convention (frame,
   rotation parameterization, gripper polarity). robosuite's OSC controllers give this for free;
   other stacks need checking.
3. **Expert-style overfitting** — MimicGen/motion-planner demos have a characteristic style;
   policies can look great in-distribution and collapse on the held-out arm for reasons unrelated
   to embodiment. Include a noise-injected or teleoped subset and evaluate on object-pose
   distributions wider than the generation distribution.
4. **Benchmark saturation** — don't headline LIBERO numbers.
5. **Silent capping** — if we drop tasks or arms for reachability reasons, say so explicitly in
   the write-up and the README.

### 7.6 Showcase implications (from §5's demo-craft synthesis)

The demo formats that earn technical credibility map directly onto this project's milestones:
- *One brain, many bodies* — identical weights driving N arms simultaneously in a grid. This is
  Figure's "same weights, two robots" proof, which is the single most persuasive generality demo.
- *Learning race* — split screen, pretrained-adapted vs from-scratch on the held-out arm, with a
  live success counter, the counted-repetitions format Generalist uses.
- *Perturbation reel* — objects shoved mid-episode, on camera, in one take (Skild/Atlas format).
- *Failure reel* — short, honest, and it buys more credibility than it costs.
- *Data-engine shot* — MimicGen regenerating one source demo across five arms, visualized. Shows
  the pipeline, not just the result.
- **Norms to follow without exception**: label real-time vs sped-up on every clip; prefer uncut
  takes; state the demo count and seed behind every number shown.

## 8. Reading queue

Spotted but not yet summarized in depth:
- DexMimicGen (ICRA 2025) — bimanual/dexterous demo multiplication, 60 demos → 21k+.
- **MOTIF** (arXiv 2602.13764, Feb 2026, ICRA 2026) — vector-quantized embodiment-agnostic
  "action motifs" for few-shot cross-embodiment transfer, evaluated on a self-built **three**-arm
  ManiSkill setup (Franka Panda, xArm6, WidowX AI) plus two real arms (ARX5, Piper). **The
  closest 2026 academic work to this project — read it first.**
- ET-VLA (2025) — synthetic continued pretraining for new embodiments.
- Being-H0.5 (BAAI, 2026) — human-centric pretraining for cross-embodiment generalization.
- Cosmos world models (NVIDIA) — synthetic data generation for robotics.
- 1X World Model / UnifoLM-WMA-0 (Unitree) — world models as policy evaluators, an alternative
  to rollout-based eval.
- TRI LBM follow-up on co-training data-modality tradeoffs (2026).
