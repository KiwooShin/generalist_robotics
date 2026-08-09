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

## 2. Google DeepMind lineage

*(agent running — RT-1 → RT-2 → RT-X / Open X-Embodiment → RoboCat → ALOHA Unleashed →
Gemini Robotics 1.0 / On-Device / 1.5 + Motion Transfer)*

## 3. Open cross-embodiment models

*(agent running — Octo, OpenVLA/-OFT, CrossFormer, HPT, RDT-1B, GR00T N1/N1.5, SmolVLA,
LAPA, UniVLA, …)*

## 4. Morphology-aware architectures & adaptation techniques

*(agent running — embodiment tokens, per-robot stems/heads, unified action spaces, LoRA
practice, normalization gotchas, few-shot adaptation numbers across the field)*

## 5. Industry landscape

*(agent running — Generalist AI GEN-0, Skild, Figure Helix, 1X, TRI/BD LBMs, Covariant
RFM-1, AgiBot, …)*

## 6. Datasets, benchmarks, simulators

*(agent running — OXE, DROID, BridgeData V2, AgiBot World, LIBERO, SimplerEnv, RoboCasa,
ManiSkill3, MuJoCo Menagerie/Playground, LeRobot ecosystem)*

## 7. Synthesis — implications for this project

*(what we adopt, what we test, experimental design)*

## 8. Reading queue

*(papers spotted but not yet summarized)*
