# GR-KTC Minecraft
## Latest experiment: Prompt Decommitment

The active research branch is a **prompt-only, training-free MineExplorer experiment**.
It does not modify attention masks, KV caches, or model weights. Instead, each
closed-loop decision is assigned a discrete event ID and the model may explicitly
revoke prior thought/action commitments that have become contradicted, completed,
or obsolete under new visual evidence.

Primary comparison: `base` vs `reflection` vs `decommit_ignore` vs `decommit`.
The critical mechanism control is `decommit_ignore`, which uses the same longer
prompt and five-key JSON schema but does not persist revocations. Primary benchmark
metrics are official MineExplorer TSR and MSR, stratified by 1/2/3/4-hop tasks.

16GB configuration: `configs/prompt_decommitment_16gb.yaml`.
Experiment protocol: `docs/PROMPT_DECOMMITMENT_EXPERIMENT.md`.
Runner: `scripts/run_mineexplorer_prompt_decommitment.py`.
Analyzer: `scripts/analyze_prompt_decommitment.py`.


Implementation workspace for group-relative KV trajectory consolidation on the
Voyager/Mineflayer substrate.

## Current status

Implemented and tested:

- verifier-based group-relative advantages;
- incremental K/V recording compatible with legacy and modern Transformers caches;
- acquisition-only PCA whitening;
- signed group-relative covariance and positive/negative eigenspaces;
- merge A residual barycenter;
- merge B differentiable soft-DTW phase barycenter;
- merge C entropic Sinkhorn token-cloud phase barycenter;
- closed-form ridge/SVD LoRA initialization;
- bounded verifier-closed retry loop;
- RTX 3090 24 GB NF4 Qwen3-VL loader and configuration.

Completed experimental gates include a 120-trajectory acquisition pilot,
leave-one-group-out latent geometry, matched fast-KV causal controls, 100-step
QLoRA consolidation, and all 99 local-Qwen PEAM-compatible task/seed/method
trials. See [`results/STATUS.md`](results/STATUS.md) and
[`results/PEAM_COMPATIBLE_REPORT.md`](results/PEAM_COMPATIBLE_REPORT.md).

The current research direction is State–Weight Reachability: useful KV state
corrections are decomposed into parameter-reachable (slow LoRA) and
parameter-unreachable (fast KV) components. The four API-free analysis paths
and the full 813-scenario MineExplorer sweep are documented in
[`docs/FOUR_NEW_IDEAS.md`](docs/FOUR_NEW_IDEAS.md). The sweep is a mechanism
test over activation-shaped tensors; real VLM KV/hidden captures are required
for causal behavioral claims.

A first real-Qwen reachability pilot is now complete on two causal quality-KV
teacher contexts, including rank 1–64 offline fits and 80 paired Minecraft
behavior trials at ranks 8 and 32. See
[`results/REAL_REACHABILITY_REPORT.md`](results/REAL_REACHABILITY_REPORT.md).
It finds a persistent individual–joint reachability gap, while also showing
that high linear hidden-effect fit is not by itself sufficient for behavioral
retention.

The runtime workspace can use the official Voyager repository, Qwen checkpoint,
MineExplorer dataset, and paper PDFs. These large/downloaded resources are not
committed; see `RESOURCE_MANIFEST.md` for their pinned sources.

## Test

```bash
python -m pytest -q
```

## Fast-loop semantics

Each matched context runs four rollouts. Only a mixed verifier-outcome group has
nonzero relative advantages and may update fast memory. The controller stops on
verified success, retry-budget exhaustion, or lack of improvement. Fast memory is
task-local and must be reset after task completion.

## 24 GB constraints

Use `configs/grktc_24gb.yaml`. The original checkpoint is loaded in NF4 with bf16
compute. Pilot runs record two text layers, cap generation at 512 tokens, and
offload incremental K/V to CPU. Slow consolidation uses QLoRA with microbatch 1
and gradient accumulation 16.

## API-free local stack

Minecraft 1.19, Java 17, the Voyager bridge, and Qwen3-VL-8B are installed. The
default experiment stack uses a local-only dedicated server in offline mode, so
neither Microsoft login nor a GPT/Azure API is required:

```bash
scripts/start_local_stack.sh
python scripts/run_local_qwen_action.py --task "Collect 4 oak logs" --execute
```

Offline mode is bound to `127.0.0.1` and must never be exposed to an untrusted
network. Qwen is both the fast policy and local rollout sampler. This is a valid
API-free GR-KTC protocol, but it is not an exact reproduction of PEAM's GPT-4o
slow-tier acquisition. PEAM's reported line therefore remains reference-only;
any direct statistical comparison must use the explicitly labelled local
acquisition control or a separately reproduced slow tier.
