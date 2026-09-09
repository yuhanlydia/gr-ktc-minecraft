# GR-KTC experiment status — 2026-08-27

## Verified results

- Gate 0 local protocol: passed. Minecraft 1.19 dedicated server, Java 17,
  Mineflayer bridge, RTX 3090, Qwen3-VL-8B, MineExplorer 813+100 rows, parser,
  and local no-API execution are operational.
- Real local Qwen gather rollout: parser-valid and environment-successful for
  `Collect 4 oak logs`; final inventory was `oak_log: 4`.
- Deterministic prerequisite craft smoke: 10/10 parser-valid and 10/10
  environment success over T1–T5 and generation seeds 42/43. BF16 peak allocated
  GPU memory was 16.424 GiB. These are integration scenes, not PEAM held-out
  outcomes.
- Real K=4 acquisition group: four T1 rollouts, four successes, 17 immutable
  checksummed files. This group supplies BC data but no group-relative signal
  because its outcome variance is zero.
- QLoRA smoke: rank 32, alpha 64, seven PEAM-matched projection families,
  87,293,952 trainable parameters. One real-rollout BC update had loss 1.834,
  took 2.97 seconds, and peaked at 9.960 GiB.
- MineExplorer Pilot acquisition: 30 finalized composite groups / 120
  trajectories, including 12 mixed-outcome groups.
- Gate 1 passed on the preregistered Pilot size. Layer-24 direction cosine was
  33/42 = 78.57%, group-bootstrap 95% CI [0.571, 0.955], random-label
  one-sided p=0.0029. Layer 35 did not pass and is not selected downstream.
- Gate 2 passed in the matched real-context causal control: layer-24 quality
  memory succeeded on 14/16 trials versus 5/16 with no memory (exact McNemar
  p=0.003906) and 7/16 with failed memory (p=0.015625). Failed memory did not
  improve over no memory (p=0.6875). All-36-layer BF16 prefix inference peaked
  at approximately 16.5 GiB.
- Gate 3 passed its preregistered directional condition on related held-out
  MineExplorer contexts: memory-free GR-KTC slow QLoRA achieved 12/16 versus
  shared BC+DPO QLoRA 10/16 and base 7/16. The GR-KTC versus BC+DPO difference
  is not significant at this pilot size (two discordant pairs, p=0.5).
- Gate 4 BF16 completed all 99 unique task/seed/method trials with pristine-world
  restoration before every condition, a 2,048-token cap, and at most four
  retries. Local Qwen base achieved 6/33, shared BC+DPO 2/33, and memory-free
  GR-KTC 6/33. GR-KTC recovered all four trials lost by BC+DPO (exact McNemar
  p=0.125) and matched base exactly, but did not exceed it. The earlier complete
  NF4 diagnostic was 6/33, 3/33, and 5/33 respectively and is retained as a
  precision ablation. See `PEAM_COMPATIBLE_REPORT.md`.
- Unit/integration tests: 55 passed.

## Synthetic mechanism gate

Across seeds 42/43/44 and 90 groups per seed, leave-group latent ranking AUROC
was 0.9985 / 0.9981 / 0.9958. Closed-form rank-4 LoRA reconstruction relative
error was approximately `1e-7`.

Merge alignment was:

| Seed | A | B | C | D-SB |
|---:|---:|---:|---:|---:|
| 42 | 0.810 | 0.903 | 0.987 | 0.065 |
| 43 | 0.940 | 0.230 | 0.957 | 0.343 |
| 44 | 0.922 | 0.305 | 0.964 | 0.608 |

This is a mechanism sanity check, not Minecraft success. Experimental
Schrödinger Bridge Method D is currently worse and unstable; it is not promoted
to the main method. A temporal-cost/epsilon sweep improved individual seeds but
did not yield one stable setting.

Method D now has a soft-control variant for K=4: effective-sample-size and
advantage-margin shrinkage toward the identity, plus an optional control-norm
cap. It remains an ablation until it beats A/B under the same real-data gates.

## Reproduction differences

- PEAM uses Azure GPT-4o as the slow-tier acquisition/fallback model. The current
  protocol uses local Qwen3-VL-8B for rollout sampling and therefore labels PEAM
  paper numbers as reference-only.
- Voyager pins Mineflayer 4.8.1. Its crafting transaction hung on the local 1.19
  dedicated server; Mineflayer 4.14.0 fixed the reference craft action and is
  pinned here.
- Headless hard reset disables the original `/kill` step because it causes an
  invalid movement packet race in Mineflayer. Inventory/equipment and setup
  commands remain deterministic.

## Claim boundaries

- The completed 11-task × 3-seed run is a local-Qwen, PEAM-compatible stress
  test, not an official PEAM reproduction. It does not match PEAM's Azure
  GPT-4o slow tier, accumulated skill pool, or A100 serving setup.
- A/B/C are implemented and mechanism-tested; Gate 2 tests the selected
  contrastive layer-24 fast memory. A full 33-trial A/B/C endpoint comparison
  has not been run and is not claimed.
- The public MineExplorer execution repository is unavailable; its dataset and
  verifier schema are implemented, but official 1,800-step parity is unverified.
