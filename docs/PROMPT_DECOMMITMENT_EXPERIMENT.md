# Prompt Decommitment — MineExplorer Experiment Protocol

## What this experiment tests

This experiment does **not** test attention routing, KV selection, LoRA, RL, or parameter unlearning. Qwen3-VL-8B is frozen. The sole intervention is a structured prompt state that lets the model revoke obsolete prior decision events.

The benchmark target is MineExplorer task success rate (TSR) and milestone success rate (MSR), especially on 2–4 hop hidden-prerequisite tasks.

## Baselines

| Condition | Extra reflection instruction | Discrete event IDs | Persistent revocation applied |
|---|---:|---:|---:|
| `base` | no | no | no |
| `reflection` | yes | no | no |
| `decommit_ignore` | yes | yes | **no** |
| `decommit` | yes | yes | **yes** |

Do not add Voyager skills, external retrieval, LoRA, reward-model scoring, or hidden milestone text in this experiment. They would confound the first causal test.

## Gate 0 — code / environment smoke

Use one scene per hop (4 total), all four conditions, greedy decoding, 50 steps.

Requirements:

- all 16 runs create `result.json`;
- parser-valid action generation does not systematically collapse under the 5-key PDC JSON schema;
- peak allocated CUDA memory stays below 15 GiB or is offloaded by `device_map=auto` without OOM;
- `decommit_ignore` and `decommit` receive byte-identical control instructions; the only difference is whether requested revocations persist.

Command:

```bash
python scripts/run_mineexplorer_prompt_decommitment.py \
  --mineexplorer-root third_party/MineExplorer \
  --model-path models/Qwen3-VL-8B-Instruct \
  --benchmark-dir third_party/MineExplorer/benchmark \
  --output-dir results/pdc_smoke \
  --per-hop 1 --hop-counts 1 2 3 4 \
  --max-steps 50 --frame-size 20 \
  --temperature 0.0 --seed 42
```

## Gate 1 — cheap phenomenon pilot

Use 12 scenes: 3 each from 1/2/3/4-hop. Run all four conditions with greedy decoding and a 200-step cap. This is not the final benchmark claim; it decides whether explicit revocation has a behavioral signal before paying for full 1,800-step runs.

```bash
python scripts/run_mineexplorer_prompt_decommitment.py \
  --mineexplorer-root third_party/MineExplorer \
  --model-path models/Qwen3-VL-8B-Instruct \
  --benchmark-dir third_party/MineExplorer/benchmark \
  --output-dir results/pdc_pilot \
  --per-hop 3 --hop-counts 1 2 3 4 \
  --max-steps 200 --frame-size 20 \
  --temperature 0.0 --seed 42

python scripts/analyze_prompt_decommitment.py \
  results/pdc_pilot/summary.json \
  --output results/pdc_pilot/report.json
```

### Pilot GO

Continue only when both paired means are positive:

\[
\overline{MSR}_{PDC}-\overline{MSR}_{Reflection}>0
\]

and

\[
\overline{MSR}_{PDC}-\overline{MSR}_{PDC-ignore}>0.
\]

Also inspect `control_records`: revocation must actually occur on multiple scenes. If `revoke_event_ids` is almost always empty, the protocol has not created the intended mechanism even if aggregate MSR fluctuates upward.

### Pilot NO-GO

Stop if PDC does not beat both prompt controls directionally. Do not add LoRA/RL or hand-labelled revoke targets to rescue a negative phenomenon.

## Gate 2 — official-cap confirmatory

Use 24 scenes: 6 per hop. Run `base`, `reflection`, and `decommit` at the official 1,800-step cap. Keep `decommit_ignore` on the first seed as a mechanism check; it may be dropped from seeds 43/44 to reduce compute only after Gate 1 has established that it differs from PDC.

Run decoding seeds 42, 43, 44 with `temperature=0.7`. The local provider resets the random sequence at every scene/method pair so methods receive matched sampling seeds.

Example seed 42:

```bash
python scripts/run_mineexplorer_prompt_decommitment.py \
  --mineexplorer-root third_party/MineExplorer \
  --model-path models/Qwen3-VL-8B-Instruct \
  --benchmark-dir third_party/MineExplorer/benchmark \
  --output-dir results/pdc_full_seed42 \
  --per-hop 6 --hop-counts 1 2 3 4 \
  --max-steps 1800 --frame-size 20 \
  --temperature 0.7 --seed 42
```

Repeat with seeds 43 and 44. For the two later seeds, pass:

```bash
--modes base reflection decommit
```

only if the `decommit_ignore` mechanism control passed Gate 1.

## Primary paper table

Report both overall and by-hop metrics:

| Method | 1-hop TSR | 2-hop TSR | 3-hop TSR | 4-hop TSR | Overall TSR | Mean MSR |
|---|---:|---:|---:|---:|---:|---:|
| Base | | | | | | |
| Reflection | | | | | | |
| PDC-ignore | | | | | | |
| **Prompt Decommitment** | | | | | | |

Primary comparison:

\[
PDC-Reflection.
\]

Mechanism comparison:

\[
PDC-PDC\text{-}ignore.
\]

Use context-paired bootstrap for MSR differences and exact McNemar for paired task success. The included analysis script computes both.

## Mechanism figures

1. **Gain vs hop depth:** paired `PDC - Reflection` MSR/TSR by hop.
2. **Revocation frequency:** number of persistent revoked events per successful vs failed episode.
3. **Time to next milestone after first revocation:** compare PDC trajectories with matched Reflection trajectories.
4. **Ablation:** PDC vs PDC-ignore to show that emitting the same five-key JSON is not enough; the revocation has to persist in subsequent prompts.

## Interpretation boundaries

A positive result supports **prompt-level commitment revision**, not sparse attention or parameter unlearning. The method intentionally leaves all visual frames available and does not alter Transformer attention masks.
