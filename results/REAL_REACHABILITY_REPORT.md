# Real Qwen State–Weight Reachability Pilot

Date: 2026-08-27

## Protocol

- Qwen3-VL-8B-Instruct, BF16 inference on RTX 3090 24GB.
- Two real MineExplorer Gate-2 contexts (`0281`, `0299`).
- Teacher target: layer-24 output hidden shift causally induced by the matched
  quality-KV memory used in the earlier fast-rescue experiment.
- Reachability proxy: rank-constrained residual-stream update fitted from the
  no-memory layer input to the KV-induced output shift.
- Behavior conditions: Base, State (quality KV), Weight-individual,
  Weight-shared, and full-State + shared-Weight upper-bound control.
- Four sampling seeds per rank, with Minecraft reset before every condition.

This is a real-model/real-environment pilot. It is not an official PEAM result,
and the residual-stream adapter is a faithful test of the fitted linear proxy,
not a claim about every PEFT projection-specific LoRA.

## Offline real-effect reachability

| Rank | Mean individual rho | Joint rho |
|---:|---:|---:|
| 1 | 0.254 | 0.023 |
| 2 | 0.435 | 0.133 |
| 4 | 0.594 | 0.210 |
| 8 | 0.749 | 0.475 |
| 16 | 0.849 | 0.574 |
| 32 | 0.941 | 0.797 |
| 64 | 0.991 | 0.928 |

The individual–joint gap is present at every tested rank and is largest in the
low-rank regime. The reachable plus unreachable components reconstruct the
original KV effect with maximum relative error below `8.1e-9`.

## Minecraft behavior

| Condition | Rank 8 | Rank 32 |
|---|---:|---:|
| Base | 1/8 | 4/8 |
| State / quality KV | **4/8** | **6/8** |
| Weight individual | 0/8 | 1/8 |
| Weight shared | 0/8 | 1/8 |
| Full state + shared weight | 0/8 | 4/8 |

Within each paired run, State remains much stronger than Weight-only. Higher
rank substantially improves hidden-effect rho, but Weight-only behavior remains
far below State. Therefore linear hidden-state reachability is not sufficient
for behavioral consolidatability; generation-path stability and where the
update is realized also matter.

Base outcomes differ between the separately executed rank-8 and rank-32 runs,
despite identical nominal seeds. Minecraft reset/world dynamics and sampling
are therefore not reproducible enough to treat the cross-rank success counts as
a paired rank comparison. Only within-run condition comparisons are valid.

The full-State + Weight condition is an upper-bound interaction control and can
double-count reachable effects. It is not the proposed exact
`unreachable-state + reachable-weight` decomposition. Its degradation at rank 8
shows that naive state/weight addition can interfere.

An additional rank-8 pilot tested the exact offline split using a shared
residual LoRA for the reachable component and a token-phase hidden-state hook
for the unreachable component. Across two new seeds (four trials per
condition): Base `1/4`, State `3/4`, Weight-shared `1/4`, decomposed `0/4`.
Thus an algebraically exact teacher-forced decomposition is not sufficient when
replayed along a freely diverging generation path. Converting the unreachable
component back into a query-responsive KV memory, rather than a fixed phase
schedule, remains a required method step.

A query-responsive hidden-memory follow-up used the current decode state to
retrieve the nearest unreachable correction instead of replaying a fixed
phase. On two further seeds: Base `0/4`, State `4/4`, Weight-shared `0/4`,
query-retrieved decomposition `0/4`. Query retrieval alone therefore does not
recover the native KV intervention. The causal interface (attention K/V versus
post-layer hidden addition) is itself part of the mechanism.

A native-KV complement follow-up kept that causal interface and selected a
single quality-prefix value scale by teacher-forced hidden MSE. Both contexts
selected `0.1875` (the full-State control uses `0.25`). On two new seeds the
paired outcomes were Base `2/4`, State `2/4`, Weight-shared `0/4`, and
Weight-shared + scaled KV complement `0/4`; mean milestone scores were `0.50`,
`0.75`, `0.25`, and `0.25`. Thus matching a hidden effect by scalar KV strength
does not preserve the behavioral rescue. The next method must learn the K/V
geometry (and likely query-dependent routing), not merely replay hidden states
or attenuate a fixed prefix.

A two-dimensional native-KV search then varied key and value scales
independently. It selected context-dependent settings: `(K=1.0,V=0.25)` for
0281 and `(K=0.5,V=0.1875)` for 0299. On held-out seeds 310/311, Base and State
were both at the ceiling (`4/4`), while Weight-shared and the selected K/V
complement were both `0/4`. This run cannot measure rescue because Base is at
ceiling, but it independently confirms that the current shared residual-weight
realization is behaviorally destructive and that teacher-forced K/V fitting is
not a sufficient selection criterion.

## Other three ideas on real effects

- KV-conditioned basis, rank 8: one shared basis `rho=0.475`; two
  latent-conditioned bases `rho=0.727`.
- Future-privileged target distance grows with horizon: `0.219` (H=2), `0.355`
  (H=4), `0.441` (H=8), confirming that future information creates a
  non-trivial student target.
- A two-item latent population retrieves its matching causal trajectory with
  posterior weight effectively 1.0. This only validates routing on the two
  training contexts, not held-out retrieval.

## Evidence files

- `real_reachability_data.json`: extraction metadata.
- `real_reachability_analysis.json`: rank sweep and individual decomposition.
- `real_reachability_conditions_rank8_pilot.json`: 40 rank-8 trials.
- `real_reachability_conditions_rank32.json`: 40 rank-32 trials.
- `real_reachability_decomposed_rank8.json`: 16 exact-split behavior trials.
- `real_reachability_retrieved_rank8.json`: 16 query-retrieved split trials.
- `causal_kv_complement_rank8.json`: teacher-forced native-KV scale search.
- `real_reachability_kv_complement_rank8.json`: 16 scaled-KV behavior trials.
- `causal_kv_geometry_rank8.json`: two-dimensional K/V scale search.
- `real_reachability_kv_geometry_rank8.json`: 16 held-out K/V geometry trials.

## MineExplorer hop generalization

A memory-free external pilot selected one 1/2/3/4-hop MineExplorer scenario and
two seeds (24 trials total):

| Method | Full success | Mean milestone score | Parser valid |
|---|---:|---:|---:|
| Base | 0/8 | 0.271 | 8/8 |
| Shared BC+DPO | 2/8 | 0.396 | 8/8 |
| GR-KTC QLoRA | 2/8 | 0.396 | 8/8 |

Both adapters improve the 2-hop crafting task from partial to full success.
They do not improve the selected 1-hop visual-location task or the 3/4-hop
tasks, where every method completes only part of the milestone chain. This is
a small stratified pilot, not evidence that reachability monotonically declines
with hop count.

Raw file: `memory_free_hop_generalization_pilot.json`.
- `real_four_ideas_analysis.json`: unified four-idea real-effect analysis.

## Four-context interference replication

A separate acquisition collected four mixed-outcome MineExplorer contexts
(`0299`, `0391`, `0499`, `0549`) with four rollouts each. At rank 8, mean
individual reachability was `0.781`, while one shared update reached only
`0.320`. Averaging over every subset of each size gives:

| Contexts N | Mean shared rho | Min–max |
|---:|---:|---:|
| 1 | 0.781 | 0.624–0.899 |
| 2 | 0.458 | 0.405–0.489 |
| 3 | 0.365 | 0.320–0.391 |
| 4 | 0.320 | 0.320–0.320 |

This is stronger evidence for context-induced consolidation interference than
the original two-context pilot. A two-basis latent-conditioned fit raises
rank-8 rho from `0.320` to `0.428`; it does not yet establish behavioral
retention. Privileged-future target distance rises from `0.206` at horizon 2
to `0.418` at horizon 8. Population retrieval identifies all four training
contexts, but held-out routing remains required.

The corresponding 32-trial behavior run (four contexts, seeds 320/321) gives
Base `6/8`, State `8/8`, Shared `3/8`, and two-basis Conditional `2/8`; all
outputs are parser-valid. Thus the conditional basis improves the offline
linear target fit but not behavioral retention. This is direct evidence that
`rho` is a diagnostic of first-order representability, not by itself a
sufficient training objective or guarantee of generation-path stability.

Evidence: `fast_kv_four_context_acquisition.json`,
`real_reachability_four_contexts.json`,
`real_reachability_four_contexts_analysis.json`, and
`real_four_ideas_four_contexts.json`. The corresponding safetensors are
intentionally git-ignored and are regenerated by the documented scripts.
Behavior evidence: `real_conditional_basis_four_contexts.json`.

## Same-model privileged-future targets

For the same four contexts, a frozen Qwen teacher receives the verified
successful future action and terminal score, while the student receives only
the current observation/task; both are teacher-forced on the same response.
The resulting non-trivial teacher–student shift has mean hidden loss `2.152`
and mean token-distribution KL `0.0140`. Rank-8 reachability is `0.291`
individually and `0.187` jointly; rank 64 reaches `0.913` individually but only
`0.442` jointly. The hindsight target is therefore even less low-rank/shared
reachable than the quality-KV correction. This establishes a real target and
diagnostic, not yet successful future-distillation training.

Evidence: `privileged_future_four_contexts.json` and
`privileged_future_four_contexts_reachability.json`; the tensor payload is
regenerated locally by `scripts/export_privileged_future_targets.py`.

The target was then used in a formal 100-step rank-32 NF4 QLoRA run with
BC+DPO, future-state loss, and anchor KL. A matched control uses the identical
four contexts, optimizer, steps, and seed schedule with future weight zero.
Memory-free evaluation gives:

| Split (8 trials each) | Base | BC+DPO control | Future QLoRA |
|---|---:|---:|---:|
| Four training contexts | 6/8 | 8/8 | 8/8 |
| Four unseen contexts | 5/8 | 6/8 | 6/8 |

All outputs are parser-valid. Future loss falls substantially during training,
but Future QLoRA exactly ties the matched control on both splits. Therefore the
current privileged target is learnable but provides no demonstrated behavioral
gain beyond BC+DPO. The result is retained as a negative ablation rather than
reported as successful hindsight distillation.

Evidence: `qlora_future_100.json`, `qlora_future_control_100.json`,
`memory_free_future_qlora_four_contexts.json`, and
`memory_free_future_qlora_heldout.json`. Adapter weights are git-ignored
(approximately 334 MiB each) and reproducibly regenerated by
`scripts/train_future_qlora.py`.

## Prompt-latent population KV behavior

A native-KV population evaluator now builds a four-item archive from the real
mixed-context trajectories. At inference it embeds the current prompt at Qwen
layer 24, forms a soft posterior over archive prompt coordinates, mixes every
layer's native K/V supports, and injects the result through attention. It uses
no task/category router.

On one held-out seed for the four archive tasks, top-1 retrieval is `4/4`, but
the posterior is shallow (typical weights `0.20–0.32`) and Base, oracle Matched,
Population, and Uniform are all `4/4` at ceiling. On four unseen tasks, Base,
Population, and Uniform all score `3/4` with mean milestone score `0.875`.
Therefore prompt-last-token cosine routing is operational but has no behavioral
advantage over a uniform population. The next defensible population method
needs a learned causal retrieval metric or harder non-ceiling contexts; tuning
temperature alone is not supported.

Evidence: `latent_population_kv_archive.json`,
`latent_population_kv_heldout.json`, and
`scripts/eval_latent_population_kv.py`.
