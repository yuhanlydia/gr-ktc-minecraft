# Four-Idea Experiment Completion Audit

Date: 2026-08-27. Hardware: one RTX 3090 24 GiB. No external model API.

## Requirement evidence

| Requirement | Implementation | Authoritative evidence | Status |
|---|---|---|---|
| State–Weight Reachability | `gr_ktc/reachability.py`, real exporter/analyzer | `real_reachability_four_contexts_analysis.json`: ranks 1–64; all subsets N=1–4 | Complete |
| Individual vs shared | closed-form individual/shared fits | rank-8 `0.781` vs `0.320` on four causal contexts | Complete |
| Rank scaling | ranks 1,2,4,8,16,32,64 | same analysis JSON | Complete |
| Context heterogeneity | every subset for N=1,2,3,4 | shared rho `0.781,0.458,0.365,0.320` | Complete |
| Four causal conditions | Base, native State, Weight, combined/decomposed controls | `real_reachability_conditions_rank8_pilot.json`, rank-32, exact/retrieved/KV-complement files | Complete; decomposition fails behaviorally |
| KV-Conditioned LoRA Basis | continuous latent-coordinate basis and multi-hook realization | `real_four_ideas_four_contexts.json`, `real_conditional_basis_four_contexts.json` | Complete; offline gain, behavior regression |
| Privileged-Future Distillation | same-model future-informed teacher; formal rank-32 NF4 QLoRA | `privileged_future_four_contexts*.json`, `qlora_future_100.json` | Complete |
| Fair future control | identical BC+DPO run with future weight zero | `qlora_future_control_100.json` | Complete |
| Future memory-free behavior | no future or KV at inference | training `8/8` vs control `8/8`; unseen `6/8` vs `6/8` | Complete; no incremental gain |
| Latent Population Memory | prompt-layer query, soft all-layer native-KV population mixture | `eval_latent_population_kv.py`, archive and held-out JSON | Complete; no gain over uniform |
| MineExplorer external/hop | 1/2/3/4-hop stratified and four unseen tasks | `memory_free_hop_generalization_pilot.json`, future/population held-out files | Complete at pilot scale |
| 24 GiB staging | BF16 inference, NF4 QLoRA, CPU SVD, staged scripts/config | `configs/four_ideas_24gb.yaml`; measured peaks 6.8–16.9 GiB | Complete |
| Reproducibility | CLI inputs/outputs, immutable JSON reports, tests, GitHub history | scripts, config, reports, `pytest` | Complete |

## Scientific outcome

The strongest positive result is the consolidation-interference finding:
useful native-KV corrections are much more reachable individually than by one
shared low-rank update, and joint reachability decreases monotonically over all
context subsets. Native State remains the strongest causal interface.

The three proposed solutions do not yet beat their strict controls:

- conditional bases improve offline rho but reduce behavioral success;
- privileged-future QLoRA learns its hidden target but ties BC+DPO;
- prompt-cosine KV population retrieval is correct on archive IDs but ties a
  uniform mixture and Base on held-out behavior.

These are completed negative validations, not missing experiments. They bound
the paper claim and identify learned generation-aware reachability/routing as
future work.

## Reproduction boundary

Large `.safetensors` payloads and 334 MiB adapters are intentionally ignored by
Git. Their metadata/results and producing commands are versioned. Minecraft
server and Mineflayer are local; Qwen3-VL-8B runs without GPT/Azure credentials.
