# Four new ideas: API-free mechanism runner

This extension follows the State–Weight Reachability proposal rather than
continuing to tune merge A/B/C.

1. **State–Weight Reachability** fits a rank-`r` first-order update to a useful
   state correction and reports `rho`. The fitted component is the slow,
   parameter-reachable part; the residual is the fast KV part. It reports both
   rank scaling and context heterogeneity (`N=1,2,4,8,16,32`).
2. **KV-Conditioned LoRA Basis** learns continuous latent-coordinate weights
   over low-rank basis updates. Coordinates, not task IDs or categories, drive
   the mixture.
3. **Privileged-Future Latent Distillation** provides KL and hidden-state
   losses plus discounted backward targets from successful future states.
4. **Latent Population** keeps a bounded archive with score-based selection,
   mutation, and distance-weighted posterior retrieval.

The runner uses all 813 records in `data/MineExplorer-Benchmark/benchmark.jsonl`
and deliberately does not require Minecraft login, Azure, or an external API.
It currently generates deterministic activation-shaped tensors to test the
mathematics and data plumbing. These results are not causal model evidence;
real KV/hidden tensors from the VLM must replace `_make_contexts` before making
behavioral claims.

## Run

```bash
PYTHONPATH=. .venv/bin/python scripts/run_new_ideas.py \
  --scenarios data/MineExplorer-Benchmark/benchmark.jsonl \
  --output results/four_new_ideas.json

PYTHONPATH=. .venv/bin/python scripts/run_four_ideas_sweep.py \
  --output results/four_ideas_sweep.json
```

## Cross-context Grassmann hypothesis (follow-up)

A four-context merged-support pilot suggested Key-subspace Grassmann distance
might predict pairwise shared-LoRA interference (`Spearman=0.829`, exact
`p=0.058`). The preregistered combined K/V metric failed, so Key was treated as
a secondary hypothesis requiring independent replication.

That replication collected 8 new rollouts for each of 8 mixed contexts and
saved raw layer-24 K/V before merging. Across all 28 pairs, the fixed Key
primary falls to `Spearman=0.159`, permutation `p=0.417`, with context-bootstrap
95% CI `[-0.714,0.833]`. V, joint, and averaged geometry also fail. Therefore
the repository stops the Grassmann Transport/SMC branch rather than tuning
subspace rank or distance after seeing the result. See
`results/CROSS_CONTEXT_MISALIGNMENT_REPLICATION.md`.

The 24GB model settings are in `configs/four_ideas_24gb.yaml`. Analysis is
offloaded to CPU and uses one BLAS thread to make hundreds of small SVDs fast;
this does not change Qwen inference or QLoRA settings.

## Current mechanism results

The full sweep completed over 813/813 scenarios (seed 42):

- Individual rank-8 reachability: `rho=1.000`; shared rank-8 reachability:
  `rho=0.809`.
- Shared rank curve: `0.189, 0.322, 0.542, 0.809, 0.9996, 0.9996,
  0.9996` for ranks `1,2,4,8,16,32,64`.
- Context heterogeneity shared `rho`: `1.000, 0.904, 0.823, 0.806, 0.808,
  0.809` for `N=1,2,4,8,16,32`; individual `rho` stays approximately `1`.
- Future-state loss falls from `0.0449` to `0.0112` at horizon 4 in the
  deterministic student/teacher check.
- Conditional basis coefficient entropy rises from `0` (one basis) to `0.673`
  (eight bases), confirming continuous rather than hard task routing.
- Population archives retain exactly the requested sizes 8/16/32/64 and return
  an 8-token posterior mixture.

See the raw machine-readable output at
`results/four_new_ideas.json` and `results/four_ideas_sweep.json`.

## Real-Qwen follow-up

The same APIs have now been run on two real layer-24 quality-KV causal effects.
The rank-8 individual/shared reachability values are `0.749/0.475`; at rank 32
they are `0.941/0.797`. Eighty paired Minecraft behavior trials show State at
`4/8` and `6/8` versus Weight-only at `0/8` and `1/8` for ranks 8 and 32,
respectively. Read `results/REAL_REACHABILITY_REPORT.md` for the necessary
cross-run and method-scope caveats.

The follow-up also tests scheduled and query-retrieved unreachable hidden
memory. Neither reproduces the native quality-KV rescue, which narrows the next
method step to preserving the attention K/V causal interface. A further
native-KV experiment selected value scale `0.1875` on both contexts by
teacher-forced hidden MSE, but its Weight + KV complement condition scored
`0/4` versus State `2/4`. Scalar attenuation is therefore insufficient; K/V
geometry must be learned. A 24-trial
memory-free 1–4-hop MineExplorer pilot is stored in
`results/memory_free_hop_generalization_pilot.json`.

The next 2D key/value search found different settings for the two contexts,
but held-out behavior remained `0/4` for Weight + K/V versus `4/4` for both
Base and State on a ceiling seed pair. This rules out reporting the hidden-MSE
gain as behavioral consolidation evidence.

A new configurable acquisition run found four mixed real contexts. Its rank-8
all-subset heterogeneity curve is `0.781, 0.458, 0.365, 0.320` for
`N=1,2,3,4`, respectively. Two latent-conditioned LoRA bases improve the
four-context fit from `rho=0.320` to `0.428`. This is still a residual-stream
linear proxy, but it independently replicates the individual-versus-shared
gap on twice as many causal contexts.

In paired behavior, however, the same two-basis model scores `2/8`, versus
Shared `3/8`, Base `6/8`, and native State `8/8`. The repository therefore
does not claim that conditional bases solve consolidation; their offline gain
currently fails the generate-path validity gate.

Privileged-future targets now use a real frozen same-model teacher: it sees the
verified successful future action and terminal score, while the student sees
only the current state. Across four contexts, rank-8 individual/shared
reachability is `0.291/0.187` (`0.913/0.442` at rank 64). This replaces the
earlier time-smoothed proxy with an actual information-asymmetric VLM target;
adapter training and held-out behavior remain open.

That training is now implemented with rank-32 NF4 QLoRA for 100 steps. Against
an otherwise identical four-context BC+DPO control, Future QLoRA ties `8/8` on
training contexts and `6/8` on four unseen contexts (Base `6/8` and `5/8`).
The future objective is therefore operational and memory-free, but currently
adds no behavioral benefit over the control.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_future_qlora.py --steps 100
PYTHONPATH=. .venv/bin/python scripts/train_future_qlora.py --steps 100 \
  --future-weight 0 --adapter-output results/qlora_future_control_100 \
  --report results/qlora_future_control_100.json
```

Latent Population now also has a real native-KV behavior path. The current
prompt's layer-24 latent retrieves and softly mixes four all-layer K/V
trajectories. Archive top-1 retrieval is `4/4`, but the posterior is shallow
and all archive conditions are at a `4/4` ceiling. On four unseen tasks,
Population, Uniform, and Base all score `3/4` (mean MSR `0.875`). This validates
the runtime interface but rejects untrained cosine routing as the method.

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_latent_population_kv.py \
  --seeds 340 --output results/latent_population_kv_archive.json
PYTHONPATH=. .venv/bin/python scripts/eval_latent_population_kv.py \
  --scene-ids 0000 0034 0044 0065 --seeds 342 \
  --output results/latent_population_kv_heldout.json
```
