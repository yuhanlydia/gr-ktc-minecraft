# Cross-Context Consolidation Misalignment: Four-Context Gate

The preregistered primary metric was the mean of separate Key and Value
Grassmann distances. On all six context pairs it does **not** predict rank-8
shared-LoRA interference: Spearman `0.429`, exact two-sided permutation
`p=0.419`, pair-bootstrap 90% CI `[-0.60, 1.00]`. The primary gate fails.

The prespecified channel-separated analysis is asymmetric:

| Geometry | Spearman | Exact p | Bootstrap 90% CI |
|---|---:|---:|---:|
| Key | 0.829 | 0.0583 | [0.091, 1.000] |
| Value | 0.200 | 0.7139 | [-1.000, 0.939] |
| Joint K+V | 0.829 | 0.0583 | [0.091, 1.000] |
| Mean(K,V), primary | 0.429 | 0.4194 | [-0.600, 1.000] |

Thus the broad K/V-misalignment claim is rejected at pilot scale. A Key-only
secondary hypothesis passes the numeric gate but is based on only four
contexts and six dependent pairs. It must replicate on newly acquired raw
rollouts before Grassmann Transport or SMC is implemented.

Consensus top-2 eigenvalue means are `0.452` (K), `0.473` (V), and `0.456`
(joint). Raw evidence is `kv_subspace_misalignment_four_contexts.json`.

Next stage is fixed before new data: 8 mixed contexts, 8 rollouts/context,
layer-24 raw K/V saved before barycentric merging. The replication primary is
Key-space Grassmann distance versus pairwise rank-8 interference. The branch
advances only if Spearman is positive with permutation `p<=0.05` and a
context-bootstrap 95% interval excluding zero.
