# Prompt Decommitment for MineExplorer — Design Spec

## Research question

Can a frozen small MLLM improve long-horizon MineExplorer performance by explicitly **revoking its own stale decision commitments in the prompt**, without modifying attention, KV caches, or model weights?

The target failure is the **Correction–Suppression Gap**:

\[
P(a_{new}\mid\text{feedback})\uparrow
\quad\not\Rightarrow\quad
P(a_{old}\mid\text{feedback})\downarrow.
\]

A model may correctly reason that a plan is obsolete while still being behaviorally biased by old thoughts/actions that remain in its history.

## Method: Prompt Decommitment (PDC)

Each previous decision is a discrete event

\[
E_i=(r_i,a_i),
\]

where `r_i` is the model's thought/plan and `a_i` is the executed MineExplorer low-level action. Visual observations remain untouched.

At decision step `t`, the frozen model emits:

```json
{
  "revoke_event_ids": ["E3", "E7"],
  "current_belief": "...",
  "thought": "...",
  "action": {...},
  "memory_update": "..."
}
```

Revocation state is persistent:

\[
z_{t,i}=z_{t-1,i}\lor \hat z_{t,i},\qquad z_{t,i}\in\{0,1\}.
\]

A revoked event stays visible as historical evidence but is explicitly rendered as `[REVOKED]`; the prompt states that only the **old thought/action commitment** is invalid. The associated image remains valid evidence, and the same low-level action may be chosen again later under a new state.

The model then writes a `current_belief` using the latest visual observations plus non-revoked commitments and chooses the next action from that belief.

No benchmark milestone, MSR, TSR, reasoning graph, or hidden verifier signal is exposed to the model.

## Four conditions

1. `base` — official MineExplorer prompt behavior.
2. `reflection` — official prompt plus a generic instruction to reconsider stale/contradicted plans; no discrete revocation state.
3. `decommit_ignore` — **exact same PDC prompt and five-key JSON schema** as the method, but the runtime ignores all requested revocations. This is the critical prompt-length/schema control.
4. `decommit` — full persistent event revocation.

The causal mechanism test is therefore:

\[
\text{decommit} > \text{decommit\_ignore}
\]

rather than merely `decommit > base`.

## Benchmark invariants

The official MineExplorer execution path is retained:

- official Minecraft sandbox;
- official 128×128 POV observations;
- official 20-frame history;
- official low-level `MinerRLActionSpace`;
- official milestone checker;
- official scene metadata and environment commands;
- official 1,800-step cap for confirmatory evaluation.

Only the prompt protocol changes.

## GPU/runtime constraints

Primary deployment target is one 16 GiB GPU:

- Qwen3-VL-8B-Instruct;
- NF4 double quantization, bf16 compute;
- CUDA max-memory budget 15 GiB;
- batch size 1 for closed-loop visual interaction;
- no gradients;
- no KV capture;
- same frame size and model on 24 GiB cards so extra memory only increases throughput, not scientific capacity.

## Main prediction

If the mechanism is correct, PDC should help more as hidden prerequisite depth grows:

\[
\Delta TSR_{4-hop}\ge \Delta TSR_{3-hop} > \Delta TSR_{1-hop}.
\]

A method that mainly improves 1-hop but not 3/4-hop does not support the stale-commitment explanation.

## Stop rules

The direction is a NO-GO if either of the following holds after the pilot:

- `decommit` does not improve paired MSR over `reflection`; or
- `decommit` does not improve paired MSR over `decommit_ignore`.

If the method only beats `base` but not `decommit_ignore`, the gain is attributed to prompt format/extra reasoning rather than persistent revocation.
