# LatentSkill / GR-KTC Research Workspace

## Active direction: training-free latent skill acquisition

The active project is now **LatentSkill / Contrastive Latent Skill Cache (CLSC)**.

We are no longer extending Prompt Decommitment, MetaPlastic, or slow LoRA consolidation. The only branch promoted to a new benchmark is the one that already produced a strong real causal result: **quality-localized fast latent memory**.

Prior MineExplorer Gate-2, Qwen3-VL-8B BF16, 16 paired trials:

| Condition | Full success |
|---|---:|
| No memory | 5/16 |
| Context only | 9/16 |
| Positive KV at all layers | 10/16 |
| Failed KV | 7/16 |
| **Layer-24 quality memory** | **14/16** |

`layer24_quality` beat no-memory with 9 one-sided paired wins and 0 reverse wins (exact McNemar `p=0.00390625`). The new experiment asks whether that latent state is a **reusable skill**, rather than an exact-context rescue effect.

## New survival benchmark: AndroidWorld

The survival gate uses official AndroidWorld pinned to:

```text
e3fea3ccc69787570e282c99573298f1c3019a34
```

AndroidWorld is favorable but scientifically clean for this mechanism because one task template can generate many randomized parameter instances with a deterministic verifier. Acquisition and evaluation therefore share the **task family** but never the concrete task parameters.

The first gate uses official **T3A text observation/action protocol** to stay close to the successful text-generation Gate-2. AndroidWorld's official `T3A`, task lifecycle, verifier, and complexity-based step budget are reused directly. The only model-side intervention is K/V capture or injection during the T3A **action-selection** call; T3A summary calls never receive latent memory.

### Method

For repeated attempts `k` on randomized acquisition instance `j` of task family `g`:

```math
A_{g,j,k}=\frac{r_{g,j,k}-\bar r_{g,j}}{\sigma_{g,j}+\epsilon}.
```

Each episode's generated action K/V is compressed to four latent tokens. Define a family context state `C_g^l` and verifier-contrastive quality state `Q_g^l`. CLSC keeps context at every layer and replaces only the preregistered quality layer 24:

```math
S_g^l =
\begin{cases}
Q_g^{24}, & l=24,\\
C_g^l, & l\neq24.
\end{cases}
```

All choices are carried over from the earlier successful Gate-2:

```text
model          Qwen3-VL-8B-Instruct
precision      BF16
quality layer  24
memory tokens  4
value scale    0.25
negative scale 0.5
weights         frozen
routing         oracle task-template ID
```

No layer search is allowed on AndroidWorld.

## Quick start

On the GPU machine:

```bash
git pull
python -m pip install -e '.[train,test]'
bash scripts/bootstrap_androidworld.sh
```

Start AndroidWorld's Pixel 6 / API 33 emulator, then run first-time setup + smoke:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase smoke \
  --perform-emulator-setup \
  --resume

python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/smoke/summary.json
```

After first-time app setup, omit `--perform-emulator-setup`.

Then pilot:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase pilot \
  --resume

python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/pilot/summary.json
```

Only if smoke/pilot are operational, run the preregistered Full Gate:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase full \
  --resume

python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/full/summary.json \
  --output results/latentskill_androidworld/full/report.json
```

Full instructions: [`docs/LATENTSKILL_ANDROIDWORLD_EXPERIMENT.md`](docs/LATENTSKILL_ANDROIDWORLD_EXPERIMENT.md).

Fixed configuration: [`configs/latentskill_androidworld_24gb.yaml`](configs/latentskill_androidworld_24gb.yaml).

Design: [`docs/superpowers/specs/2026-09-10-latentskill-androidworld-design.md`](docs/superpowers/specs/2026-09-10-latentskill-androidworld-design.md).

## Survival gate

Continue only if the Full Gate satisfies **all**:

```text
SR_CLSC - SR_Base >= 5 percentage points
SR_CLSC > SR_PositiveAll
SR_CLSC > SR_Context
CLSC improves over Base in >= 2 task families
>= 2 task families contain real mixed-outcome acquisition groups
```

If this favorable oracle-routing setting fails, **stop the entire latent-KV skill direction**. Do not add retrieval, LoRA, RL, Prompt Decommitment, or hand-written task logic to rescue it.

If it passes, the next paper-scale benchmarks are SkillLearnBench, AndroidWorld M3A, and MemGUI-Bench.

## Default AndroidWorld families

```text
ContactsAddContact
SimpleCalendarAddOneEvent
MarkorCreateNote
SimpleSmsSend
ClockTimerEntry
ExpenseAddSingle
RecipeAddSingleRecipe
VlcCreatePlaylist
```

The runner validates these names against the pinned AndroidWorld registry before execution.

## Key files

```text
gr_ktc/latent_skill.py
    Pure CLSC memory construction and serialization.

gr_ktc/kv_prefix.py
    Existing proven K/V merge and prefix-injection primitives.

gr_ktc/generation.py
    Existing generated-K/V capture and prefix-conditioned generation.

scripts/run_androidworld_latentskill_gate.py
    Official-T3A AndroidWorld acquisition/evaluation runner.

scripts/analyze_androidworld_latentskill.py
    Paired metrics and hard GO/NO-GO decision.

scripts/bootstrap_androidworld.sh
    Pins and installs the external AndroidWorld checkout.

configs/latentskill_androidworld_24gb.yaml
    Frozen scientific settings.
```

Large K/V skill tensors under `results/**/*.safetensors` remain git-ignored. Commit scalar summaries and reports, not latent binary artifacts.

## Hardware

The primary scientific gate is **24 GB BF16**, not NF4. The earlier successful fast-KV path used approximately 16.5 GiB, so a 24 GB RTX 3090-class card is sufficient with margin. A 16 GB NF4 run is a later precision ablation only; it must not replace the main gate.

## Tests

CPU-safe active-direction tests:

```bash
PYTHONPATH=. pytest -q \
  tests/test_latent_skill.py \
  tests/test_androidworld_latentskill_protocol.py \
  tests/test_androidworld_latentskill_analysis.py
```

Full historical suite remains available with:

```bash
python -m pytest -q
```

## Archived / stopped research branches

The repository intentionally keeps historical code and results for auditability, but these are not the active research direction:

- **Prompt Decommitment:** stopped as a main line; no established benchmark advantage.
- **MetaPlastic:** structural change occurred, but dynamic structure did not beat the locked baseline.
- **Slow LoRA consolidation:** no final advantage over base in the full local stress test.
- **KV Grassmann misalignment:** rejected after independent replication.
- **State-to-weight reachability:** useful diagnostic, not a winning behavioral method.

Do not spend compute extending these branches unless a new independent result changes their status.

## Historical GR-KTC evidence

All historical MineExplorer/PEAM-compatible artifacts remain under `results/`. In particular:

- `results/STATUS.md`
- `results/fast_kv_gate2_statistics.json`
- `results/layer24_quality_control.json`
- `results/PEAM_COMPATIBLE_REPORT.md`
- `results/REAL_REACHABILITY_REPORT.md`

The old local MineExplorer/Voyager execution path used high-level Mineflayer JavaScript and is not the current AndroidWorld protocol.

## Repository ownership

`yuhanlydia/gr-ktc-minecraft` is now the writable source of truth. The old destructive automatic synchronization from `Yunbo-max/gr-ktc-minecraft` has been disabled; `.github/workflows/sync-upstream.yml` is manual-only and must not overwrite new LatentSkill work.
