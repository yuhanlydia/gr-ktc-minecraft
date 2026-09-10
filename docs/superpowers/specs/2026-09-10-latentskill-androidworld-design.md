# LatentSkill on AndroidWorld — Design Specification

## Goal

Test the only GR-KTC branch with a strong real causal result: **quality-localized fast latent memory**. The new question is whether a verifier-contrastive latent state learned from repeated attempts on one AndroidWorld task template transfers to fresh randomized parameter instances of the same template.

The paper-facing hypothesis is:

> A reusable GUI skill can be stored as a tiny native latent state rather than as text or a weight update, and outcome quality is localized to a small subset of model layers.

Working method name: **LatentSkill / Contrastive Latent Skill Cache (CLSC)**.

## Evidence carried forward

The design is fixed by the prior MineExplorer Gate-2 result, not tuned on AndroidWorld:

- Qwen3-VL-8B-Instruct.
- 4 latent memory tokens.
- value scale 0.25.
- quality layer = text layer 24 (zero-indexed).
- group-relative verifier advantages.
- context state at all layers, with only layer 24 replaced by the contrastive quality state.

Prior matched-context result: layer24-quality 14/16 successes vs no-memory 5/16, context-only 9/16, positive-all-layers 10/16, failed-memory 7/16.

## Benchmark

Use official AndroidWorld pinned to:

`e3fea3ccc69787570e282c99573298f1c3019a34`

This commit is the AndroidWorld main HEAD checked on 2026-09-10. AndroidWorld provides dynamically instantiated task templates, deterministic task verifiers, and seeded parameter generation. These properties make same-template / fresh-instance transfer testable without inventing a custom benchmark.

The first gate uses AndroidWorld's **T3A-style text observation** (official UI-element descriptions and official T3A action format) rather than screenshot M3A. This deliberately preserves the successful text-generation setting from the original Gate-2 and removes multimodal-RoPE as a new confound. M3A is a confirmatory extension only after the gate passes.

## Non-goals

The first gate must not add:

- prompt decommitment;
- MetaPlastic;
- LoRA/QLoRA or any weight update;
- learned retrieval;
- cross-template skill routing;
- API teacher models;
- hand-written task-specific skills;
- layer search on AndroidWorld.

The task-template name is an oracle routing key. If the method cannot win with oracle routing, stop the latent-skill direction.

## Model and resources

Primary model: local `Qwen3-VL-8B-Instruct`, BF16, frozen.

Primary hardware profile: 24 GB GPU. The successful Gate-2 BF16 path used ~16.5 GiB; therefore NF4 is not used for the scientific gate. A 16 GB quantized run is allowed only as a later precision ablation.

## Acquisition protocol

For each task template `g`, generate seeded AndroidWorld parameter instances. For each acquisition instance `j`, run `K=4` stochastic rollouts from the same exact parameters.

For rollout `k`:

1. Run the official AndroidWorld environment and verifier.
2. Use the official T3A action prompt and action schema.
3. Capture generated-token K/V at every action-selection call for all model layers.
4. Concatenate action-call K/V within the episode to obtain `Z_{g,j,k}^l`.
5. Record verifier reward `r_{g,j,k}` in `[0,1]`.

Within each matched instance group:

`A_{g,j,k} = (r_{g,j,k} - mean_j(r)) / (std_j(r) + eps)`.

Constant-outcome groups contribute zero quality advantage but may still contribute to the family context center. A family must contain at least one mixed-outcome acquisition group to create CLSC.

## LatentSkill construction

All episode trajectories are linearly resampled to `M=4` latent tokens per layer using the existing GR-KTC raw-KV merge path.

Context center:

`C_g^l = Merge({Z_{g,j,k}^l}, weights=1)`.

Positive-all baseline:

`P_g^l = Merge({Z}, weights=1[r>0])`.

Failed-memory control:

`F_g^l = Merge({Z}, weights=1[r<=0])`.

Contrastive quality state:

`Q_g^l = Merge({Z}, weights=A, negative_scale=0.5)`.

CLSC:

`S_g^l = Q_g^24` if `l=24`, otherwise `C_g^l`.

The cache context ID is `family:<task_template>`, intentionally permitting transfer across randomized instances within the same task family while forbidding cross-family use in this gate.

## Evaluation protocol

Evaluation uses fresh parameter seeds not used in acquisition. For each evaluation task instance, every method receives the same task parameters and starts from a freshly initialized AndroidWorld state.

Primary modes:

- `base`: no memory.
- `context`: family context center at all layers.
- `positive_all`: successful-trajectory KV at all layers.
- `failed`: failure-state negative control.
- `quality_all`: contrastive state at all layers.
- `clsc`: context center at all layers, quality state only at layer 24.

Evaluation is greedy (`temperature=0`) in the first gate so variation comes from randomized benchmark instances, not decoding luck.

Primary metric: AndroidWorld verifier task success rate.

Secondary metrics: parser-valid action rate, mean steps, early-stop rate, per-family paired wins/losses, and peak GPU memory.

## Phases

### Smoke

- 2 task templates.
- 1 acquisition instance/template.
- 4 rollouts/acquisition instance.
- 1 fresh evaluation instance/template.
- modes: `base`, `clsc`.
- Purpose: environment/model/parser/KV integration only.

### Pilot

- 6 task templates.
- 2 acquisition instances/template.
- 4 rollouts/instance.
- 4 fresh evaluation instances/template.
- all six modes.
- Require at least one mixed acquisition group in at least three families.

### Full Gate

- 8 task templates.
- 4 acquisition instances/template.
- 4 rollouts/instance.
- 8 fresh evaluation instances/template.
- all six modes.
- Fresh task-parameter seeds are disjoint from acquisition.

Default candidate templates:

- `ContactsAddContact`
- `SimpleCalendarAddOneEvent`
- `MarkorCreateNote`
- `SimpleSmsSend`
- `ClockTimerEntry`
- `ExpenseAddSingle`
- `RecipeAddSingleRecipe`
- `VlcCreatePlaylist`

The runner validates all names against the pinned AndroidWorld registry before execution.

## Hard GO / NO-GO

Continue the paper only if the Full Gate satisfies all of:

1. `SR_CLSC - SR_Base >= 0.05`.
2. `SR_CLSC > SR_PositiveAll`.
3. `SR_CLSC > SR_Context`.
4. CLSC has positive paired success difference in at least 2 task families.
5. At least 2 task families contain genuine mixed-outcome acquisition groups.

If this favorable oracle-routing setting does not pass, stop the entire latent-KV skill direction. Do not rescue it with retrieval, LoRA, RL, PDC, or task-specific prompt engineering.

## If the gate passes

Only then add:

1. SkillLearnBench reusable-workflow evaluation.
2. AndroidWorld M3A screenshot observation.
3. MemGUI-Bench cross-session / failure-recovery evaluation.
4. learned or embedding-based skill routing as a separate ablation.
5. text-skill and trajectory-text baselines.

These are confirmatory paper extensions, not part of the initial survival gate.
