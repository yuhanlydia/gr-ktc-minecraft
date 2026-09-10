# LatentSkill / CLSC — AndroidWorld Survival Gate

## 1. Purpose

This experiment asks one narrow question:

> Does the previously successful quality-localized fast latent memory become a **reusable skill** when transferred to fresh randomized instances of the same AndroidWorld task template?

This is the final survival gate for the latent-KV skill direction. It deliberately gives the method the most favorable scientifically valid setting: task-family identity is known (oracle routing), acquisition and evaluation share the same task template, and only the template parameters change.

If CLSC does not beat Base, Context, and Positive-All under this setting, stop. Do not rescue the direction with retrieval, LoRA, RL, Prompt Decommitment, or task-specific prompts.

## 2. What is fixed from the successful MineExplorer Gate-2

The AndroidWorld gate does **not** tune these choices:

```text
model                 Qwen3-VL-8B-Instruct
precision             BF16
quality layer         24 (zero-indexed text layer)
memory tokens         4
value scale           0.25
negative scale        0.5
acquisition temp      0.9
acquisition top-p     0.95
evaluation temp       0.0 (greedy)
```

The prior causal result was:

```text
No memory             5/16 full success
Context only          9/16
Positive all layers  10/16
Failed memory         7/16
Layer-24 quality     14/16
```

The new benchmark tests whether that mechanism transfers beyond an exact matched context.

## 3. Benchmark pin

The experiment uses the official AndroidWorld repository at exactly:

```text
e3fea3ccc69787570e282c99573298f1c3019a34
```

The runner refuses a different AndroidWorld HEAD unless `--allow-androidworld-drift` is passed. Do not use that flag for paper results.

The first gate uses **official T3A text observation** rather than M3A screenshots. This is intentional: it keeps the generation interface close to the successful text-only Gate-2 and avoids introducing image/RoPE differences before the core transfer hypothesis is established.

## 4. Environment preparation

Use Python 3.11+.

From the project root:

```bash
git pull
python -m pip install -e '.[train,test]'
bash scripts/bootstrap_androidworld.sh
```

The bootstrap script:

1. clones AndroidWorld into `third_party/android_world` if needed;
2. checks out the exact pinned commit;
3. installs AndroidWorld requirements and package code.

You also need AndroidWorld's Pixel 6 / Android 13 (API 33) emulator running. The default ADB path search is:

```text
~/Android/Sdk/platform-tools/adb
~/Library/Android/sdk/platform-tools/adb
```

or pass:

```bash
--adb-path /absolute/path/to/adb
```

The model is expected at:

```text
models/Qwen3-VL-8B-Instruct
```

Override with `--model-path` if necessary.

No OpenAI/GCP/Azure key is used. All action and summary calls use the local frozen Qwen model.

## 5. Method

For a task family `g`, each acquisition instance has the same exact AndroidWorld parameters for four sampled rollouts:

\[
\{(\tau_{g,j,k}, r_{g,j,k})\}_{k=1}^4.
\]

The verifier reward is standardized **within that exact randomized instance**:

\[
A_{g,j,k}=\frac{r_{g,j,k}-\bar r_{g,j}}{\sigma_{g,j}+\epsilon}.
\]

Every action-selection call contributes generated-token K/V. Summary calls do not. Action-call trajectories are concatenated per episode and compressed to four latent memory tokens by the existing GR-KTC resampled K/V merge.

Context state:

\[
C_g^\ell=\operatorname{Merge}(Z^\ell;\mathbf 1).
\]

Contrastive quality state:

\[
Q_g^\ell=\operatorname{Merge}(Z^\ell;A,\text{negative-scale}=0.5).
\]

CLSC:

\[
S_g^\ell=
\begin{cases}
Q_g^{24}, & \ell=24,\\
C_g^\ell, & \ell\ne24.
\end{cases}
\]

The memory is registered under:

```text
family:<AndroidWorld task template>
```

so it can transfer across parameter instances within the same family but cannot be used across families in this gate.

## 6. Baselines

The full experiment evaluates:

```text
base          no latent memory
context       context center at all layers
positive_all  successful-trajectory state at all layers
failed        failure-state negative control
quality_all   verifier-contrastive state at every layer
clsc          context at all layers + quality only at layer 24
```

Why all six matter:

- `clsc > base`: latent skill has useful transfer.
- `clsc > context`: benefit is not generic task-context state.
- `clsc > positive_all`: simply replaying success K/V is not enough.
- `clsc > quality_all`: outcome quality is better localized than spread everywhere.
- `failed`: causal negative control.

## 7. Default task families

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

These names are validated against the pinned official registry before execution. You may use `--tasks` for an integration subset, but the paper Full Gate uses the preregistered eight.

## 8. Gate 0 — smoke

Purpose: verify emulator → official T3A → local Qwen → K/V capture → skill serialization → K/V injection → verifier.

Run first-time emulator/app setup with:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase smoke \
  --perform-emulator-setup \
  --resume
```

For later smoke reruns, omit `--perform-emulator-setup`.

Smoke uses:

```text
2 task families
1 acquisition instance/family
4 stochastic rollouts/instance
1 fresh evaluation instance/family
base + clsc
```

Output:

```text
results/latentskill_androidworld/smoke/
  manifest.json
  summary.json
  acquisition/<family>/...
  skills/<family>.safetensors
  skills/<family>.json
  eval/<mode>/<family>/...
```

Analyze:

```bash
python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/smoke/summary.json
```

A smoke run is an integration test, not a scientific GO claim. If a family has no mixed acquisition outcomes, it is marked `no_quality_signal`; the runner never fabricates a contrastive skill.

## 9. Gate 1 — pilot

After smoke executes end-to-end:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase pilot \
  --resume

python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/pilot/summary.json
```

Pilot uses:

```text
6 families
2 acquisition instances/family
4 rollouts/instance
4 fresh evaluation instances/family
all 6 modes
```

Before paying for Full Gate, require at least three families with genuine mixed-outcome acquisition groups. If fewer than three exist, the chosen base model/task subset is not producing enough verifier contrast to test CLSC cleanly; inspect task-family success rates rather than changing the method.

If CLSC is already directionally below Base, Context, and Positive-All on the pilot, stop early. Do not hyperparameter-sweep layer, value scale, or memory-token count on the same pilot.

## 10. Gate 2 — Full survival gate

Only after smoke/pilot are operational:

```bash
python scripts/run_androidworld_latentskill_gate.py \
  --phase full \
  --resume

python scripts/analyze_androidworld_latentskill.py \
  results/latentskill_androidworld/full/summary.json \
  --output results/latentskill_androidworld/full/report.json
```

Full uses:

```text
8 families
4 acquisition instances/family
4 rollouts/instance
8 fresh evaluation instances/family
all 6 modes
```

Acquisition and evaluation parameter seeds are deterministic and disjoint. Every evaluation mode regenerates the same randomized task instance and the runner verifies its parameter fingerprint matches across methods.

## 11. Hard GO / NO-GO

The analyzer returns GO only if **all** are true:

\[
SR_{CLSC}-SR_{Base}\ge 0.05,
\]

\[
SR_{CLSC}>SR_{PositiveAll},
\]

\[
SR_{CLSC}>SR_{Context},
\]

CLSC is positively better than Base in at least two task families, and at least two task families contain real mixed-outcome acquisition groups.

If any condition fails:

```text
NO-GO: stop the latent-KV skill direction.
```

Do not add another mechanism to rescue it.

## 12. If and only if Full Gate passes

Then expand to:

1. SkillLearnBench reusable workflows.
2. AndroidWorld M3A screenshot observation.
3. MemGUI-Bench cross-session Failure Recovery Rate.
4. Text-skill / successful-trajectory-text baselines.
5. Learned task-family routing as an ablation replacing oracle routing.

Those are paper-scale confirmatory experiments. None are prerequisites for deciding whether the core mechanism survives.

## 13. What to send back after a run

Commit the scalar results and reports, not model weights or K/V binaries:

```bash
git add \
  results/latentskill_androidworld/*/manifest.json \
  results/latentskill_androidworld/*/summary.json \
  results/latentskill_androidworld/*/report.json \
  results/latentskill_androidworld/*/report.md

git commit -m "Add AndroidWorld LatentSkill gate results"
git push
```

The `.gitignore` already excludes large `results/**/*.safetensors` artifacts.
