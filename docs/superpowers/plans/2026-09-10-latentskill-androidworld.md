# LatentSkill AndroidWorld Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate verifier-contrastive, quality-localized latent skills on fresh randomized AndroidWorld instances using the previously successful Gate-2 mechanism.

**Architecture:** Reuse existing GR-KTC raw-KV capture/injection and merge code. A pure `gr_ktc.latent_skill` module builds task-family memory bundles from episode-level K/V trajectories. A runtime-only AndroidWorld runner uses the official pinned registry, T3A prompts/actions, task initialization, verifier, and step budget while keeping AndroidWorld as an external checkout. An analyzer computes paired success deltas and the hard GO/NO-GO decision.

**Tech Stack:** Python 3.11+, PyTorch, Transformers/Qwen3-VL, safetensors, official AndroidWorld, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-latentskill-androidworld-design.md`

## Global Constraints

- AndroidWorld pin: `e3fea3ccc69787570e282c99573298f1c3019a34`.
- Model: `Qwen3-VL-8B-Instruct`, frozen BF16 for the scientific gate.
- Quality layer: 24; no AndroidWorld layer search.
- Memory tokens: 4; value scale: 0.25; contrastive negative scale: 0.5.
- Oracle task-template routing only.
- No PDC, MetaPlastic, LoRA, RL, retrieval, or API teacher in the gate.
- Evaluation parameter seeds must be disjoint from acquisition parameter seeds.
- Automatic destructive upstream sync must remain disabled.

---

### Task 1: Pure LatentSkill memory construction

**Files:**
- Create: `gr_ktc/latent_skill.py`
- Test: `tests/test_latent_skill.py`

**Interfaces:**
- Consumes: existing `group_relative_advantage`, `merge_raw_kv_trajectories`, `KVPrefixMemory`.
- Produces:
  - `concat_step_kv(step_kv: Sequence[Mapping[int, Tensor]]) -> dict[int, Tensor]`
  - `grouped_advantages(rewards: Sequence[float], group_ids: Sequence[str]) -> Tensor`
  - `build_family_memory_bundle(...) -> FamilyMemoryBundle`
  - `save_family_memory_bundle(...)`
  - `load_family_memory_bundle(...)`

- [ ] **Step 1: Write failing tests**

Tests must assert:

```python
# episode K/V concatenates step tokens in order
assert concat_step_kv([step1, step2])[0].shape[0] == step1[0].shape[0] + step2[0].shape[0]

# advantages are standardized independently per acquisition instance
adv = grouped_advantages([1, 0, 1, 1], ["a", "a", "b", "b"])
assert torch.allclose(adv[:2].mean(), torch.tensor(0.0), atol=1e-5)
assert torch.equal(adv[2:], torch.zeros(2))

# CLSC differs from context only at the fixed quality layer
bundle = build_family_memory_bundle(..., quality_layer=1, memory_tokens=2)
assert bundle.clsc.layers[0].key.equal(bundle.context.layers[0].key)
assert not bundle.clsc.layers[1].value.equal(bundle.context.layers[1].value)
assert bundle.clsc.context_id == "family:DemoTask"
```

Also round-trip the bundle through safetensors + JSON metadata.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_latent_skill.py
```

Expected: import failure because `gr_ktc.latent_skill` does not yet exist.

- [ ] **Step 3: Implement minimal pure module**

Use existing merge semantics without changing `gr_ktc/kv_prefix.py` or `gr_ktc/group_advantage.py`. Build six memories: `context`, `positive_all`, `failed`, `quality_all`, `clsc`, plus metadata. Require at least one mixed-outcome group for `quality_all/clsc`.

- [ ] **Step 4: Run tests and confirm GREEN**

```bash
PYTHONPATH=. pytest -q tests/test_latent_skill.py
```

- [ ] **Step 5: Commit**

Commit message: `Add verifier-contrastive LatentSkill memory builder`.

---

### Task 2: AndroidWorld protocol utilities and phase definitions

**Files:**
- Create: `scripts/run_androidworld_latentskill_gate.py`
- Test: `tests/test_androidworld_latentskill_protocol.py`

**Interfaces:**
- Produces pure helpers importable without AndroidWorld installed:
  - `PhaseSpec`
  - `phase_spec(name: str) -> PhaseSpec`
  - `default_task_names() -> tuple[str, ...]`
  - `instance_seed(split: str, family: str, index: int, base_seed: int) -> int`
  - `validate_seed_partitions(...)`
- Runtime AndroidWorld imports must occur only after `--androidworld-root` is installed into `sys.path`.

- [ ] **Step 1: Write failing protocol tests**

Tests must verify:

```python
assert phase_spec("smoke").train_instances == 1
assert phase_spec("pilot").rollouts_per_instance == 4
assert phase_spec("full").test_instances == 8
assert set(default_task_names()) == {
    "ContactsAddContact", "SimpleCalendarAddOneEvent", "MarkorCreateNote",
    "SimpleSmsSend", "ClockTimerEntry", "ExpenseAddSingle",
    "RecipeAddSingleRecipe", "VlcCreatePlaylist",
}
assert acquisition_seeds.isdisjoint(evaluation_seeds)
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_protocol.py
```

- [ ] **Step 3: Implement runner protocol shell**

The script must:

1. Verify the AndroidWorld git HEAD equals the pinned commit unless `--allow-androidworld-drift` is passed.
2. Load official `TaskRegistry`, `suite_utils`, `env_launcher`, T3A prompt utilities, JSONAction parser, and base agent only at runtime.
3. Load local Qwen3-VL-8B BF16 with `load_qwen3_vl_24gb`.
4. Support `--phase smoke|pilot|full`, `--tasks`, `--seed`, `--resume`, `--perform-emulator-setup`, `--adb-path`, `--console-port`, `--output-dir`.
5. Save a manifest before the first rollout containing model path, AndroidWorld pin, phase, task list, acquisition/test seeds, memory hyperparameters, and mode list.

- [ ] **Step 4: Run protocol tests**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_protocol.py
```

- [ ] **Step 5: Commit**

Commit message: `Add pinned AndroidWorld LatentSkill protocol`.

---

### Task 3: T3A-style local Qwen action agent with K/V capture and injection

**Files:**
- Modify: `scripts/run_androidworld_latentskill_gate.py`
- Test: `tests/test_androidworld_latentskill_protocol.py`

**Interfaces:**
- Runtime class `LatentSkillT3AAgent(EnvironmentInteractingAgent)`.
- Acquisition mode captures action-selection generated K/V for all text layers.
- Evaluation mode injects one selected family memory into every action-selection call.
- Summarization calls never receive latent memory and never contribute acquisition K/V.

- [ ] **Step 1: Add failing tests for model-call routing**

Use a tiny fake generator object to assert:

```python
# acquisition requests capture=True and memory=None
# evaluation requests capture=False and memory=<selected bundle memory>
# summary calls always memory=None and capture=False
```

The pure routing helper must be testable without AndroidWorld or Transformers.

- [ ] **Step 2: Run RED test**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_protocol.py
```

- [ ] **Step 3: Implement agent**

Action selection must use official T3A `_action_selection_prompt` and official action schema. For acquisition, call `generate_with_final_kv`; for evaluation, call `generate_with_kv_prefix`. Decode `Reason/Action`, create official `JSONAction`, execute via `env.execute_action`, then use official T3A `_summarize_prompt` for history. Episode K/V is the per-layer concatenation of action-selection generations only.

- [ ] **Step 4: Run GREEN tests**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_protocol.py
```

- [ ] **Step 5: Commit**

Commit message: `Add T3A-style latent-skill action agent`.

---

### Task 4: Acquisition, family skill construction, and paired evaluation

**Files:**
- Modify: `scripts/run_androidworld_latentskill_gate.py`
- Create: `configs/latentskill_androidworld_24gb.yaml`

**Interfaces:**
- Acquisition records per family/instance/rollout.
- Saved skill bundles under `<output>/skills/<family>.safetensors` with JSON metadata.
- Evaluation results under `<output>/eval/<mode>/<family>/<instance>.json`.
- Summary at `<output>/summary.json`.

- [ ] **Step 1: Implement acquisition loop**

For each family and seeded acquisition instance:

- generate exact params once;
- run 4 stochastic rollouts with the same params;
- reset/initialize/tear-down task for every rollout;
- record verifier reward and episode K/V;
- compute within-instance group-relative advantages;
- persist the raw scalar result immediately for resume safety.

- [ ] **Step 2: Implement memory construction**

Construct `context`, `positive_all`, `failed`, `quality_all`, and `clsc` using `gr_ktc.latent_skill`. Skip CLSC for families with no mixed acquisition group and mark `family_status="no_quality_signal"` rather than fabricating a memory.

- [ ] **Step 3: Implement matched evaluation**

Generate fresh test params from disjoint deterministic seeds. For every test instance, reinitialize the identical params separately for every mode. Use greedy decoding. The six modes are:

```text
base context positive_all failed quality_all clsc
```

- [ ] **Step 4: Add 24 GB config**

The YAML must pin:

```yaml
model: Qwen3-VL-8B-Instruct
precision: bf16
quality_layer: 24
memory_tokens: 4
value_scale: 0.25
negative_scale: 0.5
acquisition_temperature: 0.9
acquisition_top_p: 0.95
evaluation_temperature: 0.0
```

and list smoke/pilot/full counts and default tasks.

- [ ] **Step 5: Static verification**

```bash
python -m compileall -q gr_ktc scripts tests
```

- [ ] **Step 6: Commit**

Commit message: `Implement AndroidWorld LatentSkill acquisition and evaluation`.

---

### Task 5: Analyzer and hard stop gate

**Files:**
- Create: `scripts/analyze_androidworld_latentskill.py`
- Test: `tests/test_androidworld_latentskill_analysis.py`

**Interfaces:**
- `analyze(summary: dict) -> dict`
- `hard_gate(report: dict) -> dict`

- [ ] **Step 1: Write failing analysis tests**

Synthetic paired data must verify task success rate, paired win/loss counts, per-family deltas, and the hard gate. The gate must require:

```python
clsc_minus_base >= 0.05
clsc_success_rate > positive_all_success_rate
clsc_success_rate > context_success_rate
positive_family_count >= 2
mixed_signal_family_count >= 2
```

- [ ] **Step 2: Run RED test**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_analysis.py
```

- [ ] **Step 3: Implement analyzer**

Output `report.json` and a compact Markdown table with Base / Context / Positive-All / Failed / Quality-All / CLSC success rates, parser-valid rates, mean steps, paired CLSC-vs-baseline counts, and GO/NO-GO reasons.

- [ ] **Step 4: Run GREEN test**

```bash
PYTHONPATH=. pytest -q tests/test_androidworld_latentskill_analysis.py
```

- [ ] **Step 5: Commit**

Commit message: `Add LatentSkill AndroidWorld hard-gate analysis`.

---

### Task 6: Runbook, README, bootstrap helper, and CI

**Files:**
- Create: `docs/LATENTSKILL_ANDROIDWORLD_EXPERIMENT.md`
- Create: `scripts/bootstrap_androidworld.sh`
- Create: `.github/workflows/latentskill-ci.yml`
- Modify: `README.md`

**Interfaces:**
- One bootstrap command for the pinned external benchmark.
- One smoke command, one pilot command, one full-gate command, one analysis command.

- [ ] **Step 1: Add bootstrap helper**

The script must clone AndroidWorld if absent, fetch the pinned commit, checkout exactly that commit, install AndroidWorld requirements, and print emulator setup commands. It must not modify AndroidWorld source files.

- [ ] **Step 2: Add experiment runbook**

The document must explain environment setup, model path, expected directory structure, commands, output files, resume behavior, resource estimates, and the hard stop rule.

- [ ] **Step 3: Update README**

Move Prompt Decommitment to `Archived / negative-unverified branches`. Make LatentSkill the active direction and cite the prior 14/16 vs 5/16 Gate-2 evidence. State clearly that AndroidWorld transfer is not yet proven until the new gate runs.

- [ ] **Step 4: Add CI**

CI runs compileall and only the CPU-safe LatentSkill unit tests on pushes/PRs. It must not download AndroidWorld or a model.

- [ ] **Step 5: Full repository verification**

```bash
PYTHONPATH=. pytest -q \
  tests/test_latent_skill.py \
  tests/test_androidworld_latentskill_protocol.py \
  tests/test_androidworld_latentskill_analysis.py
python -m compileall -q gr_ktc scripts tests
```

- [ ] **Step 6: Remote file verification**

Confirm `main` contains all new source/tests/docs/configs and that the destructive automatic sync workflow is disabled.

- [ ] **Step 7: Commit**

Commit message: `Promote LatentSkill AndroidWorld survival gate`.
