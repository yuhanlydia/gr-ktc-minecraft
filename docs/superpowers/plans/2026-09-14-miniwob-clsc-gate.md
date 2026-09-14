# MiniWoB++ CLSC Fast Reliability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute a reproducible MiniWoB++ reliability gate for training-free CLSC transfer using native raw rewards.

**Architecture:** A pure `gr_ktc.miniwob_protocol` module owns reward, signal-selection, seed, and action-parsing invariants. A runtime script lazily loads BrowserGym and Qwen, captures action-token K/V during K=10 acquisition, builds existing family-memory bundles with an explicit terminal-success mask, and evaluates four paired modes on fresh seeds. A separate analyzer produces the stop/continue decision from JSON artifacts.

**Tech Stack:** Python 3.11, PyTorch, Transformers/Qwen3-VL, BrowserGym, MiniWoB++, Playwright, pytest, safetensors.

**Spec:** `docs/superpowers/specs/2026-09-14-miniwob-clsc-gate-design.md`

## Global Constraints

- BrowserGym pin: `9e779f087de9a65668b6974d11f9ce9816026e96`.
- MiniWoB++ pin: `7fd85d71a4b60325c6585396ec4f48377d049838`.
- Quick acquisition uses one exact instance and K=10 rollouts per family.
- Qualification requires at least 3 successes, 3 failures, and reward standard deviation at least 0.15.
- Success is `RAW_REWARD_GLOBAL >= 0.5`; quality reward is `clip(raw_reward, 0, 1)`.
- Quick evaluation uses 8 fresh seeds and `base`, `context`, `positive_all`, `clsc`.
- Qwen is frozen BF16; quality layer 24; memory tokens 4; value scale 0.25; negative scale 0.5.
- No OpenAI or other API model is used.

---

### Task 1: Pure MiniWoB protocol invariants

**Files:**
- Create: `gr_ktc/miniwob_protocol.py`
- Create: `tests/test_miniwob_protocol.py`

**Interfaces:**
- Produces: `MiniwobPhase`, `phase_spec(name)`, `quality_reward(raw)`, `is_terminal_success(raw)`, `qualify_signal(rewards)`, `protocol_seed(namespace, family, index, base_seed)`, `extract_single_action(text)`.

- [ ] **Step 1: Write failing tests** for finite raw rewards, 0.5 threshold, clipping, 3/3 qualification, low-variance rejection, deterministic namespace-separated seeds, and safe extraction of one allowed function call.
- [ ] **Step 2: Run** `../../.venv/bin/python -m pytest -q tests/test_miniwob_protocol.py` and confirm import failure.
- [ ] **Step 3: Implement the pure module** with dataclasses and no BrowserGym import. Parse actions with `ast.parse(..., mode="eval")`; accept only allowed call names, literal positional/keyword arguments, and no attribute access or extra statements.
- [ ] **Step 4: Re-run the focused tests** and confirm all pass.
- [ ] **Step 5: Commit** with message `Add MiniWoB reliability protocol primitives`.

### Task 2: Separate quality rewards from terminal success masks

**Files:**
- Modify: `gr_ktc/latent_skill.py`
- Modify: `tests/test_latent_skill.py`

**Interfaces:**
- Extends: `build_family_memory_bundle(..., successes: Sequence[bool] | None = None)`.
- Preserves: omitted `successes` uses the existing `reward > 0` behavior.

- [ ] **Step 1: Add a failing test** where rewards `[0.8, 0.4]` and successes `[True, False]` produce a valid bundle, while the explicit-mask length mismatch raises `ValueError`.
- [ ] **Step 2: Run the focused test** and confirm the missing keyword causes failure.
- [ ] **Step 3: Implement the optional mask**, validate length, derive Positive-All/Failed from it, and keep grouped advantages based on continuous rewards.
- [ ] **Step 4: Run** `../../.venv/bin/python -m pytest -q tests/test_latent_skill.py tests/test_miniwob_protocol.py`.
- [ ] **Step 5: Commit** with message `Support explicit success masks in latent skills`.

### Task 3: Qwen MiniWoB action policy and episode adapter

**Files:**
- Create: `scripts/run_miniwob_latentskill_gate.py`
- Create: `tests/test_miniwob_runner.py`

**Interfaces:**
- Produces: `build_action_prompt(goal, axtree, history, action_error)`, `MiniwobQwenPolicy.generate_action(...)`, `run_episode(env_factory, ...)`.
- Consumes: `generate_with_final_kv`, `generate_with_kv_prefix`, `concat_step_kv`, and protocol helpers.

- [ ] **Step 1: Write failing tests** using a small deterministic fake environment and fake policy. Assert raw reward is read from `info["RAW_REWARD_GLOBAL"]`, wrapper reward is ignored, invalid raw reward fails closed, action errors are recorded, acquisition returns concatenated K/V, and injected modes pass the expected memory to policy generation.
- [ ] **Step 2: Run the focused tests** and confirm the runner module is absent.
- [ ] **Step 3: Implement prompt construction and the episode loop**. Flatten accessibility trees through a dependency passed by the lazy runtime loader. Limit history to the latest four actions, stop on terminal/truncated/max steps, and record goal fingerprint, parser-valid rate, raw/acquisition rewards, success, reward reason, steps, and elapsed time.
- [ ] **Step 4: Implement the real Qwen policy** with the existing chat template and KV generation functions. Acquisition captures every generated action call; evaluation injects only the selected memory. Decode only generated tokens and use greedy generation for evaluation.
- [ ] **Step 5: Run the focused runner/protocol tests**.
- [ ] **Step 6: Commit** with message `Add MiniWoB Qwen episode adapter`.

### Task 4: Signal scan, skill construction, paired evaluation, and resume

**Files:**
- Modify: `scripts/run_miniwob_latentskill_gate.py`
- Modify: `tests/test_miniwob_runner.py`
- Create: `configs/latentskill_miniwob_24gb.yaml`

**Interfaces:**
- Produces CLI `--phase {smoke,quick}`, `--tasks`, `--target-families`, `--resume`, benchmark/model/output paths, and token/step limits.
- Writes the artifact layout specified in the design.

- [ ] **Step 1: Add failing orchestration tests** proving K=10 for quick, fixed seed manifests, stop after two qualified families, skipped-family retention, exact acquisition fingerprint equality, paired evaluation fingerprints/model seeds, and resume requiring both JSON and K/V.
- [ ] **Step 2: Run the focused tests** and confirm failures describe missing orchestration.
- [ ] **Step 3: Implement manifest and artifact helpers**, acquisition scan, explicit-success skill construction, four-mode evaluation, incremental summary writes, and fail-closed integrity checks.
- [ ] **Step 4: Add the YAML protocol** with the exact constants and task order from the spec.
- [ ] **Step 5: Add a YAML/executable consistency test** and run runner/protocol/latent-skill tests.
- [ ] **Step 6: Commit** with message `Implement MiniWoB CLSC quick gate`.

### Task 5: Analyzer and reliability decision

**Files:**
- Create: `scripts/analyze_miniwob_latentskill.py`
- Create: `tests/test_miniwob_analysis.py`

**Interfaces:**
- Produces: `analyze(summary) -> dict`, CLI input summary and optional output path.
- Writes: `report.json` and `report.md` with per-mode rates and paired comparisons.

- [ ] **Step 1: Write failing synthetic tests** for positive gate, baseline tie/failure, insufficient qualifying families, incomplete paired modes, and a 5 percentage-point aggregate threshold.
- [ ] **Step 2: Run the focused tests** and confirm analyzer import failure.
- [ ] **Step 3: Implement paired integrity checks, metrics, decisions, and Markdown rendering** without importing BrowserGym or loading tensors.
- [ ] **Step 4: Run analyzer and all MiniWoB-focused tests**.
- [ ] **Step 5: Commit** with message `Add MiniWoB CLSC gate analysis`.

### Task 6: Reproducible bootstrap and user documentation

**Files:**
- Create: `scripts/bootstrap_miniwob.sh`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `docs/LATENTSKILL_MINIWOB_EXPERIMENT.md`
- Modify: `tests/test_project_metadata.py`

**Interfaces:**
- Produces: idempotent benchmark setup and documented smoke/quick/analyze commands.

- [ ] **Step 1: Add failing metadata tests** for the two pins, executable bootstrap, optional dependency declaration, documented K=10 and raw-reward threshold, and CLI help importability without BrowserGym.
- [ ] **Step 2: Run metadata tests** and confirm the missing files/metadata fail.
- [ ] **Step 3: Implement bootstrap** that clones or verifies exact pins, installs BrowserGym core/miniwob packages, installs Playwright Chromium, and prints `MINIWOB_URL`.
- [ ] **Step 4: Add optional dependencies and documentation**. State that this is a favorable reliability gate and that MiniWoB wrapper binary reward is not used.
- [ ] **Step 5: Run shell syntax, CLI help, dependency consistency, and focused tests**.
- [ ] **Step 6: Commit** with message `Document and bootstrap MiniWoB CLSC gate`.

### Task 7: Real browser and Qwen smoke, final verification, integration

**Files:**
- Generate ignored artifacts under: `results/latentskill_miniwob/smoke/`
- Modify only if a smoke failure has a test-backed implementation fix.

**Interfaces:**
- Consumes the installed benchmark, Chromium, local frozen Qwen model, and smoke CLI.
- Produces a real seeded result JSON plus captured K/V tensor and incremental summary.

- [ ] **Step 1: Run bootstrap** and verify both checkout HEADs and Chromium launch.
- [ ] **Step 2: Stop the old AndroidWorld GPU process cleanly** after preserving its JSON checkpoint, because both Qwen runners cannot fit on one 24 GB GPU.
- [ ] **Step 3: Run a real smoke** with one short supported task and bounded step/token settings; inspect goal, action, raw reward, result JSON, and non-empty K/V tensor.
- [ ] **Step 4: For every discovered bug, write a failing regression test before changing production code**, then re-run the smoke.
- [ ] **Step 5: Run full relevant verification**: all runnable repository tests, MiniWoB tests, Python compilation, shell syntax, CLI help, config consistency, `git diff --check`, and artifact assertions.
- [ ] **Step 6: Merge the feature branch into `main`, push GitHub, and start the K=10 quick gate with long-poll-safe logging**.
- [ ] **Step 7: Commit runtime smoke metadata/results that are small and stable; keep K/V tensors and live logs ignored**.
