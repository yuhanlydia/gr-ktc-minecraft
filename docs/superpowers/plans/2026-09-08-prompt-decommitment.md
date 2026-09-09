# Prompt Decommitment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-free, prompt-only event decommitment method and an official MineExplorer evaluation path for Qwen3-VL-8B on 16 GiB GPUs.

**Architecture:** Keep the official MineExplorer environment, visual frame history, low-level action space, milestone checker, and episode cap unchanged. Add a discrete external ledger of prior thought/action events; the model may mark old events `REVOKED`, and those labels persist in later prompts. The model is frozen and no attention/KV/weight intervention is used.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, bitsandbytes NF4, Pillow, official MineExplorer Docker sandbox.

**Spec:** `docs/superpowers/specs/2026-09-08-prompt-decommitment-design.md`

## Global Constraints

- Base model: `Qwen3-VL-8B-Instruct`.
- GPU ceiling: 15 GiB allocated on a 16 GiB card; use NF4 with bf16 compute.
- Official MineExplorer frame history remains 20 for scientific runs.
- Official MineExplorer action space, milestone checker, scene metadata, and environment commands are unchanged.
- No hidden MSR/TSR/milestone/reasoning-graph signal may enter the model prompt.
- `decommit` and `decommit_ignore` must receive the same control instruction and output schema.
- No LoRA, RL, KV capture, attention masks, or external skill retrieval in this experiment.

---

### Task 1: Prompt-control state

**Files:**
- Create: `gr_ktc/prompt_decommitment.py`
- Test: `tests/test_prompt_decommitment.py`

**Interfaces:**
- Produces `PromptMode`, `DecommitmentState`, `parse_control_response`, `build_control_instruction`, `action_signature`.

- [ ] Write tests asserting fenced JSON control parsing, persistent revocation, invalid-ID rejection, exact PDC/PDC-ignore prompt matching, and canonical low-level action signatures.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_prompt_decommitment.py`; verify RED because the module does not exist.
- [ ] Implement the minimal discrete ledger and prompt-control functions.
- [ ] Re-run the test; require all prompt-control tests to pass.
- [ ] Commit with `git commit -am "feat: add prompt decommitment state"` after staging new files.

### Task 2: Local 16GB Qwen3-VL provider

**Files:**
- Create: `gr_ktc/local_qwen_vl_provider.py`
- Test: `tests/test_local_qwen_vl_provider.py`

**Interfaces:**
- Consumes OpenAI-style MineExplorer message lists containing `image_url` base64 blocks.
- Produces `LocalQwenVLProvider.chat(...) -> str`, `set_seed(int)`, and `peak_gpu_gib()`.

- [ ] Write tests for conversion of MineExplorer image-data URLs into PIL images and deterministic seed reset.
- [ ] Run the provider tests and verify RED before the module exists.
- [ ] Implement lazy Transformers/bitsandbytes imports, NF4 loading with `{0: "15GiB", "cpu": "48GiB"}`, and local multimodal generation.
- [ ] Run provider tests; require PASS without loading a real model.
- [ ] Commit with `git commit -am "feat: add local nf4 qwen vl provider"` after staging new files.

### Task 3: Official MineExplorer runner

**Files:**
- Create: `scripts/run_mineexplorer_prompt_decommitment.py`
- Modify only through imports: official `third_party/MineExplorer`; do not patch its source.
- Test: `tests/test_prompt_decommitment_analysis.py` for stratified scene discovery.

**Interfaces:**
- CLI inputs: official MineExplorer root, Qwen model path, benchmark directory, prompt modes, per-hop count, step cap, frame size, temperature, seed.
- Output: per-scene `result.json` and aggregate `summary.json`.

- [ ] Add a failing test that constructs temporary 1/2/3/4-hop metadata and requires `--per-hop`-style stratified selection.
- [ ] Implement official-environment imports, low-level action loop, persistent decommitment state, matched random seed reset, and result logging.
- [ ] Compile with `python -m py_compile scripts/run_mineexplorer_prompt_decommitment.py`.
- [ ] Run the full unit suite.
- [ ] Commit with `git commit -am "feat: run prompt decommitment on official MineExplorer"` after staging new files.

### Task 4: Analysis and preregistered gates

**Files:**
- Create: `scripts/analyze_prompt_decommitment.py`
- Create: `configs/prompt_decommitment_16gb.yaml`
- Create: `docs/PROMPT_DECOMMITMENT_EXPERIMENT.md`
- Test: `tests/test_prompt_decommitment_analysis.py`

**Interfaces:**
- Input: runner `summary.json`.
- Output: overall/by-hop TSR/MSR plus paired `PDC-Reflection` and `PDC-PDC-ignore` statistics.

- [ ] Write failing tests for same-scene pairing and TSR/MSR aggregation.
- [ ] Implement context-paired bootstrap MSR CI and exact McNemar task-success analysis.
- [ ] Encode Gate 0, Gate 1, Gate 2 commands and GO/NO-GO rules in the experiment doc.
- [ ] Run all unit tests and compile all new Python files.
- [ ] Commit with `git commit -am "test: add prompt decommitment evaluation gates"` after staging new files.

### Task 5: GPU smoke and scientific pilot

**Files:**
- Outputs only: `results/pdc_smoke/`, then `results/pdc_pilot/`.

- [ ] Start the official Docker sandbox: `docker run -d --name mineexplorer -p 8000:8000 davidzhth/mineexplorer:0.0.1` and export `MC_SANDBOX_URL=http://localhost:8000`.
- [ ] Run one scene per hop, all four methods, 50 steps, greedy decoding.
- [ ] Verify every run emits a result file and no run exceeds the 15 GiB GPU allocation target.
- [ ] Run 3 scenes per hop, all four methods, 200 steps, greedy decoding.
- [ ] Run `scripts/analyze_prompt_decommitment.py` and apply the two preregistered directional GO criteria.
- [ ] Only if both pass, schedule the 1,800-step confirmatory experiment; otherwise stop the method without adding new modules.
