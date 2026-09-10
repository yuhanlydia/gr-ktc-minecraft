# LatentSkill / GR-KTC resource manifest

Active protocol snapshot: 2026-09-10

## Active external benchmark: AndroidWorld

- Local checkout: `third_party/android_world/` (git-ignored).
- Source: `https://github.com/google-research/android_world.git`.
- **Pinned commit:** `e3fea3ccc69787570e282c99573298f1c3019a34`.
- Bootstrap: `bash scripts/bootstrap_androidworld.sh`.
- Required runtime: Python 3.11+, Android SDK/ADB, Pixel 6 Android 13 (API 33) AndroidWorld AVD.
- The runner verifies the git HEAD before any benchmark episode.
- First-time app installation/permissions: pass `--perform-emulator-setup` once.
- Scientific protocol reuses official T3A, task initialization/teardown, verifier, and step budget.
- No model-provider API key is required; LatentSkill uses local Qwen inference.

## Active model

- Local path: `models/Qwen3-VL-8B-Instruct/` (git-ignored).
- Source: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`.
- Primary LatentSkill gate precision: **BF16**.
- Backbone is frozen; no LoRA/QLoRA training occurs in the AndroidWorld survival gate.
- Prior fast-KV BF16 execution measured approximately 16.5 GiB peak allocated memory, so use a 24 GB GPU for the primary gate.
- NF4/16 GB is not a replacement for the primary scientific gate; it is only a later precision ablation if the method survives.

## Active experiment files

```text
configs/latentskill_androidworld_24gb.yaml
docs/LATENTSKILL_ANDROIDWORLD_EXPERIMENT.md
gr_ktc/latent_skill.py
scripts/run_androidworld_latentskill_gate.py
scripts/analyze_androidworld_latentskill.py
scripts/bootstrap_androidworld.sh
```

Large K/V memory files under `results/**/*.safetensors` are intentionally git-ignored. Push scalar JSON/Markdown results only.

## Historical resources retained for auditability

### Voyager / Minecraft runtime

- `third_party/voyager/`
  - Source: `https://github.com/MineDojo/Voyager`.
  - Historical commit: `55e45a880755d0c8c66ca7fb5fe7962ac8974f89`.
- Historical local runtime: Minecraft 1.19, Java 17, Fabric, Mineflayer bridge.
- Mineflayer was moved from Voyager's 4.8.1 pin to 4.14.0 because 4.8.1 crafting hung against the local Minecraft 1.19 dedicated server.
- Reproduction helper: `scripts/bootstrap_voyager.sh`.

### MineExplorer

- Historical dataset: `data/MineExplorer-Benchmark/`.
- Dataset source: `https://huggingface.co/datasets/jometeorie/MineExplorer-Benchmark`.
- Historical local dataset contained 813 main scenarios + 100 hard scenarios.
- The official execution repository is now publicly available at `https://github.com/meituan-longcat/MineExplorer`; earlier result files that state the execution repository was unavailable reflect the repository state at the time those experiments were run and must not be retroactively relabeled as official-protocol results.

### Papers / old branches

- Historical local paper copies are not committed.
- PEAM did not have an official implementation in the old experiment and any `peam_repro/` path remains a reproduction.
- Prompt Decommitment, MetaPlastic, slow LoRA consolidation, Grassmann misalignment, and state-to-weight reachability code/results remain in the repository as archived research branches, not active extensions.

## Historical result interpretation

The result that motivates LatentSkill is `results/fast_kv_gate2_statistics.json`:

```text
No memory             5/16
Context only          9/16
Positive all layers  10/16
Failed memory         7/16
Layer-24 quality     14/16
```

This was a two-context causal mechanism gate. The new AndroidWorld experiment is specifically designed to test whether the mechanism transfers to fresh randomized instances of the same discrete task family.
