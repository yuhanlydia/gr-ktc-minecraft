# GR-KTC local resource manifest

Snapshot date: 2026-08-26

## Downloaded

- `third_party/voyager/`
  - Source: https://github.com/MineDojo/Voyager
  - Commit: `55e45a880755d0c8c66ca7fb5fe7962ac8974f89`
  - Status: official code cloned; Python and pinned Node dependencies installed.
- `models/Qwen3-VL-8B-Instruct/`
  - Source: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
  - Status: complete original checkpoint, four safetensors shards, about 17 GB.
- `data/MineExplorer-Benchmark/`
  - Source: https://huggingface.co/datasets/jometeorie/MineExplorer-Benchmark
  - Status: 813 main scenarios, 100 hard scenarios, dataset card and demo videos.
- `docs/PEAM_2605.27762v2.pdf`
- `docs/MineExplorer_2605.30931v2.pdf`

## Unavailable upstream

- PEAM: no official public code repository was found. Any implementation under
  `peam_repro/` must be labeled as a reproduction, not official PEAM code.
- MineExplorer execution code: the dataset card and paper point to
  `https://github.com/Jometeorie/MineExplorer`, but that repository does not
  currently resolve through GitHub. The public benchmark dataset is downloaded;
  the claimed execution framework is not.

## Local Minecraft runtime

- Java 17, Minecraft client 1.19, Fabric Loader 0.14.18, and five Voyager mods
  are installed.
- Mojang's Minecraft 1.19 dedicated server jar is installed and SHA-1 verified.
- The API-free stack binds an offline-mode server to `127.0.0.1`; the Mineflayer
  bot is allowlisted and operator-enabled for deterministic environment setup.
- The Voyager bridge was patched for current `minecrafthawkeye` CommonJS export
  compatibility and a headless-safe hard reset option.
- The exact tracked modifications are in `patches/voyager-local-1.19.patch`;
  reproduce the checkout and Node/Python install with
  `scripts/bootstrap_voyager.sh`.
- Voyager pins Mineflayer 4.8.1, but its crafting transaction hangs against the
  local Minecraft 1.19 dedicated server. Mineflayer is therefore locked to
  4.14.0 after a reference `craftItem` regression test succeeded. This is a
  deliberate Gate-0 compatibility deviation, not an algorithmic change.

## RTX 3090 24 GB execution profile

- Use bf16 for short acquisition (measured successful), int8 for longer
  acquisition, and NF4/bf16 compute for QLoRA training.
- Train LoRA/QLoRA only; keep the backbone frozen.
- Start the pilot with middle and late text layers only.
- Copy incremental K/V for each generated token to pinned CPU memory immediately.
- Cap pilot action generation at 512 new tokens; raise toward 2048 only after a
  measured memory profile.
- Use batch size 1 initially and gradient accumulation 16 to preserve the planned
  effective batch size of 16.
- Run A/B/C merge jobs off-GPU or sequentially; do not retain all rollout KV
  trajectories on the GPU.
- Use gradient checkpointing for slow consolidation and disable model cache while
  training.

## Protocol differences / unavailable upstream

- No GPT-4o/Azure credential is needed for the chosen local-Qwen protocol.
  Consequently, PEAM's paper results are a reference line, not a paired direct
  result under identical acquisition.
- Microsoft authentication is also unnecessary for the loopback-only dedicated
  server protocol.
- Hugging Face CLI is not authenticated; all required public assets are already
  complete.
