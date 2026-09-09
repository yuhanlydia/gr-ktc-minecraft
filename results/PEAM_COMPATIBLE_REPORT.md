# Gate 4 — PEAM-compatible local-Qwen BF16 stress test

Local Qwen-only stress test with published PEAM task/seed/retry/verifier structure. It is not an official PEAM reproduction and is not paired with the paper's GPT-4o tier.

| Method | Success | Wilson 95% CI | Parser valid | Median call | Tokens/task |
|---|---:|---:|---:|---:|---:|
| Local Qwen base | 6/33 (18.2%) | [8.6%, 34.4%] | 104/107 | 16.93s | 280.4 |
| Shared QLoRA BC+DPO | 2/33 (6.1%) | [1.7%, 19.6%] | 120/120 | 26.87s | 253.5 |
| GR-KTC slow (memory-free) | 6/33 (18.2%) | [8.6%, 34.4%] | 113/113 | 16.19s | 198.2 |

## Per-task successes / 3

| Task | Base | BC+DPO | GR-KTC |
|---|---:|---:|---:|
| T1 Craft a crafting table | 3 | 0 | 3 |
| T2 Craft a wooden pickaxe | 0 | 0 | 0 |
| T3 Craft a stone pickaxe | 0 | 0 | 0 |
| T4 Craft a furnace | 0 | 0 | 0 |
| T5 Craft an iron pickaxe | 0 | 0 | 0 |
| T6 Collect 4 oak logs | 3 | 2 | 3 |
| T7 Mine 8 cobblestone | 0 | 0 | 0 |
| T8 Mine 2 iron ore, including required processing | 0 | 0 | 0 |
| T9 Collect 4 coal | 0 | 0 | 0 |
| T10 Defeat a zombie at night | 0 | 0 | 0 |
| T11 Defeat a skeleton with bow | 0 | 0 | 0 |

## Paired tests

- `full_vs_control`: left-only=4, right-only=0, two-sided exact McNemar p=0.125.
- `full_vs_base`: left-only=0, right-only=0, two-sided exact McNemar p=1.
- `control_vs_base`: left-only=0, right-only=4, two-sided exact McNemar p=0.125.

Published Voyager 18/33 and PEAM 23/33 are reference-only and are not used in paired tests.
