# Gate 4 — PEAM-compatible local-Qwen NF4 diagnostic

Diagnostic NF4 run with the published PEAM task/seed/retry/verifier structure.
It is not the BF16 primary run, an official PEAM reproduction, or paired with
the paper's GPT-4o tier.

| Method | Success | Wilson 95% CI | Parser valid | Median call | Tokens/task |
|---|---:|---:|---:|---:|---:|
| Local Qwen base | 6/33 (18.2%) | [8.6%, 34.4%] | 110/116 | 18.54s | 367.4 |
| Shared QLoRA BC+DPO | 3/33 (9.1%) | [3.1%, 23.6%] | 123/123 | 30.59s | 275.3 |
| GR-KTC slow (memory-free) | 5/33 (15.2%) | [6.7%, 30.9%] | 123/123 | 23.25s | 245.6 |

## Per-task successes / 3

| Task | Base | BC+DPO | GR-KTC |
|---|---:|---:|---:|
| T1 Craft a crafting table | 3 | 0 | 2 |
| T2 Craft a wooden pickaxe | 0 | 0 | 0 |
| T3 Craft a stone pickaxe | 0 | 0 | 0 |
| T4 Craft a furnace | 0 | 0 | 0 |
| T5 Craft an iron pickaxe | 0 | 0 | 0 |
| T6 Collect 4 oak logs | 3 | 3 | 3 |
| T7 Mine 8 cobblestone | 0 | 0 | 0 |
| T8 Mine 2 iron ore, including required processing | 0 | 0 | 0 |
| T9 Collect 4 coal | 0 | 0 | 0 |
| T10 Defeat a zombie at night | 0 | 0 | 0 |
| T11 Defeat a skeleton with bow | 0 | 0 | 0 |

## Paired tests

- `full_vs_control`: left-only=2, right-only=0, two-sided exact McNemar p=0.5.
- `full_vs_base`: left-only=0, right-only=1, two-sided exact McNemar p=1.
- `control_vs_base`: left-only=0, right-only=3, two-sided exact McNemar p=0.25.

Published Voyager 18/33 and PEAM 23/33 are reference-only and are not used in paired tests.
