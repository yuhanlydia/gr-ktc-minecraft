# MiniWoB++ LatentSkill quick-gate status — 2026-09-14

The fixed K=10 signal scan completed on all eight preregistered task families.
Success means native `RAW_REWARD_GLOBAL >= 0.5`.

| Family | Success | Failure | Reward std | Qualified |
|---|---:|---:|---:|:---:|
| `form-sequence` | 0 | 10 | 0.000 | no |
| `choose-date` | 2 | 8 | 0.400 | no |
| `email-inbox` | 8 | 2 | 0.400 | no |
| `login-user-popup` | 1 | 9 | 0.300 | no |
| `social-media-some` | 0 | 10 | 0.000 | no |
| `use-autocomplete` | 1 | 9 | 0.300 | no |
| `navigate-tree` | 4 | 6 | 0.490 | yes |
| `book-flight-nodelay` | 0 | 10 | 0.000 | no |

The protocol required two qualified families and found one. The analyzer status
is therefore `insufficient_quality_signal`, rather than a method-level pass or
failure.

The qualified `navigate-tree` family completed all 32 paired evaluations:

| Mode | Fresh-seed success |
|---|---:|
| Base | 6/8 |
| Context | 6/8 |
| Positive-All | 6/8 |
| CLSC | 6/8 |

CLSC versus each baseline had 0 wins, 0 losses, and 8 ties. The memory changed
an action sequence on one failed instance but did not change any task outcome.
Peak allocated GPU memory was 18.25 GiB.

Decision: this run supplies no positive reusable-skill evidence. Stop further
MiniWoB expansion and do not use this result as the basis for an ICASSP paper.
The result does not mathematically disprove CLSC because the aggregate gate was
underpowered by task-family qualification; it does provide negative directional
evidence on the only family that could be evaluated.
