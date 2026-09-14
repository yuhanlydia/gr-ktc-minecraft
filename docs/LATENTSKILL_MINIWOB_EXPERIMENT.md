# LatentSkill / CLSC MiniWoB++ Fast Reliability Gate

## Question

This gate asks whether verifier-contrastive four-token native K/V memory learned
from repeated attempts transfers to fresh randomized instances of the same UI
task family. It is a fast favorable mechanism test before expensive
AndroidWorld confirmation, not a MiniWoB++ leaderboard submission.

## Correct reward signal

The runner reads MiniWoB's native `RAW_REWARD_GLOBAL` from BrowserGym task info.
It never uses BrowserGym's binary wrapper reward, because that wrapper currently
turns every positive partial reward into success.

```text
terminal success    RAW_REWARD_GLOBAL >= 0.5
acquisition quality clip(RAW_REWARD_GLOBAL, 0, 1)
```

Missing or non-finite raw reward aborts the episode. Every result stores the raw
reward, clipped quality reward, threshold, wrapper reward, and reward reason.

## Fixed quick protocol

The signal scan runs K=10 stochastic rollouts on one exact seeded instance per
task family. A family qualifies only with at least three successes, three
failures, and reward standard deviation at least 0.15. The scan uses a fixed
task order and stops after two families qualify.

Each qualified family is evaluated on eight fresh seeds under paired greedy
conditions:

```text
base
context
positive_all
clsc
```

All modes for a fresh instance receive the same randomized goal and model seed.
CLSC retains the AndroidWorld preregistration: frozen Qwen3-VL-8B-Instruct BF16,
four memory tokens, quality layer 24, value scale 0.25, and negative scale 0.5.

## Setup

```bash
python -m pip install -e '.[train,test,miniwob]'
PYTHON_BIN=python bash scripts/bootstrap_miniwob.sh
```

The bootstrap pins BrowserGym to
`9e779f087de9a65668b6974d11f9ce9816026e96`, pins MiniWoB++ to
`7fd85d71a4b60325c6585396ec4f48377d049838`, installs the local BrowserGym
packages, and installs Playwright Chromium.

## Real smoke

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_miniwob_latentskill_gate.py \
  --phase smoke \
  --max-steps 6 \
  --resume
```

Smoke verifies real Chromium reset, accessibility-tree extraction, Qwen action
generation, action execution, native reward capture, and K/V serialization. Two
smoke outcomes need not be mixed; the runner reports `no_quality_signal` rather
than inventing a skill.

The 2026-09-14 real smoke at seed 42 completed both rollouts in one action. Both
generated `click('20')`, received native raw reward 1.0, and serialized 36
non-empty K/V layer tensors. Peak allocated GPU memory was 16.39 GiB. The smoke
status is `insufficient_quality_signal` because the two outcomes were both
successful; this validates the runtime path but is not evidence that CLSC
improves transfer.

## Quick reliability gate

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_miniwob_latentskill_gate.py \
  --phase quick \
  --resume

python scripts/analyze_miniwob_latentskill.py \
  results/latentskill_miniwob/quick/summary.json
```

The analyzer issues a positive result only when two qualified families exist,
aggregate CLSC exceeds Base by at least five percentage points, CLSC exceeds
Context and Positive-All, and CLSC has more paired wins than losses against Base
in both families. Failure to find two qualified families is reported separately
as `insufficient_quality_signal`.

## 2026-09-14 quick-gate result

The completed K=10 scan found one qualified family out of eight. The planned
gate required two, so the formal result is `insufficient_quality_signal`, with
no pass/fail claim about CLSC as a whole. `miniwob.navigate-tree` qualified with
4/10 successes. Its eight fresh instances produced the same 6/8 success rate
for Base, Context, Positive-All, and CLSC (all eight CLSC/Base pairs tied).

This is not positive evidence for reusable latent skill. The benchmark scan did
not provide enough qualified families for the preregistered aggregate test, and
the one evaluable family showed zero task-level benefit. Per the one-final-gate
decision rule, do not expand this experiment or prepare an ICASSP submission
from these results.

Large K/V tensors remain ignored by Git. Commit manifests, scalar episode JSON,
summaries, and reports.
