# MiniWoB++ CLSC Fast Reliability Gate

## Purpose

Build a fast, reproducible test of whether training-free verifier-contrastive
latent K/V state transfers to fresh randomized instances of the same UI task
family. This is a mechanism reliability test before returning to expensive
AndroidWorld confirmation. It is not a MiniWoB++ leaderboard submission.

## Benchmark and pins

Use BrowserGym at commit
`9e779f087de9a65668b6974d11f9ce9816026e96` and MiniWoB++ at BrowserGym's
recommended commit `7fd85d71a4b60325c6585396ec4f48377d049838`.
`scripts/bootstrap_miniwob.sh` clones, verifies, and installs both checkouts and
the Playwright Chromium runtime. Runtime imports remain lazy so protocol tests
do not require BrowserGym.

The observation is BrowserGym's flattened accessibility tree. The action space
is one high-level BrowserGym action per model call using element `bid` values:
`click`, `fill`, `select_option`, `press`, `clear`, `focus`, `scroll`, or
`noop`. The model is the same frozen Qwen3-VL-8B-Instruct BF16 checkpoint used
by AndroidWorld. Each action-selection call captures or receives latent K/V;
there is no extra summary-model call.

## Reward correctness

BrowserGym's MiniWoB wrapper currently maps every positive raw reward to binary
success. Some MiniWoB tasks award small positive partial credit for an incorrect
terminal action, so the gate must not consume that wrapper reward.

Every record stores `RAW_REWARD_GLOBAL` as `raw_reward`. Acquisition quality is
`clip(raw_reward, 0, 1)`. Terminal success is preregistered as
`raw_reward >= 0.5`. The threshold, raw reward, normalized acquisition reward,
termination state, and reward reason are written to every result. Non-finite or
missing raw rewards fail closed.

CLSC uses continuous acquisition rewards for group-relative advantages, while
Positive-All and Failed use the separately supplied terminal-success mask. The
existing latent-skill API gains an optional explicit success mask; callers that
omit it preserve the current `reward > 0` behavior.

## Protocol

The quick phase scans a fixed ordered task list. For each family it runs one
exact seeded acquisition instance with K=10 stochastic model rollouts. A family
qualifies only when all ten trajectories are valid, at least three are terminal
successes, at least three are terminal failures, and acquisition-reward standard
deviation is at least 0.15. The fixed rule is recorded before sampling; skipped
families and their outcomes remain in the summary.

Scanning stops after two qualifying families. Each qualifying family is then
evaluated on eight fresh task seeds. Every seed uses the same task parameters
and model seed across four conditions:

1. `base`: no latent memory;
2. `context`: context center at every layer;
3. `positive_all`: successful-trajectory state at every layer;
4. `clsc`: context at every layer, with verifier-contrastive quality only at
   the preregistered layer 24.

Evaluation is greedy. Acquisition uses temperature 0.9 and top-p 0.95. Memory
size remains four native K/V tokens, value scale 0.25, negative scale 0.5, and
quality layer 24. Acquisition and evaluation seed namespaces are disjoint.

The smoke phase uses two acquisition rollouts and one fresh seed on one task to
verify BrowserGym reset, accessibility-tree observation, action execution, raw
reward capture, model generation, and K/V capture. If its two outcomes are not
mixed, it reports `no_quality_signal` without fabricating evaluation memory.

## Default scan families

The fixed scan order favors multi-step tasks supported by the bid action set:

1. `miniwob.form-sequence`
2. `miniwob.choose-date`
3. `miniwob.email-inbox`
4. `miniwob.login-user-popup`
5. `miniwob.social-media-some`
6. `miniwob.use-autocomplete`
7. `miniwob.navigate-tree`
8. `miniwob.book-flight-nodelay`

CLI task overrides are allowed for smoke and debugging. A scientific quick run
records overrides in its manifest and must not silently replace the fixed list.

## Artifacts and resume behavior

Results live under `results/latentskill_miniwob/<phase>/`:

- `manifest.json`: pins, model, protocol constants, task order, and every seed;
- `acquisition/<family>/rollout_<k>/result.json` and ignored K/V tensors;
- `skills/<family>.json` and ignored serialized tensors;
- `eval/<mode>/<family>/eval_<j>.json`;
- `summary.json`, `report.json`, and `report.md`.

`--resume` reuses an episode only when both result JSON and its acquisition K/V
tensor exist. It verifies a single goal fingerprint across the K acquisition
rollouts and a single goal fingerprint across modes for each fresh seed.

## Decision and interpretation

The analyzer reports paired wins/losses for CLSC against Base, Context, and
Positive-All. The fast gate is positive only if two qualified families exist,
aggregate CLSC success exceeds all three baselines, CLSC beats Base by at least
5 percentage points, and CLSC has a positive paired difference in both
families. With two families and eight fresh seeds each, one paired outcome is
6.25 percentage points.

Failure to find two qualifying families is `insufficient_quality_signal`, not a
CLSC performance failure. Finding them and then failing the paired comparison
is a method-level negative result under this favorable setting.

## Tests

Pure tests cover raw-reward validation, threshold semantics, qualification,
explicit success masks, deterministic seed separation, action extraction,
paired-mode integrity, resume rules, and analyzer decisions. A real smoke must
launch Chromium, reset a seeded MiniWoB task, execute at least one Qwen action,
read raw reward, and save captured K/V without BrowserGym/OpenAI API calls.
