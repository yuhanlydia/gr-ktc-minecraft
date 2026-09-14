# AndroidWorld execution status — 2026-09-14

Status: **the preregistered two-family Smoke is complete**.

The analyzer reports `smoke_complete` with `pass: null`. Smoke is integration
evidence only, so it does not issue a scientific GO/NO-GO decision.

## Completed evidence

- AndroidWorld pin: `e3fea3ccc69787570e282c99573298f1c3019a34`.
- Local frozen model: Qwen3-VL-8B-Instruct, BF16, RTX 4090 24 GB.
- Official T3A text observations, official task lifecycle and verifier.
- Contacts acquisition used one exact randomized instance and four model seeds:
  3 successes and 1 failure, all with valid parser output and no exceptions.
- The mixed Contacts group produced a serialized four-token CLSC skill.
- Fresh-instance Contacts evaluation used identical params and model seed across
  conditions: Base `1/1`, CLSC `1/1`. This tie provides no transfer-advantage
  evidence at this sample size.
- Calendar acquisition used one exact randomized instance and four model seeds:
  0 successes and 4 failures, all with valid parser output and no exceptions.
  The model consistently treated minute `45` in the start-time picker as a
  45-minute duration, producing the wrong event time. Because the group had no
  outcome contrast, it was correctly marked `no_quality_signal`; no Calendar
  skill or evaluation was fabricated.
- Peak allocated GPU memory: `21.208 GiB`.

Artifacts are under `smoke/`, including eight acquisition trajectories, the
Contacts serialized skill, paired Contacts evaluation records, `summary.json`,
`report.json`, and `report.md`.

## Infrastructure findings and fixes

The host exposes AMD-V in the CPU and has the KVM kernel module, but this
container cannot open `/dev/kvm` (`EPERM` from the device cgroup). A Pixel 6 / API
33 emulator was therefore run with software acceleration. It boots and can run
AndroidWorld, but Android operations take much longer than the KVM assumptions
in the upstream defaults.

The runner now has a `--slow-emulator` path that:

1. wakes and unlocks the device before repairing an unbound/crashed AndroidEnv
   accessibility forwarder;
2. sends tree logging and the selected gRPC port with asynchronous broadcasts;
3. waits up to 60 seconds for the first accessibility tree;
4. uses 60-second app-start, text-input, and generic ADB timeouts;
5. waits eight seconds after each GUI transition so the UI tree matches the
   action target.

A cold boot can remove AndroidWorld's snapshot directories even while installed
app data remains. The runner now checks every selected task's required snapshot
directory before sampling parameters and fails closed with the missing package
names. This prevents scientifically invalid runs against a dirty initial state.

The model path also stopped returning full-vocabulary logits for every prompt
token during KV-prefix prefill. It now requests only last-token logits, which is
all generation needs. A 24,009-token real-model regression completed at
`22.029 GiB` peak allocation. Before this fix, a shorter step-8 summary attempted
an extra 5.22 GiB allocation and OOMed. AndroidWorld exception records with a
`NaN` episode length are now recorded safely instead of crashing the runner.

The `train` extra now includes `torchvision`, which Qwen's processor requires.
The analysis script no longer emits a scientific NO-GO for incomplete runs or
for Smoke, which is an integration phase.

## Next gate

Pilot is the next scientific experiment:

```bash
export ANDROID_SDK_ROOT=/opt/android
export PATH=/opt/android/platform-tools:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/run_androidworld_latentskill_gate.py \\
  --phase pilot \\
  --adb-path /opt/android/platform-tools/adb \\
  --slow-emulator \\
  --resume
```

The six Pilot apps are now installed, their initial states were inspected, and
fresh snapshots were saved. Markor's onboarding and all-files permission, the
SMS default-app role, Expense onboarding, and the Clock initial launch were
completed explicitly after upstream's short setup waits timed out. A bounded
Pilot launch passed snapshot preflight, loaded the model, repaired the
accessibility forwarder, and began the first acquisition episode; it was then
stopped by the 90-second validation timeout before producing a result. The
timeout-created Pilot manifest was removed so it cannot be mistaken for data.
The six validated device snapshots were archived locally at
`/root/androidworld-pilot-snapshots-20260914.tar.gz` before emulator shutdown.

The formal protocol was subsequently raised from 4 to 10 fixed rollouts per
acquisition instance before any Pilot result was collected. Pilot therefore
requires 120 acquisition episodes and up to 144 evaluation episodes. On
this software-only emulator, the observed Smoke episode times imply a run
measured in many hours or days rather than the two-hour debugging budget. Use
KVM or another accelerated Android host for the scientific Pilot.

Under the one-final-gate policy, do not interpret the Contacts tie as a rescue
signal and do not add PDC, MetaPlastic, LoRA, retrieval, or other modules. Pilot
must compare the preregistered baselines before any stop/continue decision.

## Verification

- 97 runnable repository tests pass.
- Python compilation, bootstrap shell syntax, dependency consistency, artifact
  assertions, and `git diff --check` pass.
- The unfiltered suite has 10 known asset failures: six require the ignored
  MineExplorer benchmark JSONL and four require Voyager's ignored Babel install.
  These assets are outside the AndroidWorld gate and were not fabricated.
