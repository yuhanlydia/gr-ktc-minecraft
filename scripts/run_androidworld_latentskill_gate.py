#!/usr/bin/env python3
"""Run the LatentSkill/CLSC survival gate on pinned AndroidWorld.

The first gate is intentionally favorable to the previously successful
Gate-2 mechanism:

* same AndroidWorld task template, fresh randomized parameter instances;
* oracle routing by the official task-template name;
* frozen Qwen3-VL-8B-Instruct in BF16;
* 4 native latent K/V tokens;
* context state at all layers and verifier-contrastive quality at layer 24;
* official AndroidWorld T3A prompt/action/history, task lifecycle, verifier, and
  step budget.

No Prompt Decommitment, MetaPlastic, LoRA, learned retrieval, or API teacher is
used in this gate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.generation import generate_with_final_kv, generate_with_kv_prefix
from gr_ktc.latent_skill import (
    FamilyMemoryBundle,
    build_family_memory_bundle,
    concat_step_kv,
    load_family_memory_bundle,
    save_family_memory_bundle,
)
from gr_ktc.model_loader import load_qwen3_vl_24gb


ANDROIDWORLD_PIN = "e3fea3ccc69787570e282c99573298f1c3019a34"
QUALITY_LAYER = 24
MEMORY_TOKENS = 4
VALUE_SCALE = 0.25
NEGATIVE_SCALE = 0.5
ACQUISITION_TEMPERATURE = 0.9
ACQUISITION_TOP_P = 0.95
EVALUATION_TEMPERATURE = 0.0
MODES = ("base", "context", "positive_all", "failed", "quality_all", "clsc")
_DEFAULT_TASKS = (
    "ContactsAddContact",
    "SimpleCalendarAddOneEvent",
    "MarkorCreateNote",
    "SimpleSmsSend",
    "ClockTimerEntry",
    "ExpenseAddSingle",
    "RecipeAddSingleRecipe",
    "VlcCreatePlaylist",
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    task_count: int
    train_instances: int
    rollouts_per_instance: int
    test_instances: int
    modes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRoute:
    capture_kv: bool
    inject_memory: bool


def phase_spec(name: str) -> PhaseSpec:
    specs = {
        "smoke": PhaseSpec("smoke", 2, 1, 4, 1, ("base", "clsc")),
        "pilot": PhaseSpec("pilot", 6, 2, 4, 4, MODES),
        "full": PhaseSpec("full", 8, 4, 4, 8, MODES),
    }
    try:
        return specs[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown phase {name!r}; choose smoke, pilot, or full") from exc


def default_task_names() -> tuple[str, ...]:
    return _DEFAULT_TASKS


def instance_seed(split: str, family: str, index: int, base_seed: int) -> int:
    """Stable seed with explicit acquisition/evaluation namespace separation."""
    if index < 0:
        raise ValueError("instance index must be non-negative")
    payload = f"{int(base_seed)}|{str(split)}|{str(family)}|{int(index)}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16) % (2**31 - 1)


def generation_route(
    *, call_kind: str, acquisition: bool, mode: str
) -> GenerationRoute:
    """Decide whether one official T3A LLM call captures or consumes skill K/V."""
    if call_kind not in {"action", "summary"}:
        raise ValueError("call_kind must be action or summary")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    if call_kind == "summary":
        return GenerationRoute(False, False)
    if acquisition:
        return GenerationRoute(True, False)
    return GenerationRoute(False, mode != "base")


def _install_androidworld(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "android_world" / "registry.py").exists():
        raise FileNotFoundError(
            f"AndroidWorld checkout not found at {root}. "
            "Run scripts/bootstrap_androidworld.sh first."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_androidworld_runtime(root: Path, *, allow_drift: bool = False) -> SimpleNamespace:
    _install_androidworld(root)
    head = _git_head(root)
    if head != ANDROIDWORLD_PIN and not allow_drift:
        raise RuntimeError(
            f"AndroidWorld HEAD is {head}, expected pinned {ANDROIDWORLD_PIN}. "
            "Run scripts/bootstrap_androidworld.sh or pass --allow-androidworld-drift "
            "only for debugging, not paper results."
        )
    from android_world import constants, episode_runner, registry, suite_utils
    from android_world.agents import agent_utils, m3a_utils, t3a
    from android_world.env import env_launcher, json_action

    return SimpleNamespace(
        head=head,
        constants=constants,
        episode_runner=episode_runner,
        registry=registry,
        suite_utils=suite_utils,
        agent_utils=agent_utils,
        m3a_utils=m3a_utils,
        t3a=t3a,
        env_launcher=env_launcher,
        json_action=json_action,
    )


def _eos_ids(model: Any, processor: Any) -> tuple[int, ...]:
    candidates: list[int] = []
    for raw in (
        getattr(processor.tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if isinstance(raw, int):
            candidates.append(raw)
        elif isinstance(raw, (list, tuple)):
            candidates.extend(int(value) for value in raw if isinstance(value, int))
    return tuple(dict.fromkeys(candidates))


def _model_inputs(model: Any, processor: Any, prompt: str) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": str(prompt)}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _call_kind(prompt: str) -> str:
    lowered = prompt.lower()
    if "summary of this step:" in lowered or "summerize the latest step" in lowered:
        return "summary"
    return "action"


class LocalQwenT3AWrapper:
    """Drop-in `predict` wrapper for official T3A with latent K/V instrumentation."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        family: str,
        acquisition: bool,
        mode: str,
        memory: Any = None,
        seed: int = 0,
        action_max_new_tokens: int = 192,
        summary_max_new_tokens: int = 96,
        action_temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.processor = processor
        self.family = str(family)
        self.acquisition = bool(acquisition)
        self.mode = str(mode)
        self.memory = memory
        self.seed = int(seed)
        self.action_max_new_tokens = int(action_max_new_tokens)
        self.summary_max_new_tokens = int(summary_max_new_tokens)
        self.action_temperature = float(action_temperature)
        self.top_p = float(top_p)
        self.call_index = 0
        self.captured_action_kv: list[dict[int, torch.Tensor]] = []
        self.action_calls = 0
        self.summary_calls = 0
        self.layer_ids = list(range(model.config.text_config.num_hidden_layers))
        self._eos = _eos_ids(model, processor)

    def _decode(self, ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)

    def predict(self, prompt: str) -> tuple[str, bool, str]:
        call_kind = _call_kind(prompt)
        route = generation_route(
            call_kind=call_kind, acquisition=self.acquisition, mode=self.mode
        )
        if call_kind == "action":
            self.action_calls += 1
            max_tokens = self.action_max_new_tokens
            temperature = self.action_temperature
        else:
            self.summary_calls += 1
            max_tokens = self.summary_max_new_tokens
            temperature = 0.0

        torch.manual_seed(self.seed + self.call_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + self.call_index)
        generator = torch.Generator(device=self.model.device).manual_seed(
            self.seed + self.call_index
        )
        self.call_index += 1
        inputs = _model_inputs(self.model, self.processor, prompt)

        if route.capture_kv:
            generated = generate_with_final_kv(
                self.model,
                inputs,
                layer_ids=self.layer_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
            )
            self.captured_action_kv.append(generated.kv_by_layer)
            text = self._decode(generated.all_generated_token_ids)
        else:
            memory = self.memory if route.inject_memory else None
            ids = generate_with_kv_prefix(
                self.model,
                inputs,
                memory,
                context_id=(f"family:{self.family}" if memory is not None else None),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                eos_token_ids=self._eos,
                generator=generator,
            )
            text = self._decode(ids)
        return text, True, text


def _param_fingerprint(params: Mapping[str, Any]) -> str:
    try:
        raw = pickle.dumps(dict(params), protocol=4)
    except Exception:
        raw = repr(params).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def _parser_stats(runtime: SimpleNamespace, episode: Mapping[str, Any]) -> tuple[bool, float]:
    constants = runtime.constants.EpisodeConstants
    data = episode.get(constants.EPISODE_DATA) or {}
    outputs = list(data.get("action_output", [])) if isinstance(data, Mapping) else []
    if not outputs:
        return False, 0.0
    valid = 0
    for raw in outputs:
        try:
            reason, action = runtime.m3a_utils.parse_reason_action_output(raw or "")
            if not reason or not action:
                continue
            runtime.json_action.JSONAction(
                **runtime.agent_utils.extract_json(action)
            )
            valid += 1
        except Exception:
            continue
    rate = valid / len(outputs)
    return valid == len(outputs), rate


def _run_one_episode(
    *,
    runtime: SimpleNamespace,
    env: Any,
    task_type: Any,
    param_seed: int,
    model: Any,
    processor: Any,
    family: str,
    acquisition: bool,
    mode: str,
    memory: Any,
    model_seed: int,
    action_max_new_tokens: int,
    summary_max_new_tokens: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor] | None]:
    task = runtime.suite_utils._instantiate_task(
        task_type, seed=param_seed, env=env
    )
    fingerprint = _param_fingerprint(task.params)
    wrapper = LocalQwenT3AWrapper(
        model,
        processor,
        family=family,
        acquisition=acquisition,
        mode=mode,
        memory=memory,
        seed=model_seed,
        action_max_new_tokens=action_max_new_tokens,
        summary_max_new_tokens=summary_max_new_tokens,
        action_temperature=(ACQUISITION_TEMPERATURE if acquisition else 0.0),
        top_p=ACQUISITION_TOP_P,
    )
    agent = runtime.t3a.T3A(env, wrapper, name=f"LatentSkill-{mode}")

    def run_episode(current_task):
        return runtime.episode_runner.run_episode(
            goal=current_task.goal,
            agent=agent,
            max_n_steps=runtime.suite_utils._allocate_step_budget(
                current_task.complexity
            ),
            start_on_home_screen=current_task.start_on_home_screen,
            print_fn=lambda _message: None,
        )

    started = time.perf_counter()
    result = runtime.suite_utils._run_task(
        task, run_episode, env, demo_mode=False
    )
    # `_run_task` tears down on normal completion, but not when initialization
    # or execution raises. Restore benchmark hygiene before the next condition.
    if getattr(task, "initialized", False):
        try:
            task.tear_down(env)
        except Exception:
            pass

    fields = runtime.constants.EpisodeConstants
    reward = float(result.get(fields.IS_SUCCESSFUL, 0.0) or 0.0)
    parser_valid, parser_valid_rate = _parser_stats(runtime, result)
    exception = result.get(fields.EXCEPTION_INFO)
    record = {
        "family": family,
        "goal": str(result.get(fields.GOAL, getattr(task, "goal", ""))),
        "param_seed": int(param_seed),
        "param_fingerprint": fingerprint,
        "mode": mode,
        "verifier_reward": reward,
        "success": bool(reward > 0.5),
        "steps": int(result.get(fields.EPISODE_LENGTH, 0) or 0),
        "parser_valid": bool(parser_valid),
        "parser_valid_rate": float(parser_valid_rate),
        "action_calls": wrapper.action_calls,
        "summary_calls": wrapper.summary_calls,
        "exception": str(exception) if exception else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    trajectory = None
    if acquisition and wrapper.captured_action_kv and not exception:
        trajectory = concat_step_kv(wrapper.captured_action_kv)
    return record, trajectory


def _save_episode_kv(path: Path, trajectory: Mapping[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"layer_{int(layer)}": tensor.detach().cpu().contiguous()
            for layer, tensor in trajectory.items()
        },
        str(path),
    )


def _load_episode_kv(path: Path) -> dict[int, torch.Tensor]:
    tensors = load_file(str(path))
    return {
        int(name.removeprefix("layer_")): tensor.contiguous()
        for name, tensor in tensors.items()
        if name.startswith("layer_")
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _resolve_adb_path(raw: Path | None) -> Path:
    candidates = []
    if raw is not None:
        candidates.append(raw.expanduser())
    candidates.extend(
        [
            Path.home() / "Android/Sdk/platform-tools/adb",
            Path.home() / "Library/Android/sdk/platform-tools/adb",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "adb not found. Pass --adb-path or install Android SDK platform-tools."
    )


def _build_family_skill(
    *,
    runtime: SimpleNamespace,
    env: Any,
    family: str,
    task_type: Any,
    spec: PhaseSpec,
    base_seed: int,
    model: Any,
    processor: Any,
    output_dir: Path,
    action_max_new_tokens: int,
    summary_max_new_tokens: int,
    resume: bool,
) -> tuple[dict[str, Any], FamilyMemoryBundle | None]:
    trajectories: list[dict[int, torch.Tensor]] = []
    rewards: list[float] = []
    group_ids: list[str] = []
    acquisition_records: list[dict[str, Any]] = []
    complete_groups = 0

    for instance_index in range(spec.train_instances):
        group_id = f"{family}:acq:{instance_index}"
        param_seed = instance_seed("acquisition", family, instance_index, base_seed)
        group_entries: list[tuple[dict[str, Any], dict[int, torch.Tensor]]] = []
        fingerprints: set[str] = set()

        for rollout_index in range(spec.rollouts_per_instance):
            rollout_seed = instance_seed(
                "rollout", family, instance_index * 100 + rollout_index, base_seed
            )
            run_dir = (
                output_dir
                / "acquisition"
                / family
                / f"instance_{instance_index:02d}"
                / f"rollout_{rollout_index:02d}"
            )
            result_path = run_dir / "result.json"
            kv_path = run_dir / "episode_kv.safetensors"
            if resume and result_path.exists() and kv_path.exists():
                record = json.loads(result_path.read_text())
                trajectory = _load_episode_kv(kv_path)
            else:
                record, trajectory = _run_one_episode(
                    runtime=runtime,
                    env=env,
                    task_type=task_type,
                    param_seed=param_seed,
                    model=model,
                    processor=processor,
                    family=family,
                    acquisition=True,
                    mode="base",
                    memory=None,
                    model_seed=rollout_seed,
                    action_max_new_tokens=action_max_new_tokens,
                    summary_max_new_tokens=summary_max_new_tokens,
                )
                record.update(
                    {
                        "split": "acquisition",
                        "instance_id": f"acq-{instance_index:02d}",
                        "rollout_index": rollout_index,
                        "model_seed": rollout_seed,
                        "group_id": group_id,
                    }
                )
                _write_json(result_path, record)
                if trajectory is not None:
                    _save_episode_kv(kv_path, trajectory)

            acquisition_records.append(record)
            fingerprints.add(str(record.get("param_fingerprint", "")))
            if trajectory is not None and trajectory:
                group_entries.append((record, trajectory))

        # Scientific grouping requires all K attempts on exactly the same
        # randomized task instance.  Partial/crashed groups are not reweighted.
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"AndroidWorld regenerated different params within {group_id}"
            )
        if len(group_entries) != spec.rollouts_per_instance:
            continue
        complete_groups += 1
        for record, trajectory in group_entries:
            trajectories.append(trajectory)
            rewards.append(float(record["verifier_reward"]))
            group_ids.append(group_id)

    family_info: dict[str, Any] = {
        "family": family,
        "status": "no_quality_signal",
        "train_instances_requested": spec.train_instances,
        "complete_acquisition_groups": complete_groups,
        "rollouts_per_instance": spec.rollouts_per_instance,
        "acquisition": acquisition_records,
        "mixed_group_count": 0,
    }
    if not trajectories:
        family_info["reason"] = "no complete acquisition groups with K/V trajectories"
        return family_info, None

    text_config = model.config.text_config
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    try:
        bundle = build_family_memory_bundle(
            family_id=family,
            trajectories=trajectories,
            rewards=rewards,
            group_ids=group_ids,
            kv_heads=text_config.num_key_value_heads,
            head_dim=head_dim,
            quality_layer=QUALITY_LAYER,
            memory_tokens=MEMORY_TOKENS,
            value_scale=VALUE_SCALE,
            negative_scale=NEGATIVE_SCALE,
        )
    except ValueError as exc:
        family_info["reason"] = str(exc)
        return family_info, None

    skill_dir = output_dir / "skills"
    tensor_path = skill_dir / f"{family}.safetensors"
    metadata_path = skill_dir / f"{family}.json"
    save_family_memory_bundle(bundle, tensor_path, metadata_path)
    family_info.update(
        {
            "status": "ready",
            "mixed_group_count": bundle.mixed_group_count,
            "rollout_count": bundle.rollout_count,
            "skill_tensor": str(tensor_path),
            "skill_metadata": str(metadata_path),
        }
    )
    return family_info, bundle


def _memory_for_mode(bundle: FamilyMemoryBundle, mode: str):
    if mode == "base":
        return None
    return bundle.memories()[mode]


def _evaluate_family(
    *,
    runtime: SimpleNamespace,
    env: Any,
    family: str,
    task_type: Any,
    spec: PhaseSpec,
    modes: Sequence[str],
    base_seed: int,
    model: Any,
    processor: Any,
    bundle: FamilyMemoryBundle,
    output_dir: Path,
    action_max_new_tokens: int,
    summary_max_new_tokens: int,
    resume: bool,
    existing: Mapping[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for instance_index in range(spec.test_instances):
        instance_id = f"eval-{instance_index:02d}"
        param_seed = instance_seed("evaluation", family, instance_index, base_seed)
        reference_fingerprint: str | None = None
        for mode in modes:
            key = (family, instance_id, mode)
            if resume and key in existing:
                record = dict(existing[key])
                records.append(record)
                fingerprint = str(record.get("param_fingerprint", ""))
                if reference_fingerprint is None:
                    reference_fingerprint = fingerprint
                elif fingerprint != reference_fingerprint:
                    raise RuntimeError(
                        f"resume data contains unmatched params for {family}/{instance_id}"
                    )
                continue

            memory = _memory_for_mode(bundle, mode)
            model_seed = instance_seed(
                "evaluation-model", family, instance_index, base_seed
            )
            record, _ = _run_one_episode(
                runtime=runtime,
                env=env,
                task_type=task_type,
                param_seed=param_seed,
                model=model,
                processor=processor,
                family=family,
                acquisition=False,
                mode=mode,
                memory=memory,
                model_seed=model_seed,
                action_max_new_tokens=action_max_new_tokens,
                summary_max_new_tokens=summary_max_new_tokens,
            )
            record.update(
                {
                    "split": "evaluation",
                    "instance_id": instance_id,
                    "model_seed": model_seed,
                }
            )
            fingerprint = str(record["param_fingerprint"])
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
            elif fingerprint != reference_fingerprint:
                raise RuntimeError(
                    f"modes received different randomized params for {family}/{instance_id}"
                )
            run_dir = output_dir / "eval" / mode / family
            _write_json(run_dir / f"{instance_id}.json", record)
            records.append(record)
    return records


def _manifest(
    *,
    phase: PhaseSpec,
    tasks: Sequence[str],
    model_path: Path,
    androidworld_head: str,
    base_seed: int,
    modes: Sequence[str],
) -> dict[str, Any]:
    return {
        "protocol": "latentskill-androidworld-oracle-family-v1",
        "phase": phase.name,
        "androidworld_pin": ANDROIDWORLD_PIN,
        "androidworld_head": androidworld_head,
        "model": str(model_path),
        "precision": "bf16",
        "observation_protocol": "official-T3A-text",
        "quality_layer": QUALITY_LAYER,
        "memory_tokens": MEMORY_TOKENS,
        "value_scale": VALUE_SCALE,
        "negative_scale": NEGATIVE_SCALE,
        "acquisition_temperature": ACQUISITION_TEMPERATURE,
        "acquisition_top_p": ACQUISITION_TOP_P,
        "evaluation_temperature": EVALUATION_TEMPERATURE,
        "oracle_routing": True,
        "base_seed": int(base_seed),
        "tasks": list(tasks),
        "modes": list(modes),
        "train_instances_per_family": phase.train_instances,
        "rollouts_per_instance": phase.rollouts_per_instance,
        "test_instances_per_family": phase.test_instances,
        "acquisition_param_seeds": {
            family: [
                instance_seed("acquisition", family, i, base_seed)
                for i in range(phase.train_instances)
            ]
            for family in tasks
        },
        "evaluation_param_seeds": {
            family: [
                instance_seed("evaluation", family, i, base_seed)
                for i in range(phase.test_instances)
            ]
            for family in tasks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument(
        "--androidworld-root", type=Path,
        default=ROOT / "third_party/android_world",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=ROOT / "models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/latentskill_androidworld",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=list(MODES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adb-path", type=Path)
    parser.add_argument("--console-port", type=int, default=5554)
    parser.add_argument("--perform-emulator-setup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-androidworld-drift", action="store_true")
    parser.add_argument("--action-max-new-tokens", type=int, default=192)
    parser.add_argument("--summary-max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    spec = phase_spec(args.phase)
    tasks = tuple(args.tasks) if args.tasks else default_task_names()[: spec.task_count]
    modes = tuple(args.modes) if args.modes else spec.modes
    if any(mode not in MODES for mode in modes):
        raise ValueError(f"modes must be a subset of {MODES}")
    if "clsc" not in modes and args.phase != "smoke":
        raise ValueError("pilot/full scientific runs must include clsc")

    output_dir = args.output_dir.expanduser().resolve() / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _load_androidworld_runtime(
        args.androidworld_root, allow_drift=args.allow_androidworld_drift
    )
    registry = runtime.registry.TaskRegistry().get_registry(
        runtime.registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    missing = [family for family in tasks if family not in registry]
    if missing:
        raise ValueError(f"task names absent from pinned AndroidWorld registry: {missing}")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"model not found at {model_path}; expected Qwen3-VL-8B-Instruct"
        )
    adb_path = _resolve_adb_path(args.adb_path)
    manifest = _manifest(
        phase=spec,
        tasks=tasks,
        model_path=model_path,
        androidworld_head=runtime.head,
        base_seed=args.seed,
        modes=modes,
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, processor = load_qwen3_vl_24gb(model_path, precision="bf16")
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    env = runtime.env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=args.perform_emulator_setup,
        adb_path=str(adb_path),
    )

    previous_summary = {}
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.exists():
        previous_summary = json.loads(summary_path.read_text())
    existing_eval = {
        (
            str(record.get("family")),
            str(record.get("instance_id")),
            str(record.get("mode")),
        ): record
        for record in previous_summary.get("evaluation", [])
    }

    families: dict[str, Any] = {}
    evaluation: list[dict[str, Any]] = list(existing_eval.values())
    try:
        for family in tasks:
            print(f"\n=== Acquisition: {family} ===", flush=True)
            info, bundle = _build_family_skill(
                runtime=runtime,
                env=env,
                family=family,
                task_type=registry[family],
                spec=spec,
                base_seed=args.seed,
                model=model,
                processor=processor,
                output_dir=output_dir,
                action_max_new_tokens=args.action_max_new_tokens,
                summary_max_new_tokens=args.summary_max_new_tokens,
                resume=args.resume,
            )
            families[family] = info
            partial = {
                **manifest,
                "families": families,
                "evaluation": evaluation,
                "peak_gpu_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available() else 0.0
                ),
            }
            _write_json(summary_path, partial)
            if bundle is None:
                print(
                    f"[NO QUALITY SIGNAL] {family}: {info.get('reason')}",
                    flush=True,
                )
                continue

            print(
                f"=== Evaluation: {family} (mixed groups={bundle.mixed_group_count}) ===",
                flush=True,
            )
            new_records = _evaluate_family(
                runtime=runtime,
                env=env,
                family=family,
                task_type=registry[family],
                spec=spec,
                modes=modes,
                base_seed=args.seed,
                model=model,
                processor=processor,
                bundle=bundle,
                output_dir=output_dir,
                action_max_new_tokens=args.action_max_new_tokens,
                summary_max_new_tokens=args.summary_max_new_tokens,
                resume=args.resume,
                existing=existing_eval,
            )
            # Replace, rather than duplicate, any resumed records.
            for record in new_records:
                key = (
                    str(record["family"]),
                    str(record["instance_id"]),
                    str(record["mode"]),
                )
                existing_eval[key] = record
            evaluation = list(existing_eval.values())
            _write_json(
                summary_path,
                {
                    **manifest,
                    "families": families,
                    "evaluation": evaluation,
                    "peak_gpu_gib": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if torch.cuda.is_available() else 0.0
                    ),
                },
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    final = {
        **manifest,
        "families": families,
        "evaluation": list(existing_eval.values()),
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else 0.0
        ),
    }
    _write_json(summary_path, final)
    print(f"Saved {summary_path}", flush=True)
    print(
        "Analyze with: python scripts/analyze_androidworld_latentskill.py "
        f"{summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
