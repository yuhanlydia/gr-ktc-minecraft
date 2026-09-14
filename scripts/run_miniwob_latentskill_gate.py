#!/usr/bin/env python3
"""Run the fast LatentSkill/CLSC reliability gate on MiniWoB++."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

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
from gr_ktc.miniwob_protocol import (
    MODES,
    extract_single_action,
    is_terminal_success,
    phase_spec,
    protocol_seed,
    qualify_signal,
    quality_reward,
)
from gr_ktc.model_loader import load_qwen3_vl_24gb


BROWSERGYM_PIN = "9e779f087de9a65668b6974d11f9ce9816026e96"
MINIWOB_PIN = "7fd85d71a4b60325c6585396ec4f48377d049838"
QUALITY_LAYER = 24
MEMORY_TOKENS = 4
VALUE_SCALE = 0.25
NEGATIVE_SCALE = 0.5
DEFAULT_TASKS = (
    "miniwob.form-sequence",
    "miniwob.choose-date",
    "miniwob.email-inbox",
    "miniwob.login-user-popup",
    "miniwob.social-media-some",
    "miniwob.use-autocomplete",
    "miniwob.navigate-tree",
    "miniwob.book-flight-nodelay",
)


@dataclass(frozen=True)
class GeneratedAction:
    text: str
    kv_by_layer: Mapping[int, torch.Tensor] | None = None


def _model_inputs(model: Any, processor: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
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


def _eos_ids(model: Any, processor: Any) -> tuple[int, ...]:
    values: list[int] = []
    for raw in (
        getattr(processor.tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if isinstance(raw, int):
            values.append(raw)
        elif isinstance(raw, (tuple, list)):
            values.extend(int(item) for item in raw if isinstance(item, int))
    return tuple(dict.fromkeys(values))


class MiniwobQwenPolicy:
    """Generate one BrowserGym action while capturing or injecting native K/V."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 96,
        acquisition_temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.acquisition_temperature = float(acquisition_temperature)
        self.top_p = float(top_p)
        self.layer_ids = list(range(model.config.text_config.num_hidden_layers))
        self.eos_ids = _eos_ids(model, processor)

    def _decode(self, ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        *,
        capture_kv: bool,
        memory: Any,
        seed: int,
    ) -> GeneratedAction:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        inputs = _model_inputs(self.model, self.processor, prompt)
        if capture_kv:
            generated = generate_with_final_kv(
                self.model,
                inputs,
                layer_ids=self.layer_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.acquisition_temperature,
                top_p=self.top_p,
            )
            return GeneratedAction(
                self._decode(generated.all_generated_token_ids),
                generated.kv_by_layer,
            )
        generator = torch.Generator(device=self.model.device).manual_seed(int(seed))
        ids = generate_with_kv_prefix(
            self.model,
            inputs,
            memory,
            context_id=getattr(memory, "context_id", "family:miniwob"),
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=self.top_p,
            eos_token_ids=self.eos_ids,
            generator=generator,
        )
        return GeneratedAction(self._decode(ids), None)


def build_action_prompt(
    goal: str,
    axtree: str,
    history: Sequence[tuple[str, str]],
    action_error: str,
) -> str:
    recent = list(history)[-4:]
    history_text = "\n".join(
        f"Action: {action}\nResulting accessibility tree:\n{tree}"
        for action, tree in recent
    ) or "(none)"
    error_text = action_error.strip() or "(none)"
    return f"""You are controlling a MiniWoB web task through BrowserGym.

Goal:
{goal}

Current accessibility tree (interactive elements have bid identifiers):
{axtree}

Recent history:
{history_text}

Last action error:
{error_text}

Choose exactly one action. Allowed forms are:
click('bid')
fill('bid', 'text')
select_option('bid', 'option')
press('bid', 'key')
clear('bid')
focus('bid')
scroll(0, pixels)
noop()

Respond with a short reason and one final line formatted as:
Action: <one action>
"""


def _goal_fingerprint(goal: str) -> str:
    return hashlib.sha256(str(goal).encode()).hexdigest()


def _tree_text(observation: Mapping[str, Any]) -> str:
    if "axtree_txt" not in observation:
        raise ValueError("preprocessed MiniWoB observation lacks axtree_txt")
    return str(observation["axtree_txt"])


def can_resume_acquisition(result_path: Path, kv_path: Path) -> bool:
    return result_path.is_file() and kv_path.is_file()


def validate_same_fingerprint(records: Sequence[Mapping[str, Any]], label: str) -> str:
    fingerprints = {str(record.get("goal_fingerprint", "")) for record in records}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise RuntimeError(f"{label} contains different randomized goals")
    return next(iter(fingerprints))


def memory_for_mode(bundle: FamilyMemoryBundle, mode: str) -> Any:
    if mode == "base":
        return None
    memories = bundle.memories()
    if mode not in memories:
        raise ValueError(f"unsupported MiniWoB memory mode {mode!r}")
    return memories[mode]


def build_manifest(
    *,
    phase: str,
    tasks: Sequence[str],
    model_path: Path,
    browsergym_head: str,
    miniwob_head: str,
    base_seed: int,
) -> dict[str, Any]:
    spec = phase_spec(phase)
    task_list = list(map(str, tasks))
    return {
        "protocol": "latentskill-miniwob-raw-reward-v1",
        "phase": spec.name,
        "browsergym_pin": BROWSERGYM_PIN,
        "browsergym_head": str(browsergym_head),
        "miniwob_pin": MINIWOB_PIN,
        "miniwob_head": str(miniwob_head),
        "model": str(model_path),
        "precision": "bf16",
        "observation_protocol": "browsergym-accessibility-tree",
        "raw_reward_success_threshold": 0.5,
        "quality_reward": "clip(RAW_REWARD_GLOBAL, 0, 1)",
        "quality_layer": QUALITY_LAYER,
        "memory_tokens": MEMORY_TOKENS,
        "value_scale": VALUE_SCALE,
        "negative_scale": NEGATIVE_SCALE,
        "acquisition_temperature": 0.9,
        "acquisition_top_p": 0.95,
        "evaluation_temperature": 0.0,
        "rollouts_per_instance": spec.rollouts_per_instance,
        "test_instances_per_family": spec.test_instances,
        "target_qualified_families": spec.target_families,
        "minimum_successes": 3,
        "minimum_failures": 3,
        "minimum_reward_std": 0.15,
        "modes": list(spec.modes),
        "tasks": task_list,
        "base_seed": int(base_seed),
        "acquisition_task_seeds": {
            family: protocol_seed("acquisition-task", family, 0, base_seed)
            for family in task_list
        },
        "acquisition_model_seeds": {
            family: [
                protocol_seed("acquisition-model", family, index, base_seed)
                for index in range(spec.rollouts_per_instance)
            ]
            for family in task_list
        },
        "evaluation_task_seeds": {
            family: [
                protocol_seed("evaluation-task", family, index, base_seed)
                for index in range(spec.test_instances)
            ]
            for family in task_list
        },
        "evaluation_model_seeds": {
            family: [
                protocol_seed("evaluation-model", family, index, base_seed)
                for index in range(spec.test_instances)
            ]
            for family in task_list
        },
    }


def run_episode(
    env_factory: Callable[[str, int], Any],
    policy: Any,
    *,
    family: str,
    task_seed: int,
    model_seed: int,
    acquisition: bool,
    mode: str,
    memory: Any,
    max_steps: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor] | None]:
    """Run one seeded episode and trust only MiniWoB's native raw reward."""
    env = env_factory(str(family), int(task_seed))
    started = time.perf_counter()
    history: list[tuple[str, str]] = []
    captured: list[Mapping[int, torch.Tensor]] = []
    parser_valid_count = 0
    action_errors: list[str] = []
    wrapper_reward = 0.0
    final_info: Mapping[str, Any] = {}
    terminated = False
    truncated = False
    try:
        observation, _ = env.reset()
        goal = str(observation.get("goal", ""))
        if not goal:
            raise ValueError("MiniWoB observation has no goal")
        for step_index in range(int(max_steps)):
            tree = _tree_text(observation)
            prompt = build_action_prompt(
                goal,
                tree,
                history,
                str(observation.get("last_action_error", "")),
            )
            generated = policy.generate(
                prompt,
                capture_kv=bool(acquisition),
                memory=memory if not acquisition else None,
                seed=int(model_seed) + step_index,
            )
            try:
                action = extract_single_action(generated.text)
                parser_valid_count += 1
            except ValueError:
                action = "noop()"
            if acquisition and generated.kv_by_layer:
                captured.append(generated.kv_by_layer)
            observation, wrapper_reward, terminated, truncated, final_info = env.step(
                action
            )
            next_tree = _tree_text(observation)
            history.append((action, next_tree))
            error = str(observation.get("last_action_error", "")).strip()
            if error:
                action_errors.append(error)
            if terminated or truncated:
                break

        task_info = final_info.get("task_info", final_info)
        if not isinstance(task_info, Mapping):
            task_info = {}
        raw_reward = task_info.get("RAW_REWARD_GLOBAL")
        acquisition_reward = quality_reward(raw_reward)
        steps = len(history)
        record = {
            "family": str(family),
            "goal": goal,
            "goal_fingerprint": _goal_fingerprint(goal),
            "task_seed": int(task_seed),
            "model_seed": int(model_seed),
            "mode": str(mode),
            "raw_reward": float(raw_reward),
            "acquisition_reward": acquisition_reward,
            "success_threshold": 0.5,
            "success": is_terminal_success(raw_reward),
            "browsergym_wrapper_reward": float(wrapper_reward),
            "reward_reason": str(task_info.get("REWARD_REASON", "")),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps,
            "parser_valid": parser_valid_count == steps,
            "parser_valid_rate": parser_valid_count / steps if steps else 0.0,
            "action_errors": action_errors,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trajectory = concat_step_kv(captured) if acquisition and captured else None
        return record, trajectory
    finally:
        env.close()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


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
    return {
        int(name.removeprefix("layer_")): tensor.contiguous()
        for name, tensor in load_file(str(path)).items()
        if name.startswith("layer_")
    }


def _git_head(path: Path, expected: str, name: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if head != expected:
        raise RuntimeError(f"{name} HEAD is {head}, expected pinned {expected}")
    return head


class _FlattenedBrowserEnv:
    def __init__(
        self, env: Any, flatten: Callable[..., str], *, task_seed: int
    ) -> None:
        self.env = env
        self.flatten = flatten
        self.task_seed = int(task_seed)

    def _convert(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        converted = dict(observation)
        converted["axtree_txt"] = self.flatten(
            observation["axtree_object"],
            extra_properties=observation.get("extra_element_properties"),
            with_visible=True,
            with_clickable=True,
            hide_bid_if_invisible=True,
        )
        return converted

    def reset(self):
        observation, info = self.env.reset(seed=self.task_seed)
        return self._convert(observation), info

    def step(self, action: str):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._convert(observation), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


def _load_browsergym_runtime(
    browsergym_root: Path, miniwob_root: Path
) -> SimpleNamespace:
    browsergym_head = _git_head(browsergym_root, BROWSERGYM_PIN, "BrowserGym")
    miniwob_head = _git_head(miniwob_root, MINIWOB_PIN, "MiniWoB++")
    miniwob_html = miniwob_root / "miniwob" / "html" / "miniwob"
    if not miniwob_html.is_dir():
        raise FileNotFoundError(f"MiniWoB HTML root not found: {miniwob_html}")
    os.environ["MINIWOB_URL"] = miniwob_html.resolve().as_uri() + "/"
    try:
        import gymnasium as gym
        import browsergym.miniwob  # noqa: F401
        from browsergym.core.action.highlevel import HighLevelActionSet
        from browsergym.utils.obs import flatten_axtree_to_str
    except ImportError as exc:
        raise RuntimeError(
            "BrowserGym is not installed; run scripts/bootstrap_miniwob.sh"
        ) from exc

    action_set = HighLevelActionSet(
        subsets=["bid"], strict=False, multiaction=False
    )

    def env_factory(family: str, task_seed: int):
        env = gym.make(
            f"browsergym/{family}",
            task_kwargs={
                "base_url": os.environ["MINIWOB_URL"],
                "episode_max_time": 300_000,
            },
            headless=True,
            action_mapping=action_set.to_python_code,
            pre_observation_delay=0.1,
        )
        return _FlattenedBrowserEnv(
            env, flatten_axtree_to_str, task_seed=int(task_seed)
        )

    return SimpleNamespace(
        browsergym_head=browsergym_head,
        miniwob_head=miniwob_head,
        env_factory=env_factory,
    )


def _build_skill(
    family: str,
    records: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[int, torch.Tensor]],
    model: Any,
    output_dir: Path,
) -> FamilyMemoryBundle:
    text_config = model.config.text_config
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    bundle = build_family_memory_bundle(
        family_id=family,
        trajectories=trajectories,
        rewards=[float(record["acquisition_reward"]) for record in records],
        successes=[bool(record["success"]) for record in records],
        group_ids=[f"{family}:acq:0"] * len(records),
        kv_heads=text_config.num_key_value_heads,
        head_dim=head_dim,
        quality_layer=QUALITY_LAYER,
        memory_tokens=MEMORY_TOKENS,
        value_scale=VALUE_SCALE,
        negative_scale=NEGATIVE_SCALE,
    )
    skill_dir = output_dir / "skills"
    save_family_memory_bundle(
        bundle,
        skill_dir / f"{family}.safetensors",
        skill_dir / f"{family}.json",
    )
    return bundle


def _acquire_family(
    *,
    family: str,
    manifest: Mapping[str, Any],
    output_dir: Path,
    env_factory: Callable[[str, int], Any],
    policy: MiniwobQwenPolicy,
    model: Any,
    max_steps: int,
    resume: bool,
) -> tuple[dict[str, Any], FamilyMemoryBundle | None]:
    records: list[dict[str, Any]] = []
    trajectories: list[dict[int, torch.Tensor]] = []
    task_seed = int(manifest["acquisition_task_seeds"][family])
    for rollout_index, model_seed in enumerate(
        manifest["acquisition_model_seeds"][family]
    ):
        run_dir = output_dir / "acquisition" / family / f"rollout_{rollout_index:02d}"
        result_path = run_dir / "result.json"
        kv_path = run_dir / "episode_kv.safetensors"
        if resume and can_resume_acquisition(result_path, kv_path):
            record = json.loads(result_path.read_text())
            trajectory = _load_episode_kv(kv_path)
        else:
            record, trajectory = run_episode(
                env_factory,
                policy,
                family=family,
                task_seed=task_seed,
                model_seed=int(model_seed),
                acquisition=True,
                mode="base",
                memory=None,
                max_steps=max_steps,
            )
            record.update(
                {
                    "split": "acquisition",
                    "instance_id": "acq-00",
                    "rollout_index": rollout_index,
                    "group_id": f"{family}:acq:0",
                }
            )
            _write_json(result_path, record)
            if trajectory is not None:
                _save_episode_kv(kv_path, trajectory)
        records.append(record)
        if trajectory is not None:
            trajectories.append(trajectory)

    validate_same_fingerprint(records, f"{family} acquisition group")
    if len(trajectories) != len(records):
        raise RuntimeError(f"{family} acquisition group lacks K/V trajectories")
    qualification = qualify_signal(
        [record["acquisition_reward"] for record in records]
    )
    if manifest["phase"] == "smoke":
        successes = sum(bool(record["success"]) for record in records)
        failures = len(records) - successes
        smoke_qualified = successes >= 1 and failures >= 1
        qualification = SimpleNamespace(
            qualified=smoke_qualified,
            successes=successes,
            failures=failures,
            reward_std=qualification.reward_std,
            reason="qualified" if smoke_qualified else "smoke requires one success and one failure",
        )
    info = {
        "family": family,
        "status": "qualified" if qualification.qualified else "no_quality_signal",
        "qualification": {
            "successes": qualification.successes,
            "failures": qualification.failures,
            "reward_std": qualification.reward_std,
            "reason": qualification.reason,
        },
        "acquisition": records,
    }
    if not qualification.qualified:
        return info, None
    bundle = _build_skill(family, records, trajectories, model, output_dir)
    info["skill_metadata"] = str(output_dir / "skills" / f"{family}.json")
    info["mixed_group_count"] = bundle.mixed_group_count
    return info, bundle


def _evaluate_family(
    *,
    family: str,
    manifest: Mapping[str, Any],
    output_dir: Path,
    env_factory: Callable[[str, int], Any],
    policy: MiniwobQwenPolicy,
    bundle: FamilyMemoryBundle,
    max_steps: int,
    resume: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (task_seed, model_seed) in enumerate(
        zip(
            manifest["evaluation_task_seeds"][family],
            manifest["evaluation_model_seeds"][family],
            strict=True,
        )
    ):
        instance_records: list[dict[str, Any]] = []
        for mode in manifest["modes"]:
            path = output_dir / "eval" / mode / family / f"eval_{index:02d}.json"
            if resume and path.is_file():
                record = json.loads(path.read_text())
            else:
                record, _ = run_episode(
                    env_factory,
                    policy,
                    family=family,
                    task_seed=int(task_seed),
                    model_seed=int(model_seed),
                    acquisition=False,
                    mode=mode,
                    memory=memory_for_mode(bundle, mode),
                    max_steps=max_steps,
                )
                record.update({"split": "evaluation", "instance_id": f"eval-{index:02d}"})
                _write_json(path, record)
            instance_records.append(record)
            records.append(record)
        validate_same_fingerprint(instance_records, f"{family} eval-{index:02d}")
        if len({int(record["model_seed"]) for record in instance_records}) != 1:
            raise RuntimeError(f"{family} eval-{index:02d} modes use different model seeds")
    return records


def run_gate(args: argparse.Namespace) -> Path:
    spec = phase_spec(args.phase)
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    if args.phase == "smoke" and not args.tasks:
        tasks = ("miniwob.click-dialog-2",)
    browsergym_root = args.browsergym_root.expanduser().resolve()
    miniwob_root = args.miniwob_root.expanduser().resolve()
    runtime = _load_browsergym_runtime(browsergym_root, miniwob_root)
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {model_path}")
    output_dir = args.output_dir.expanduser().resolve() / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        phase=args.phase,
        tasks=tasks,
        model_path=model_path,
        browsergym_head=runtime.browsergym_head,
        miniwob_head=runtime.miniwob_head,
        base_seed=args.seed,
    )
    _write_json(output_dir / "manifest.json", manifest)
    model, processor = load_qwen3_vl_24gb(model_path, precision="bf16")
    model.eval()
    policy = MiniwobQwenPolicy(
        model, processor, max_new_tokens=args.action_max_new_tokens
    )
    families: dict[str, Any] = {}
    evaluation: list[dict[str, Any]] = []
    qualified = 0
    summary_path = output_dir / "summary.json"
    for family in tasks:
        print(f"\n=== Signal scan: {family} ===", flush=True)
        info, bundle = _acquire_family(
            family=family,
            manifest=manifest,
            output_dir=output_dir,
            env_factory=runtime.env_factory,
            policy=policy,
            model=model,
            max_steps=args.max_steps,
            resume=args.resume,
        )
        families[family] = info
        if bundle is not None:
            qualified += 1
            print(f"[QUALIFIED] {family}", flush=True)
            evaluation.extend(
                _evaluate_family(
                    family=family,
                    manifest=manifest,
                    output_dir=output_dir,
                    env_factory=runtime.env_factory,
                    policy=policy,
                    bundle=bundle,
                    max_steps=args.max_steps,
                    resume=args.resume,
                )
            )
        else:
            print(f"[NO QUALITY SIGNAL] {family}: {info['qualification']['reason']}", flush=True)
        _write_json(
            summary_path,
            {
                **manifest,
                "qualified_families": qualified,
                "families": families,
                "evaluation": evaluation,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0,
            },
        )
        if qualified >= spec.target_families:
            break
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "quick"), default="smoke")
    parser.add_argument(
        "--browsergym-root", type=Path, default=ROOT / "third_party/browsergym"
    )
    parser.add_argument(
        "--miniwob-root", type=Path, default=ROOT / "third_party/miniwob-plusplus"
    )
    parser.add_argument(
        "--model-path", type=Path, default=ROOT / "models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/latentskill_miniwob"
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--action-max-new-tokens", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_gate(args)
    print(f"Saved {summary}", flush=True)
    print(
        "Analyze with: python scripts/analyze_miniwob_latentskill.py " + str(summary),
        flush=True,
    )


if __name__ == "__main__":
    main()
