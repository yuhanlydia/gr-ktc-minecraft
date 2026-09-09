#!/usr/bin/env python3
"""Run prompt-only decommitment on the official MineExplorer environment.

Scientific intervention: prompt semantics only.  The script does not modify
attention masks, KV caches, model weights, MineExplorer actions, milestones, or
world dynamics.  Four conditions share the same local Qwen3-VL-8B model:

  base              official prompt behavior
  reflection        generic reconsideration instruction only
  decommit_ignore   decommitment prompt/output schema, but revocations ignored
  decommit          discrete event revocations are persisted in later prompts

The decommit_ignore condition is the critical mechanism control: it matches the
longer prompt and five-key JSON format without granting behavioral decommitment.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.local_qwen_vl_provider import LocalQwenVLProvider
from gr_ktc.prompt_decommitment import (
    DecommitmentState,
    PromptMode,
    build_control_instruction,
    parse_control_response,
)


def _install_mineexplorer(root: Path) -> None:
    root = root.resolve()
    if not (root / "mc_agent" / "agent.py").exists():
        raise FileNotFoundError(f"official MineExplorer checkout not found at {root}")
    for path in (root, root / "benchmark_gen"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _imports():
    from benchmark_gen.milestone_checker import MilestoneChecker
    from env.minerl_sandbox import MineRLSandboxEnv
    from env.render import RenderWrapper
    from mc_agent.action_space import MinerRLActionSpace
    from mc_agent.context import DefaultContextBuilder
    from mc_agent.utils import convert_buffer_to_base64_images
    return (
        MilestoneChecker,
        MineRLSandboxEnv,
        RenderWrapper,
        MinerRLActionSpace,
        DefaultContextBuilder,
        convert_buffer_to_base64_images,
    )


def _key_action(action: dict) -> dict:
    return {
        key: value for key, value in action.items()
        if value not in (0, 0.0, [0, 0], [0.0, 0.0], None, False)
    }


class PromptDecommitmentAgent:
    def __init__(
        self, action_space, provider, context_builder_class, image_converter, mode: PromptMode,
        *, max_new_tokens: int = 192, temperature: float = 0.7,
    ):
        self.action_space = action_space
        self.provider = provider
        self.context_builder_class = context_builder_class
        self.image_converter = image_converter
        self.mode = PromptMode(mode)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.task_desc = ""
        self.control = DecommitmentState(
            ignore_revocations=self.mode is PromptMode.DECOMMIT_IGNORE
        )
        self.control_records: list[dict[str, Any]] = []

    def load_system_prompt(self, task_desc: str) -> None:
        self.task_desc = task_desc

    def get_default_action(self):
        action_state = self.action_space.load_default_action()
        return "I will stay still because action parsing failed.", self.action_space.dump_action_to_dict(action_state)

    def _system_text(self, long_term_memory: str) -> str:
        text = self.context_builder_class.system_prompt(
            self.task_desc, long_term_memory=long_term_memory
        ).build()
        instruction = build_control_instruction(self.mode)
        if instruction:
            text += "\n\n## Prompt-control protocol\n" + instruction
        if self.mode in {PromptMode.DECOMMIT, PromptMode.DECOMMIT_IGNORE}:
            text += (
                "\n\n## Decision-event ledger\n"
                + self.control.render_ledger(max_events=16)
                + "\nREVOKED means only the old thought/action commitment is invalid; "
                  "historical images remain observations and the same action may be valid later."
            )
        return text

    def get_action(
        self,
        frame_buffer,
        thought_history,
        action_history,
        current_step,
        *,
        long_term_memory="",
    ):
        images = self.image_converter(frame_buffer)
        if not images:
            thought, action = self.get_default_action()
            return thought, action, long_term_memory

        content = [{"type": "text", "text": self._system_text(long_term_memory)}]
        for index, image in enumerate(images):
            frame_step = current_step - (len(images) - 1 - index)
            hist_index = frame_step - 2
            if frame_step > 1 and hist_index < len(action_history) and hist_index < len(thought_history):
                frame_text = self.context_builder_class.next_step(
                    images_idx=index,
                    total_images_length=len(images),
                    frame_step=frame_step,
                    hist_action=action_history[hist_index],
                    hist_thought=thought_history[hist_index],
                ).build()
                frame_text += f"\n  Decision event ID for that prior commitment: E{frame_step - 1}"
            else:
                frame_text = self.context_builder_class.next_step(
                    images_idx=index,
                    total_images_length=len(images),
                    frame_step=frame_step,
                ).build()
            content.append({"type": "text", "text": frame_text})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})

        messages = [{"role": "user", "content": content}]
        response_content = None
        for _attempt in range(3):
            try:
                response_content = self.provider.chat(
                    messages, max_tokens=self.max_new_tokens, temperature=self.temperature
                )
                action_state = self.action_space.load_action(response_content)
                break
            except Exception:
                response_content = None
                time.sleep(0.1)
        if response_content is None:
            thought, action = self.get_default_action()
            return thought, action, long_term_memory

        action = self.action_space.dump_action_to_dict(action_state)
        thought = action_state.think
        memory_update = action_state.memory_update

        proposed: tuple[str, ...] = ()
        applied: tuple[str, ...] = ()
        current_belief = ""
        if self.mode in {PromptMode.DECOMMIT, PromptMode.DECOMMIT_IGNORE}:
            try:
                parsed = parse_control_response(response_content)
                proposed = parsed.revoke_event_ids
                current_belief = parsed.current_belief
                applied = self.control.apply_revocations(
                    proposed, current_step=current_step
                )
            except ValueError:
                pass

        self.control.observe_decision(step=current_step, thought=thought, action=action)
        self.control_records.append({
            "step": current_step,
            "proposed_revocations": list(proposed),
            "applied_revocations": list(applied),
            "revoked_total": list(self.control.revoked_event_ids),
            "current_belief": current_belief,
        })
        return thought, action, memory_update


def _benchmark_env_class(base_cls):
    class BenchmarkEnv(base_cls):
        def __init__(self, metadata_path: str):
            self._metadata_path = Path(metadata_path)
            self._metadata = json.loads(self._metadata_path.read_text())
            self._commands_list = self._metadata.get("commands", [])
            self._task_text = self._metadata.get("task_text", "")
            self._scene_name = self._metadata.get("scene_name", "benchmark_scene")
            super().__init__(env_id=self._scene_name, use_friday=False)

        def _init_remote_env(self):
            response = self.create_env(
                env="MinecraftSim",
                obs_size=[128, 128],
                render_size=[640, 360],
                seed=0,
                record=False,
                record_path="./output/",
                yaml_config=None,
                commands=self._commands_list,
                task_text=self._task_text,
                call_timeout=120,
            )
            if response.get("status") != 0:
                raise RuntimeError(f"create_env failed: {response.get('msg')}")
            self.task = response.get("task_text", "") or self._task_text
            time.sleep(10)

    return BenchmarkEnv


def _scenario_entries(
    benchmark_dir: Path,
    limit: int | None,
    *,
    per_hop: int | None = None,
    hop_counts: tuple[int, ...] = (1, 2, 3, 4),
) -> list[tuple[str, Path, int]]:
    entries: list[tuple[str, Path, int]] = []
    for scene_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
        meta = scene_dir / "multi-agent" / "metadata.json"
        if not meta.exists():
            continue
        payload = json.loads(meta.read_text())
        hop = len(payload.get("milestones", []))
        if hop in hop_counts:
            entries.append((scene_dir.name, meta, hop))

    if per_hop is not None:
        selected: list[tuple[str, Path, int]] = []
        for hop in hop_counts:
            selected.extend([entry for entry in entries if entry[2] == hop][:per_hop])
        entries = selected
    if limit is not None:
        entries = entries[:limit]
    return entries


def run_episode(
    *,
    metadata_path: Path,
    scene_id: str,
    mode: PromptMode,
    provider,
    output_dir: Path,
    max_steps: int,
    frame_size: int,
    loading_command_steps: int,
    max_new_tokens: int,
    temperature: float,
    episode_seed: int,
    imported,
) -> dict[str, Any]:
    MilestoneChecker, MineRLSandboxEnv, RenderWrapper, MinerRLActionSpace, DefaultContextBuilder, image_converter = imported
    BenchmarkEnv = _benchmark_env_class(MineRLSandboxEnv)
    metadata = json.loads(metadata_path.read_text())
    task_desc = metadata.get("task_text", f"Complete MineExplorer scene {scene_id}.")
    checker = MilestoneChecker.from_metadata(metadata_path)
    base_env = BenchmarkEnv(str(metadata_path))
    env = RenderWrapper(base_env, save_messages=False, save_path=str(output_dir))
    provider.set_seed(episode_seed)
    agent = PromptDecommitmentAgent(
        MinerRLActionSpace(), provider, DefaultContextBuilder, image_converter, mode,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    agent.load_system_prompt(task_desc)
    frame_buffer = deque(maxlen=frame_size)
    thought_history: list[str] = []
    action_history: list[dict] = []
    long_term_memory = ""
    frame_completed: dict[str, int] = {}
    presatisfied: set[str] = set()
    action_trace: list[dict[str, Any]] = []

    obs, info = env.reset(save_frame=loading_command_steps == 0)
    frame_buffer.append(obs["pov"])
    for loading_step in range(loading_command_steps):
        _, noop = agent.get_default_action()
        obs, _, _, _, info = env.step(noop, save_frame=loading_step + 1 == loading_command_steps)
    if not info.get("player_pos"):
        _, noop = agent.get_default_action()
        obs, _, _, _, info = env.step(noop, save_frame=True)
        frame_buffer.append(obs["pov"])

    checker.reset(info)
    for ms in checker.check(info):
        if ms.get("completed") and not ms.get("no_milestone"):
            presatisfied.add(ms["milestone_id"])
    checker.reset(info)
    frame_offset = env.frame_count

    started = time.perf_counter()
    step = -1
    try:
        for step in range(max_steps):
            thought, action, memory_update = agent.get_action(
                list(frame_buffer), thought_history, action_history, step + 1,
                long_term_memory=long_term_memory,
            )
            if memory_update and memory_update.strip():
                long_term_memory = memory_update.strip()
            thought_history.append(thought)
            action_history.append(action)
            before = checker.num_completed()
            augmented = checker.augment_action_with_queries(action, info)
            obs, _, terminated, truncated, info = env.step(augmented)
            frame_buffer.append(obs["pov"])
            status = checker.check(info)
            for ms in status:
                mid = ms["milestone_id"]
                if mid in presatisfied:
                    continue
                if ms.get("completed") and mid not in frame_completed:
                    frame_completed[mid] = env.frame_count - frame_offset
            after = checker.num_completed()
            action_trace.append({
                "step": step + 1,
                "thought": thought,
                "action": _key_action(action),
                "milestones_before": before,
                "milestones_after": after,
            })
            if terminated or truncated or (after > 0 and checker.all_done()):
                break
    finally:
        env.close()

    trackable = {
        ms.get("milestone_id", "") for ms in checker._milestones
        if len(ms.get("rules", [])) > 0
    }
    completed = sum(
        1 for mid in trackable if mid in frame_completed and mid not in presatisfied
    )
    total = len(trackable)
    result = {
        "protocol": "official-mineexplorer-prompt-decommitment-v1",
        "scene_id": scene_id,
        "mode": mode.value,
        "task": task_desc,
        "hop_count": len(checker._milestones),
        "total_steps": step + 1 if step >= 0 else 0,
        "milestones_completed": completed,
        "milestones_trackable": total,
        "msr": completed / total if total else 0.0,
        "task_success": bool(total and completed == total),
        "elapsed_seconds": time.perf_counter() - started,
        "control_records": agent.control_records,
        "action_trace": action_trace,
        "peak_gpu_gib": provider.peak_gpu_gib(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mineexplorer-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/prompt_decommitment")
    parser.add_argument(
        "--modes", nargs="+",
        choices=[mode.value for mode in PromptMode],
        default=["base", "reflection", "decommit_ignore", "decommit"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-hop", type=int)
    parser.add_argument("--hop-counts", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--frame-size", type=int, default=20)
    parser.add_argument("--loading-command-steps", type=int, default=20)
    parser.add_argument("--gpu-memory-gib", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _install_mineexplorer(args.mineexplorer_root)
    imported = _imports()
    provider = LocalQwenVLProvider(
        args.model_path, gpu_memory_gib=args.gpu_memory_gib, precision="nf4"
    )
    entries = _scenario_entries(
        args.benchmark_dir, args.limit, per_hop=args.per_hop,
        hop_counts=tuple(args.hop_counts),
    )
    if not entries:
        raise RuntimeError(f"no MineExplorer metadata found under {args.benchmark_dir}")

    all_results = []
    for scene_index, (scene_id, metadata_path, _hop) in enumerate(entries):
        for raw_mode in args.modes:
            mode = PromptMode(raw_mode)
            result = run_episode(
                metadata_path=metadata_path,
                scene_id=scene_id,
                mode=mode,
                provider=provider,
                output_dir=args.output_dir / mode.value / scene_id,
                max_steps=args.max_steps,
                frame_size=args.frame_size,
                loading_command_steps=args.loading_command_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                episode_seed=args.seed + scene_index * 1000,
                imported=imported,
            )
            all_results.append(result)
            print(json.dumps({
                "scene": scene_id,
                "mode": mode.value,
                "msr": result["msr"],
                "success": result["task_success"],
                "steps": result["total_steps"],
                "peak_gpu_gib": result["peak_gpu_gib"],
            }), flush=True)

    summary = {
        "protocol": "official-mineexplorer-prompt-decommitment-v1",
        "model": str(args.model_path),
        "frame_size": args.frame_size,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "results": all_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
