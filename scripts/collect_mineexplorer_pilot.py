#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from acquisition.pool import ImmutableTrajectoryPool
from acquisition.schema import RolloutMetadata, stable_id
from evaluation.mineexplorer_verifier import VerifierState, verify_milestones
from evaluation.parser_validity import VoyagerActionParser
from gr_ktc.generation import generate_with_final_kv
from gr_ktc.group_advantage import group_relative_advantage
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def eligible(scenario, difficulty: str) -> bool:
    rules = [rule for milestone in scenario.milestones for rule in milestone["rules"]]
    # The local no-API executor currently exposes mining/crafting helpers only.
    # Exclude tasks whose required state transition cannot be expressed through
    # that action grammar; keeping them as negatives would confound policy
    # quality with a missing actuator and creates avoidable Mineflayer timeouts.
    unsupported = (
        "trade", "sell ", "chest", "eat ", "drink ", "villager",
        "furnace", "smelt", "cook ", "shear", "breed", "tame ",
        "potted", "place ", "equip ", "wear ", "shoot ", "kill ",
    )
    expected_count = 1 if difficulty == "simple" else range(2, 5)
    return (
        (len(scenario.milestones) == expected_count if difficulty == "simple"
         else len(scenario.milestones) in expected_count)
        and bool(rules)
        and all(rule["type"] == "inventory_has" for rule in rules)
        and len(scenario.commands) <= (6 if difficulty == "simple" else 20)
        and not any(term in scenario.task_text.lower() for term in unsupported)
    )


def _verifier_result(milestones, observation: dict):
    status = observation.get("status", {})
    position = status.get("position", {})
    state = VerifierState(
        inventory=observation.get("inventory", {}),
        player_position=(
            float(position.get("x") or 0), float(position.get("y") or 0),
            float(position.get("z") or 0),
        ),
        player_facing=(0.0, 0.0, 1.0),
        spawn_position=(0.0, 0.0, 0.0),
        objects=(),
    )
    return verify_milestones(milestones, state)


def verifier_score_events(milestones, events: list[list[object]]) -> float:
    latched = {str(milestone["milestone_id"]): False for milestone in milestones}
    for kind, payload in events:
        if kind != "observe" or not isinstance(payload, dict):
            continue
        result = _verifier_result(milestones, payload)
        for milestone_id, passed in result.milestone_results.items():
            latched[milestone_id] = latched[milestone_id] or passed
    return sum(latched.values()) / len(latched) if latched else 0.0


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--groups", type=int, default=5)
    argument_parser.add_argument("--offset", type=int, default=0)
    argument_parser.add_argument(
        "--scene-ids", nargs="+",
        help="Explicit stable scene IDs; overrides --offset/--groups selection.",
    )
    argument_parser.add_argument("--pool", type=Path)
    argument_parser.add_argument(
        "--difficulty", choices=("simple", "composite"), default="simple"
    )
    argument_parser.add_argument("--resume", action="store_true")
    args = argument_parser.parse_args()
    all_scenarios = list(load_mineexplorer(
        ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
    ))
    eligible_scenarios = [
        scenario for scenario in all_scenarios if eligible(scenario, args.difficulty)
    ]
    if args.scene_ids:
        by_id = {scenario.scene_id: scenario for scenario in all_scenarios}
        missing = [scene_id for scene_id in args.scene_ids if scene_id not in by_id]
        if missing:
            raise ValueError(f"unknown scene IDs: {missing}")
        scenarios = [by_id[scene_id] for scene_id in args.scene_ids]
        invalid = [scenario.scene_id for scenario in scenarios
                   if not eligible(scenario, args.difficulty)]
        if invalid:
            raise ValueError(f"scenes are not eligible for {args.difficulty}: {invalid}")
        args.groups = len(scenarios)
    else:
        scenarios = eligible_scenarios[args.offset : args.offset + args.groups]
        if len(scenarios) != args.groups:
            raise ValueError(f"only {len(scenarios)} eligible scenarios")
    pool_path = args.pool or ROOT / (
        f"data/acquisition/mineexplorer_{args.difficulty}_o{args.offset}_g{args.groups}_k4"
    )
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    if pool_path.exists() and not args.resume:
        raise RuntimeError(f"pool path already exists: {pool_path}")
    pool = ImmutableTrajectoryPool(pool_path)
    if pool.finalized:
        raise RuntimeError(f"pool is already finalized: {pool_path}")

    client = VoyagerHTTPClient(timeout_seconds=120)
    action_parser = VoyagerActionParser(ROOT / "third_party/voyager/voyager/env/mineflayer")
    programs = primitive_programs()
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    layers = [(2 * model.config.text_config.num_hidden_layers) // 3,
              model.config.text_config.num_hidden_layers - 1]
    group_reports = []
    composite_system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    active_system = composite_system if args.difficulty == "composite" else SYSTEM

    for group_index, scenario in enumerate(scenarios):
        group = scenario.acquisition_group(seed=42, rollout_count=4)
        group_dir = pool_path / "groups" / group.group_id
        if not group_dir.exists():
            pool.add_group(group)
        existing = {}
        for metadata_path in group_dir.glob("trajectories/*/metadata.json"):
            metadata_record = json.loads(metadata_path.read_text())
            existing[int(metadata_record["rollout_index"])] = metadata_record
        score_by_rollout = {
            index: float(record["verifier_score"])
            for index, record in existing.items()
        }
        for rollout_index, generation_seed in enumerate((100, 101, 102, 103)):
            if rollout_index in existing:
                print(json.dumps({
                    "group": group_index, "scene": scenario.scene_id,
                    "rollout": rollout_index, "resumed": True,
                    "score": score_by_rollout[rollout_index],
                }), flush=True)
                continue
            torch.manual_seed(generation_seed)
            events = client.reset(
                hard=True,
                kill_on_hard_reset=False,
                setup_commands=scenario.commands,
            )
            observation = final_observation(events)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": active_system}]},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"Observation: {compact_observation(observation)}\n"
                    f"Task: {scenario.task_text}"
                )}]},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            started = time.perf_counter()
            generated = generate_with_final_kv(
                model, inputs, layer_ids=layers,
                max_new_tokens=384 if args.difficulty == "composite" else 192,
                temperature=0.9 if args.difficulty == "composite" else 0.7,
                top_p=0.95 if args.difficulty == "composite" else 0.9,
            )
            response = processor.tokenizer.decode(
                generated.all_generated_token_ids[0], skip_special_tokens=True
            )
            parsed = None
            parse_error = None
            try:
                parsed = action_parser.parse(response)
            except ValueError as exc:
                parse_error = str(exc)
            result_events = events
            execution_error = None
            if parsed is not None:
                try:
                    result_events = client.step(
                        code=parsed.exec_code,
                        programs=programs + "\n\n" + parsed.program_code,
                    )
                except Exception as exc:
                    execution_error = f"{type(exc).__name__}: {exc}"
            terminal = final_observation(result_events)
            score = verifier_score_events(scenario.milestones, result_events)
            score_by_rollout[rollout_index] = score
            trajectory_id = stable_id("traj", {
                "group": group.group_id, "rollout": rollout_index,
                "generation_seed": generation_seed,
            })
            metadata = RolloutMetadata(
                trajectory_id=trajectory_id,
                group_id=group.group_id,
                rollout_index=rollout_index,
                seed=generation_seed,
                verifier_score=score,
                parser_valid=parsed is not None,
                task_success=score == 1.0,
                generated_tokens=int(generated.all_generated_token_ids.shape[-1]),
                latency_seconds=time.perf_counter() - started,
                model_id="Qwen/Qwen3-VL-8B-Instruct",
                selected_layers=tuple(layers),
                terminal_state={
                    "inventory": terminal.get("inventory", {}),
                    "response": response,
                    "parse_error": parse_error,
                    "execution_error": execution_error,
                },
            )
            pool.add_rollout(
                metadata,
                tokens=generated.trajectory_token_ids[0],
                kv_by_layer=generated.kv_by_layer,
                execution_events=[
                    {"event": kind, "payload": payload} for kind, payload in result_events
                ],
            )
            print(json.dumps({
                "group": group_index, "scene": scenario.scene_id,
                "rollout": rollout_index, "parser_valid": parsed is not None,
                "score": score, "execution_error": execution_error,
            }), flush=True)
            if execution_error is not None:
                raise RuntimeError(
                    "rollout execution timed out/failed; restart the Mineflayer "
                    "bridge and rerun with --resume to preserve isolation"
                )
        scores = [score_by_rollout[index] for index in range(4)]
        advantages = group_relative_advantage(torch.tensor(scores)).tolist()
        group_reports.append({
            "scene_id": scenario.scene_id,
            "task": scenario.task_text,
            "scores": scores,
            "advantages": advantages,
            "mixed_outcome": min(scores) != max(scores),
        })

    manifest = pool.finalize()
    pool.verify()
    report = {
        "protocol": "mineexplorer-local-pilot-v1",
        "difficulty": args.difficulty,
        "offset": args.offset,
        "requested_scene_ids": args.scene_ids,
        "groups": len(group_reports),
        "trajectories": len(group_reports) * 4,
        "mixed_groups": sum(group["mixed_outcome"] for group in group_reports),
        "successes": sum(sum(score == 1.0 for score in group["scores"]) for group in group_reports),
        "parser_valid": sum(
            1 for path in pool_path.rglob("metadata.json")
            if json.loads(path.read_text())["parser_valid"]
        ),
        "manifest_files": len(manifest["files"]),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "pool": str(pool_path.relative_to(ROOT)),
        "group_results": group_reports,
    }
    output = ROOT / (
        f"results/mineexplorer_{args.difficulty}_o{args.offset}_g{args.groups}_k4.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
