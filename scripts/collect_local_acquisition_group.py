#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.pool import ImmutableTrajectoryPool
from acquisition.schema import (
    AcquisitionGroup,
    ContextKey,
    RolloutMetadata,
    stable_id,
)
from evaluation.parser_validity import VoyagerActionParser
from evaluation.peam_suite import PEAM_TASKS, verify_events
from gr_ktc.generation import generate_with_final_kv
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    task = next(task for task in PEAM_TASKS if task.task_id == "T1")
    context = ContextKey(
        task_text=task.instruction,
        scene_id="local_deterministic_craft_table",
        seed=42,
        inventory=(("oak_planks", 4),),
        biome="dark_forest",
        time_bucket="day",
    )
    group = AcquisitionGroup.create(
        context,
        setup_commands=("/give @s minecraft:oak_planks 4",),
        milestones=({
            "milestone_id": "crafting_table",
            "rules": [{"type": "inventory_has", "params": {
                "item": "crafting_table", "min_count": 1,
            }}],
        },),
        rollout_count=4,
        source="local_voyager",
    )
    pool_path = ROOT / "data/acquisition/local_t1_k4"
    if pool_path.exists():
        raise RuntimeError(f"immutable pool already exists: {pool_path}")
    pool = ImmutableTrajectoryPool(pool_path)
    pool.add_group(group)

    client = VoyagerHTTPClient(timeout_seconds=60)
    parser = VoyagerActionParser(ROOT / "third_party/voyager/voyager/env/mineflayer")
    programs = primitive_programs()
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    layers = [(2 * model.config.text_config.num_hidden_layers) // 3,
              model.config.text_config.num_hidden_layers - 1]

    scores = []
    for rollout_index, generation_seed in enumerate((100, 101, 102, 103)):
        torch.manual_seed(generation_seed)
        events = client.reset(
            hard=True, inventory={"oak_planks": 4}, kill_on_hard_reset=False
        )
        observation = final_observation(events)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {compact_observation(observation)}\nTask: {task.instruction}"
            )}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        started = time.perf_counter()
        generated = generate_with_final_kv(
            model, inputs, layer_ids=layers, max_new_tokens=128,
            temperature=0.7, top_p=0.9,
        )
        response = processor.tokenizer.decode(
            generated.all_generated_token_ids[0], skip_special_tokens=True
        )
        parsed = None
        try:
            parsed = parser.parse(response)
        except ValueError:
            pass
        result_events = events
        if parsed is not None:
            result_events = client.step(
                code=parsed.exec_code,
                programs=programs + "\n\n" + parsed.program_code,
            )
        success = bool(parsed is not None and verify_events(task, result_events))
        scores.append(float(success))
        trajectory_id = stable_id("traj", {
            "group_id": group.group_id,
            "rollout_index": rollout_index,
            "generation_seed": generation_seed,
        })
        metadata = RolloutMetadata(
            trajectory_id=trajectory_id,
            group_id=group.group_id,
            rollout_index=rollout_index,
            seed=generation_seed,
            verifier_score=float(success),
            parser_valid=parsed is not None,
            task_success=success,
            generated_tokens=int(generated.all_generated_token_ids.shape[-1]),
            latency_seconds=time.perf_counter() - started,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            selected_layers=tuple(layers),
            terminal_state={
                "inventory": final_observation(result_events).get("inventory", {}),
                "response": response,
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
            "rollout": rollout_index, "seed": generation_seed,
            "parser_valid": parsed is not None, "success": success,
        }))

    manifest = pool.finalize()
    pool.verify()
    report = {
        "protocol": "local-voyager-acquisition-v1",
        "group_id": group.group_id,
        "rollouts": 4,
        "scores": scores,
        "mixed_outcome": min(scores) != max(scores),
        "files": len(manifest["files"]),
        "pool": str(pool_path.relative_to(ROOT)),
    }
    (ROOT / "results/local_acquisition_group.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
