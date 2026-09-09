#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parser_validity import VoyagerActionParser
from evaluation.peam_suite import PEAM_TASKS, verify_events
from gr_ktc.generation import generate_with_final_kv
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation, primitive_programs


@dataclass(frozen=True)
class SmokeScene:
    task_id: str
    generation_seed: int
    inventory: dict[str, int]
    setup_commands: tuple[str, ...] = ()


SCENES = (
    SmokeScene("T1", 42, {"oak_planks": 4}),
    SmokeScene("T2", 42, {"oak_planks": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T3", 42, {"cobblestone": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T4", 42, {"cobblestone": 8}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T5", 42, {"iron_ingot": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T1", 43, {"oak_planks": 4}),
    SmokeScene("T2", 43, {"oak_planks": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T3", 43, {"cobblestone": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T4", 43, {"cobblestone": 8}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
    SmokeScene("T5", 43, {"iron_ingot": 3, "stick": 2}, ("/setblock ~1 ~ ~ minecraft:crafting_table",)),
)


SYSTEM = """You control a Minecraft 1.19 Mineflayer bot. Return exactly one fenced
JavaScript block containing one complete async function with exactly one argument
named bot. Use only exploreUntil, mineBlock, craftItem, placeItem, smeltItem, and
killMob. Never use chat commands. Prefer one direct helper call with the requested
count. mineBlock already searches and mines. Keep the answer under 100 tokens.
Required form: ```javascript
async function solveTask(bot) {
  await mineBlock(bot, "stone", 1);
}
```"""


def main() -> None:
    output = ROOT / "results/local_smoke_10.json"
    kv_dir = ROOT / "results/local_smoke_10_kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    client = VoyagerHTTPClient(timeout_seconds=60)
    action_parser = VoyagerActionParser(ROOT / "third_party/voyager/voyager/env/mineflayer")
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    layer_count = model.config.text_config.num_hidden_layers
    layer_ids = [(2 * layer_count) // 3, layer_count - 1]
    programs = primitive_programs()
    records = []

    for index, scene in enumerate(SCENES):
        task = next(task for task in PEAM_TASKS if task.task_id == scene.task_id)
        torch.manual_seed(scene.generation_seed)
        events = client.reset(
            hard=True, inventory=scene.inventory, kill_on_hard_reset=False,
            setup_commands=scene.setup_commands,
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
            model, inputs, layer_ids=layer_ids, max_new_tokens=128,
            temperature=0.7, top_p=0.9,
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
        if parsed is not None:
            result_events = client.step(
                code=parsed.exec_code,
                programs=programs + "\n\n" + parsed.program_code,
            )
        success = parsed is not None and verify_events(task, result_events)
        record = {
            "index": index,
            "task_id": task.task_id,
            "task": task.instruction,
            "generation_seed": scene.generation_seed,
            "parser_valid": parsed is not None,
            "task_success": success,
            "parse_error": parse_error,
            "response": response,
            "generated_tokens": int(generated.all_generated_token_ids.shape[-1]),
            "kv_tokens": int(generated.trajectory_token_ids.shape[-1]),
            "elapsed_seconds": time.perf_counter() - started,
            "final_inventory": final_observation(result_events).get("inventory", {}),
        }
        records.append(record)
        save_file(
            {f"layer_{layer}": value for layer, value in generated.kv_by_layer.items()},
            str(kv_dir / f"{index:02d}_{task.task_id}_seed{scene.generation_seed}.safetensors"),
        )
        output.write_text(
            json.dumps({"protocol": "local-smoke-v1", "records": records}, indent=2) + "\n"
        )
        print(json.dumps({key: record[key] for key in (
            "index", "task_id", "parser_valid", "task_success", "elapsed_seconds"
        )}))

    summary = {
        "protocol": "local-smoke-v1",
        "scope_warning": "Deterministic prerequisite smoke scenes; not PEAM held-out results.",
        "scenes": len(records),
        "parser_valid": sum(record["parser_valid"] for record in records),
        "successes": sum(record["task_success"] for record in records),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "records": records,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "scenes", "parser_valid", "successes", "peak_gpu_gib"
    )}, indent=2))


if __name__ == "__main__":
    main()
