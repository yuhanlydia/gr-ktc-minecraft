#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parser_validity import VoyagerActionParser
from gr_ktc.generation import generate_with_final_kv
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation


MINEFLAYER = ROOT / "third_party/voyager/voyager/env/mineflayer"


def compact_observation(observation: dict) -> str:
    status = observation.get("status", {})
    return json.dumps(
        {
            "nearby_blocks": observation.get("voxels", []),
            "inventory": observation.get("inventory", {}),
            "biome": status.get("biome"),
            "time": status.get("timeOfDay"),
            "health": status.get("health"),
            "food": status.get("food"),
            "nearby_entities": status.get("entities", {}),
        },
        ensure_ascii=False,
    )


def primitive_programs() -> str:
    directory = ROOT / "third_party/voyager/voyager/control_primitives"
    names = ("exploreUntil", "mineBlock", "craftItem", "placeItem", "smeltItem", "killMob")
    programs = "\n\n".join((directory / f"{name}.js").read_text() for name in names)
    # MineExplorer milestones are latched across decision steps. A generated
    # Voyager program may execute several helpers in one /step, so expose the
    # same intermediate states after every primitive completes.
    wrappers = []
    for name in names:
        wrappers.append(
            f"const __grktc_{name} = {name};\n"
            f"{name} = async (...args) => {{\n"
            f"  const result = await __grktc_{name}(...args);\n"
            f"  if (args[0] && args[0].event) args[0].event('observe');\n"
            f"  return result;\n"
            f"}};"
        )
    return programs + "\n\n" + "\n\n".join(wrappers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Collect 4 oak logs")
    parser.add_argument("--precision", choices=("bf16", "int8", "nf4"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local_qwen_action.json")
    args = parser.parse_args()

    client = VoyagerHTTPClient()
    health = client.health()
    if not health.get("botSpawned"):
        events = client.reset(hard=True, kill_on_hard_reset=False)
    else:
        # A no-op obtains the current authoritative observation.
        events = client.step(code="await bot.waitForTicks(1);")
    observation = final_observation(events)

    system = (
        "You control a Minecraft 1.19 Mineflayer bot. Return exactly one fenced "
        "JavaScript block containing one complete async function with exactly one "
        "argument named bot. Use only these helpers: exploreUntil(bot, direction, "
        "maxTime, callback), mineBlock(bot, name, count), craftItem(bot, name, count), "
        "placeItem(bot, name, position), smeltItem(bot, itemName, fuelName, count), "
        "killMob(bot, mobName, timeout). Do not use chat commands. Prefer one direct "
        "helper call with the requested count; mineBlock already searches and mines, "
        "so do not wrap it in repeated exploration or bot.findBlock calls. Keep the "
        "entire answer under 100 tokens. Required form example: ```javascript\n"
        "async function solveTask(bot) {\n  await mineBlock(bot, \"stone\", 1);\n}\n```"
    )
    user = f"Observation: {compact_observation(observation)}\nTask: {args.task}"

    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision=args.precision
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    layer_count = model.config.text_config.num_hidden_layers
    layer_ids = [(2 * layer_count) // 3, layer_count - 1]
    started = time.perf_counter()
    generated = generate_with_final_kv(
        model,
        inputs,
        layer_ids=layer_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    elapsed = time.perf_counter() - started
    response = processor.tokenizer.decode(
        generated.all_generated_token_ids[0], skip_special_tokens=True
    )

    action_parser = VoyagerActionParser(MINEFLAYER)
    parsed = None
    parse_error = None
    try:
        parsed = action_parser.parse(response)
    except ValueError as exc:
        parse_error = str(exc)

    post_observation = None
    if args.execute and parsed is not None:
        post_events = client.step(
            code=parsed.exec_code,
            programs=primitive_programs() + "\n\n" + parsed.program_code,
        )
        post_observation = final_observation(post_events)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": "local-qwen-voyager-v1",
        "task": args.task,
        "precision": args.precision,
        "elapsed_seconds": elapsed,
        "prompt_tokens": generated.prompt_tokens,
        "generated_tokens": int(generated.all_generated_token_ids.shape[-1]),
        "kv_tokens": int(generated.trajectory_token_ids.shape[-1]),
        "layers": layer_ids,
        "response": response,
        "parser_valid": parsed is not None,
        "parse_error": parse_error,
        "executed": post_observation is not None,
        "post_inventory": post_observation.get("inventory", {}) if post_observation else None,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save_file(
        {f"layer_{layer}": value for layer, value in generated.kv_by_layer.items()},
        str(args.output.with_suffix(".safetensors")),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
