#!/usr/bin/env python3
"""Real Qwen all-layer KV-prefix generation smoke (not a Gate 2 result)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parser_validity import VoyagerActionParser
from gr_ktc.generation import generate_with_final_kv, generate_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory, merge_raw_kv_trajectories
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    torch.cuda.reset_peak_memory_stats()
    client = VoyagerHTTPClient(timeout_seconds=30)
    events = client.reset(
        hard=True,
        setup_commands=(
            "/fill ~-3 ~-1 ~-3 ~3 ~-1 ~3 minecraft:grass_block",
            "/fill ~-3 ~0 ~-3 ~3 ~3 ~3 minecraft:air",
            "/setblock ~2 ~0 ~2 minecraft:crafting_table",
            "/give @p minecraft:oak_log 2",
        ),
    )
    observation = final_observation(events)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": (
            f"Observation: {compact_observation(observation)}\n"
            "Task: Craft a crafting table"
        )}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    layer_count = model.config.text_config.num_hidden_layers
    torch.manual_seed(42)
    teacher = generate_with_final_kv(
        model, inputs, layer_ids=list(range(layer_count)), max_new_tokens=128,
        temperature=0.7, top_p=0.9,
    )
    text_config = model.config.text_config
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    eos_token_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))
    parser = VoyagerActionParser(
        ROOT / "third_party/voyager/voyager/env/mineflayer"
    )
    torch.manual_seed(43)
    baseline = generate_with_final_kv(
        model, inputs, layer_ids=[24], max_new_tokens=128,
        temperature=0.7, top_p=0.9,
    )
    baseline_text = processor.tokenizer.decode(
        baseline.all_generated_token_ids[0], skip_special_tokens=True
    )
    sweep = []
    for memory_tokens in (1, 2, 4, 8):
        merged = merge_raw_kv_trajectories(
            {layer: [trajectory] for layer, trajectory in teacher.kv_by_layer.items()},
            torch.tensor([1.0]), memory_tokens=memory_tokens,
        )
        for value_scale in (0.1, 0.25, 0.5, 1.0):
            memory = KVPrefixMemory.from_flattened(
                merged,
                kv_heads=text_config.num_key_value_heads,
                head_dim=head_dim,
                context_id="fast-kv-smoke-context",
                value_scale=value_scale,
            )
            generated_ids = generate_with_kv_prefix(
                model, inputs, memory,
                context_id="fast-kv-smoke-context",
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                eos_token_ids=eos_token_ids,
                generator=torch.Generator(device=model.device).manual_seed(43),
            )
            memory_text = processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            sweep.append({
                "memory_tokens": memory_tokens,
                "value_scale": value_scale,
                "parser_valid": parser.is_valid(memory_text),
                "generated_tokens": int(generated_ids.shape[-1]),
                "text": memory_text,
            })
    teacher_text = processor.tokenizer.decode(
        teacher.all_generated_token_ids[0], skip_special_tokens=True
    )
    report = {
        "protocol": "real-fast-kv-prefix-smoke-v1",
        "layers": layer_count,
        "teacher_parser_valid": parser.is_valid(teacher_text),
        "baseline_parser_valid": parser.is_valid(baseline_text),
        "teacher_text": teacher_text,
        "baseline_text": baseline_text,
        "sweep": sweep,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "Interface smoke only; teacher advantage is synthetic.",
    }
    output = ROOT / "results/fast_kv_prefix_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
