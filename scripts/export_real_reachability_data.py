#!/usr/bin/env python3
"""Export real Qwen residual-stream features and causal KV teacher effects.

The output is a small CPU safetensors dataset.  For text layer ``l`` it stores
the no-memory response-token residual stream entering that layer and the
already measured change at its output caused by matched quality KV memory.
This is the first-order proxy dataset used by the reachability sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.generation import teacher_forced_hidden_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "results/fast_kv_cross_context.json",
    )
    parser.add_argument(
        "--memory-input", type=Path,
        default=ROOT / "results/fast_kv_cross_context_memories.safetensors",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/real_reachability_data.safetensors",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "results/real_reachability_data.json",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    raw_memories = load_file(args.memory_input)
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    model.eval()
    text_config = model.config.text_config
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    if not 0 <= args.layer < text_config.num_hidden_layers:
        raise ValueError("layer out of range")
    client = VoyagerHTTPClient(timeout_seconds=120)
    system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    tensors: dict[str, torch.Tensor] = {}
    records = []
    torch.cuda.reset_peak_memory_stats()
    for scene_id in scene_ids:
        scenario = scenarios[scene_id]
        events = client.reset(
            hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands
        )
        observation = final_observation(events)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {compact_observation(observation)}\n"
                f"Task: {scenario.task_text}"
            )}]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt_ids = prompt["input_ids"]
        best = max(
            source["acquisitions"][scene_id]["records"],
            key=lambda item: (item["score"], item["parser_valid"]),
        )
        response = processor.tokenizer(
            best["text"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        input_ids = torch.cat((prompt_ids, response), dim=-1).to(model.device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        start = prompt_ids.shape[-1]
        length = response.shape[-1]
        layer_input = output.hidden_states[args.layer][0, start:start + length].float().cpu()
        base_output = output.hidden_states[args.layer + 1][0, start:start + length].float().cpu()
        advantages = torch.tensor(source["acquisitions"][scene_id]["advantages"])
        pos_count, neg_count = int((advantages > 0).sum()), int((advantages < 0).sum())
        quality = {}
        for index in range(text_config.num_hidden_layers):
            positive = raw_memories[f"scene_{scene_id}_positive_layer_{index}"]
            failed = raw_memories[f"scene_{scene_id}_failed_layer_{index}"]
            quality[index] = (pos_count * positive + neg_count * failed) / (pos_count + neg_count)
        quality[args.layer] = raw_memories[f"scene_{scene_id}_contrastive_layer_{args.layer}"]
        memory = KVPrefixMemory.from_flattened(
            quality, kv_heads=text_config.num_key_value_heads, head_dim=head_dim,
            context_id=f"{scene_id}:matched", value_scale=0.25,
        )
        prompt_gpu = {key: value.to(model.device) for key, value in prompt.items()}
        response_gpu = response.to(model.device)
        no_memory = teacher_forced_hidden_with_kv_prefix(
            model, prompt_gpu, response_gpu, None, context_id=None, layer_id=args.layer,
        )
        quality_state = teacher_forced_hidden_with_kv_prefix(
            model, prompt_gpu, response_gpu, memory,
            context_id=f"{scene_id}:matched", layer_id=args.layer,
        )
        teacher = quality_state - no_memory
        take = min(layer_input.shape[0], teacher.shape[0])
        tensors[f"features_{scene_id}"] = layer_input[:take].contiguous()
        tensors[f"base_output_{scene_id}"] = base_output[:take].contiguous()
        tensors[f"kv_effect_{scene_id}"] = teacher[:take].contiguous()
        records.append({
            "scene_id": scene_id,
            "task_text": scenario.task_text,
            "prompt_tokens": int(start),
            "response_tokens": int(length),
            "aligned_tokens": int(take),
            "feature_dim": int(layer_input.shape[-1]),
        })
        del output
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    report = {
        "protocol": "real-qwen-kv-reachability-data-v1",
        "model": "Qwen3-VL-8B-Instruct",
        "precision": "bf16",
        "layer": args.layer,
        "contexts": records,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "feature_semantics": "no-memory residual stream entering selected text layer",
        "target_semantics": "matched quality-KV causal output hidden shift",
        "source": str(args.source),
        "memory_input": str(args.memory_input),
        "scope": "first-order residual-stream proxy; actual LoRA optimization is a separate gate",
    }
    args.metadata.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
