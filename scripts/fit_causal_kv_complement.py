#!/usr/bin/env python3
"""Fit an unreachable-state complement while preserving the native KV interface.

For each real teacher context, keep the shared rank-r residual weight update
fixed and search separate key/value strengths of the original quality-KV memory. Selection is
made by teacher-forced hidden-effect error, never by Minecraft evaluation
outcome.  This is a conservative one-dimensional causal complement baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.generation import teacher_forced_hidden_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.reachability import fit_individual_and_shared
from gr_ktc.residual_adapter import ResidualLoRAHook, text_layer
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--key-scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--value-scales", type=float, nargs="+", default=[0.0, 0.0625, 0.125, 0.1875, 0.25])
    parser.add_argument("--output", type=Path, default=ROOT / "results/causal_kv_complement_rank8.json")
    args = parser.parse_args()

    source = json.loads((ROOT / "results/fast_kv_cross_context.json").read_text())
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    data = load_file(ROOT / "results/real_reachability_data.safetensors")
    contexts = [(data[f"features_{s}"], data[f"kv_effect_{s}"]) for s in scene_ids]
    shared = fit_individual_and_shared(contexts, rank=args.rank)["shared"]
    raw = load_file(ROOT / "results/fast_kv_cross_context_memories.safetensors")

    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    model.eval()
    config = model.config.text_config
    layer = text_layer(model, 24)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    client = VoyagerHTTPClient(timeout_seconds=120)
    system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    results = {}
    for scene in scene_ids:
        scenario = scenarios[scene]
        events = client.reset(hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands)
        observation = final_observation(events)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {compact_observation(observation)}\nTask: {scenario.task_text}"
            )}]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt_gpu = {key: value.to(model.device) for key, value in prompt.items()}
        best = max(source["acquisitions"][scene]["records"], key=lambda x: (x["score"], x["parser_valid"]))
        response = processor.tokenizer(
            best["text"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        quality = {}
        advantages = torch.tensor(source["acquisitions"][scene]["advantages"])
        pos_count, neg_count = int((advantages > 0).sum()), int((advantages < 0).sum())
        for index in range(config.num_hidden_layers):
            positive = raw[f"scene_{scene}_positive_layer_{index}"]
            failed = raw[f"scene_{scene}_failed_layer_{index}"]
            quality[index] = (pos_count * positive + neg_count * failed) / (pos_count + neg_count)
        quality[24] = raw[f"scene_{scene}_contrastive_layer_24"]

        no_memory = teacher_forced_hidden_with_kv_prefix(
            model, prompt_gpu, response, None, context_id=None, layer_id=24
        )
        target = no_memory + data[f"kv_effect_{scene}"][: no_memory.shape[0]]
        candidates = []
        # Weight-only is an explicit candidate, unlike scale zero which still
        # appends keys and changes attention normalization.
        candidate_specs = [("weight_only", None, None)] + [
            (f"kv_k{key_scale:g}_v{value_scale:g}", key_scale, value_scale)
            for key_scale in args.key_scales for value_scale in args.value_scales
        ]
        for name, key_scale, value_scale in candidate_specs:
            memory = None
            if value_scale is not None:
                memory = KVPrefixMemory.from_flattened(
                    quality, kv_heads=config.num_key_value_heads, head_dim=head_dim,
                    context_id=f"{scene}:matched", key_scale=key_scale,
                    value_scale=value_scale,
                )
            with ResidualLoRAHook(layer, shared.lora_a, shared.lora_b):
                state = teacher_forced_hidden_with_kv_prefix(
                    model, prompt_gpu, response, memory,
                    context_id=f"{scene}:matched" if memory else None,
                    layer_id=24,
                )
            take = min(state.shape[0], target.shape[0])
            residual = state[:take].float() - target[:take].float()
            relative_mse = float(residual.square().sum() / target[:take].float().square().sum().clamp_min(1e-12))
            effect = state[:take].float() - no_memory[:take].float()
            target_effect = target[:take].float() - no_memory[:take].float()
            effect_rho = 1 - float((effect - target_effect).square().sum() / target_effect.square().sum().clamp_min(1e-12))
            candidates.append({
                "name": name, "key_scale": key_scale, "value_scale": value_scale,
                "relative_state_mse": relative_mse,
                "effect_rho": effect_rho,
            })
        best_candidate = min(candidates, key=lambda x: x["relative_state_mse"])
        results[scene] = {"candidates": candidates, "selected": best_candidate}

    report = {
        "protocol": "causal-kv-complement-search-v2",
        "rank": args.rank,
        "rho_shared_linear_proxy": shared.rho,
        "contexts": results,
        "selection_rule": "minimum teacher-forced layer-24 state MSE; no environment outcome used",
        "scope_warning": "Two-dimensional key/value scale search over the native quality-KV support; not a full learned unreachable KV projection.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
