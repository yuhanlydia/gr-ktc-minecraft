#!/usr/bin/env python3
"""Test whether the Gate-1-selected layer 24 adds quality beyond context KV."""
from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from evaluation.parser_validity import VoyagerActionParser
from gr_ktc.generation import generate_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.collect_mineexplorer_pilot import verifier_score_events
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--resume", action="store_true")
    args = argument_parser.parse_args()
    output = ROOT / "results/layer24_quality_control.json"
    evaluation_seeds = tuple(range(204, 212))
    source_report = json.loads(
        (ROOT / "results/fast_kv_cross_context.json").read_text()
    )
    tensors = load_file(ROOT / "results/fast_kv_cross_context_memories.safetensors")
    scene_ids = source_report["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    torch.cuda.reset_peak_memory_stats()
    client = VoyagerHTTPClient(timeout_seconds=120)
    parser = VoyagerActionParser(
        ROOT / "third_party/voyager/voyager/env/mineflayer"
    )
    programs = primitive_programs()
    system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    text_config = model.config.text_config
    layer_count = text_config.num_hidden_layers
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )

    def memory(flattened, scene_id):
        return KVPrefixMemory.from_flattened(
            flattened,
            kv_heads=text_config.num_key_value_heads,
            head_dim=head_dim,
            context_id=f"{scene_id}:matched",
            value_scale=0.25,
        )

    memories = {}
    for scene_id in scene_ids:
        advantages = torch.tensor(source_report["acquisitions"][scene_id]["advantages"])
        positive_count = int((advantages > 0).sum())
        failed_count = int((advantages < 0).sum())
        positive = {
            layer: tensors[f"scene_{scene_id}_positive_layer_{layer}"]
            for layer in range(layer_count)
        }
        failed = {
            layer: tensors[f"scene_{scene_id}_failed_layer_{layer}"]
            for layer in range(layer_count)
        }
        contrastive = {
            layer: tensors[f"scene_{scene_id}_contrastive_layer_{layer}"]
            for layer in range(layer_count)
        }
        center = {
            layer: (
                positive_count * positive[layer] + failed_count * failed[layer]
            ) / (positive_count + failed_count)
            for layer in range(layer_count)
        }
        quality = dict(center)
        quality[24] = contrastive[24]
        memories[scene_id] = {
            "context_only": memory(center, scene_id),
            "layer24_quality": memory(quality, scene_id),
            "positive_all_layers": memory(positive, scene_id),
            "failed_all_layers": memory(failed, scene_id),
        }

    def reset_inputs(scenario):
        events = client.reset(
            hard=True, kill_on_hard_reset=False,
            setup_commands=scenario.commands,
        )
        observation = final_observation(events)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {compact_observation(observation)}\n"
                f"Task: {scenario.task_text}"
            )}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        return events, {key: value.to(model.device) for key, value in inputs.items()}

    def execute(scenario, text, initial_events):
        try:
            parsed = parser.parse(text)
        except ValueError as exc:
            return False, 0.0, str(exc)
        try:
            events = client.step(
                code=parsed.exec_code,
                programs=programs + "\n\n" + parsed.program_code,
            )
            return True, verifier_score_events(scenario.milestones, events), None
        except Exception as exc:
            return True, 0.0, f"{type(exc).__name__}: {exc}"

    eos_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))
    trials = []
    if args.resume and output.exists():
        trials = json.loads(output.read_text()).get("trials", [])
    completed = {
        (item["scene_id"], item["seed"], item["condition"]) for item in trials
    }
    started = time.perf_counter()
    for scene_id in scene_ids:
        scenario = scenarios[scene_id]
        conditions = {"no_memory": None, **memories[scene_id]}
        for seed in evaluation_seeds:
            for condition, kv_memory in conditions.items():
                if (scene_id, seed, condition) in completed:
                    continue
                events, inputs = reset_inputs(scenario)
                ids = generate_with_kv_prefix(
                    model, inputs, kv_memory,
                    context_id=(f"{scene_id}:matched" if kv_memory else None),
                    max_new_tokens=384, temperature=0.9, top_p=0.95,
                    eos_token_ids=eos_ids,
                    generator=torch.Generator(device=model.device).manual_seed(seed),
                )
                text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
                valid, score, error = execute(scenario, text, events)
                trials.append({
                    "scene_id": scene_id, "seed": seed, "condition": condition,
                    "parser_valid": valid, "score": score, "error": error,
                    "generated_tokens": int(ids.shape[-1]), "text": text,
                })
                output.write_text(json.dumps({
                    "protocol": "real-layer24-quality-control-v1-partial",
                    "scene_ids": scene_ids,
                    "seeds": list(evaluation_seeds),
                    "trials": trials,
                    "completed_trials": len(trials),
                }, indent=2) + "\n")
    conditions = ("no_memory", "context_only", "layer24_quality",
                  "positive_all_layers", "failed_all_layers")
    summary = {}
    for condition in conditions:
        subset = [trial for trial in trials if trial["condition"] == condition]
        summary[condition] = {
            "mean_score": sum(item["score"] for item in subset) / len(subset),
            "full_successes": sum(item["score"] == 1.0 for item in subset),
            "parser_valid": sum(item["parser_valid"] for item in subset),
            "trials": len(subset),
        }
    report = {
        "protocol": "real-layer24-quality-control-v1",
        "scene_ids": scene_ids,
        "seeds": list(evaluation_seeds),
        "summary": summary,
        "trials": trials,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "Two-context causal pilot; not full Gate 2 statistics.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
