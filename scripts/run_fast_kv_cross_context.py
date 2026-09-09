#!/usr/bin/env python3
"""Two-context paired control for real all-layer KV memory."""
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

from acquisition.mineexplorer import load_mineexplorer
from evaluation.parser_validity import VoyagerActionParser
from gr_ktc.generation import generate_with_final_kv, generate_with_kv_prefix
from gr_ktc.group_advantage import group_relative_advantage
from gr_ktc.kv_prefix import KVPrefixMemory, merge_raw_kv_trajectories
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.collect_mineexplorer_pilot import verifier_score_events
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    parser_cli = argparse.ArgumentParser()
    parser_cli.add_argument(
        "--candidate-ids", nargs="+",
        default=["0281", "0299", "0391", "0499", "0532", "0549", "0251"],
    )
    parser_cli.add_argument("--target-contexts", type=int, default=2)
    parser_cli.add_argument("--acquisition-seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    parser_cli.add_argument("--evaluation-seeds", type=int, nargs="+", default=[200, 201, 202, 203])
    parser_cli.add_argument("--skip-evaluation", action="store_true")
    parser_cli.add_argument(
        "--output", type=Path, default=ROOT / "results/fast_kv_cross_context.json",
    )
    parser_cli.add_argument(
        "--memory-output", type=Path,
        default=ROOT / "results/fast_kv_cross_context_memories.safetensors",
    )
    parser_cli.add_argument(
        "--raw-layer-output", type=Path,
        help="Optionally save every accepted rollout's unmerged K/V trajectory.",
    )
    parser_cli.add_argument("--raw-layer", type=int, default=24)
    args = parser_cli.parse_args()
    if args.target_contexts < 1:
        raise ValueError("target-contexts must be positive")
    if args.target_contexts == 1 and not args.skip_evaluation:
        raise ValueError("cross-context evaluation needs at least two contexts")
    candidate_ids = tuple(args.candidate_ids)
    wanted = set(candidate_ids)
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in wanted
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

    layer_count = model.config.text_config.num_hidden_layers
    text_config = model.config.text_config
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    acquisitions = {}
    skipped_all_equal = {}
    memories = {}
    frozen_flattened = {}
    raw_layer_tensors = {}
    mixed_scene_ids = []
    for scene_id in candidate_ids:
        scenario = scenarios[scene_id]
        records, trajectories = [], []
        for seed in args.acquisition_seeds:
            events, inputs = reset_inputs(scenario)
            torch.manual_seed(seed)
            generated = generate_with_final_kv(
                model, inputs, layer_ids=list(range(layer_count)),
                max_new_tokens=384, temperature=0.9, top_p=0.95,
            )
            text = processor.tokenizer.decode(
                generated.all_generated_token_ids[0], skip_special_tokens=True
            )
            valid, score, error = execute(scenario, text, events)
            records.append({
                "seed": seed, "parser_valid": valid, "score": score,
                "error": error, "text": text,
            })
            trajectories.append(generated.kv_by_layer)
        scores = torch.tensor([record["score"] for record in records])
        advantages = group_relative_advantage(scores)
        if not (advantages > 0).any() or not (advantages < 0).any():
            skipped_all_equal[scene_id] = records
            continue
        acquisitions[scene_id] = {
            "records": records, "advantages": advantages.tolist()
        }
        if args.raw_layer_output is not None:
            for rollout_index, trajectory in enumerate(trajectories):
                raw_layer_tensors[
                    f"scene_{scene_id}_rollout_{rollout_index}_layer_{args.raw_layer}"
                ] = trajectory[args.raw_layer].cpu().contiguous()
        for label, mask in (
            ("positive", advantages > 0), ("failed", advantages < 0)
        ):
            selected = [trajectory for trajectory, keep in zip(
                trajectories, mask, strict=True
            ) if bool(keep)]
            flattened = merge_raw_kv_trajectories(
                {layer: [item[layer] for item in selected]
                 for layer in range(layer_count)},
                torch.ones(len(selected)), memory_tokens=4,
            )
            frozen_flattened.update({
                f"scene_{scene_id}_{label}_layer_{layer}": tensor
                for layer, tensor in flattened.items()
            })
            memories[(scene_id, label)] = KVPrefixMemory.from_flattened(
                flattened,
                kv_heads=text_config.num_key_value_heads,
                head_dim=head_dim,
                context_id=f"{scene_id}:matched",
                value_scale=0.25,
            )
        contrastive_flattened = merge_raw_kv_trajectories(
            {layer: [item[layer] for item in trajectories]
             for layer in range(layer_count)},
            advantages,
            memory_tokens=4,
            negative_scale=0.5,
        )
        frozen_flattened.update({
            f"scene_{scene_id}_contrastive_layer_{layer}": tensor
            for layer, tensor in contrastive_flattened.items()
        })
        memories[(scene_id, "contrastive")] = KVPrefixMemory.from_flattened(
            contrastive_flattened,
            kv_heads=text_config.num_key_value_heads,
            head_dim=head_dim,
            context_id=f"{scene_id}:matched",
            value_scale=0.25,
        )
        mixed_scene_ids.append(scene_id)
        if len(mixed_scene_ids) == args.target_contexts:
            break

    if len(mixed_scene_ids) != args.target_contexts:
        raise RuntimeError(
            f"only found {len(mixed_scene_ids)}/{args.target_contexts} mixed contexts; "
            f"all-equal={list(skipped_all_equal)}"
        )

    eos_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))
    trials = []
    started = time.perf_counter()
    scene_ids = mixed_scene_ids
    for scene_id in (() if args.skip_evaluation else scene_ids):
        scenario = scenarios[scene_id]
        other_id = next(item for item in scene_ids if item != scene_id)
        other = memories[(other_id, "positive")]
        cross_memory = KVPrefixMemory(other.layers, f"{scene_id}:matched")
        conditions = {
            "no_memory": None,
            "positive_memory": memories[(scene_id, "positive")],
            "failed_memory": memories[(scene_id, "failed")],
            "contrastive_memory": memories[(scene_id, "contrastive")],
            "cross_context_memory": cross_memory,
        }
        for seed in args.evaluation_seeds:
            for condition, memory in conditions.items():
                events, inputs = reset_inputs(scenario)
                ids = generate_with_kv_prefix(
                    model, inputs, memory,
                    context_id=(f"{scene_id}:matched" if memory else None),
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
    summary = {}
    for condition in (
        "no_memory", "positive_memory", "failed_memory", "contrastive_memory",
        "cross_context_memory"
    ):
        subset = [trial for trial in trials if trial["condition"] == condition]
        summary[condition] = {
            "mean_score": (sum(item["score"] for item in subset) / len(subset)) if subset else None,
            "full_successes": sum(item["score"] == 1.0 for item in subset),
            "parser_valid": sum(item["parser_valid"] for item in subset),
            "trials": len(subset),
        }
    report = {
        "protocol": "real-fast-kv-cross-context-v1",
        "scene_ids": scene_ids,
        "memory_tokens": 4,
        "value_scale": 0.25,
        "target_contexts": args.target_contexts,
        "acquisition_seeds": args.acquisition_seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "evaluation_skipped": args.skip_evaluation,
        "acquisitions": acquisitions,
        "skipped_all_equal": skipped_all_equal,
        "summary": summary,
        "trials": trials,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "Mechanism acquisition; mixed outcomes are selected without using evaluation seeds.",
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    args.memory_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in frozen_flattened.items()},
        args.memory_output,
    )
    if args.raw_layer_output is not None:
        args.raw_layer_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(raw_layer_tensors, args.raw_layer_output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
