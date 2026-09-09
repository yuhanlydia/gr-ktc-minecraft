#!/usr/bin/env python3
"""Evaluate prompt-latent retrieval over a population of native KV memories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from evaluation.parser_validity import VoyagerActionParser
from evaluation.statistics import exact_mcnemar
from gr_ktc.generation import generate_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.collect_mineexplorer_pilot import verifier_score_events
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-source", type=Path,
        default=ROOT / "results/fast_kv_four_context_acquisition.json",
    )
    parser.add_argument(
        "--archive-memories", type=Path,
        default=ROOT / "results/fast_kv_four_context_memories.safetensors",
    )
    parser.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[340, 341])
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/latent_population_kv.json",
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    source = json.loads(args.archive_source.read_text())
    archive_ids = source["scene_ids"]
    eval_ids = args.scene_ids or archive_ids
    wanted = set(archive_ids) | set(eval_ids)
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in wanted
    }
    raw = load_file(args.archive_memories)
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16",
    )
    model.eval()
    config = model.config.text_config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads,
    )
    client = VoyagerHTTPClient(timeout_seconds=120)
    action_parser = VoyagerActionParser(
        ROOT / "third_party/voyager/voyager/env/mineflayer",
    )
    programs = primitive_programs()

    def reset_inputs(scene):
        scenario = scenarios[scene]
        events = client.reset(
            hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands,
        )
        observation = compact_observation(final_observation(events))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {observation}\nTask: {scenario.task_text}"
            )}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        return events, {key: value.to(model.device) for key, value in inputs.items()}

    def prompt_coordinate(inputs):
        with torch.no_grad():
            output = model(
                **inputs, output_hidden_states=True, use_cache=False,
                return_dict=True,
            )
        return F.normalize(output.hidden_states[24][0, -1].float().cpu(), dim=0)

    quality = {}
    for scene in archive_ids:
        advantages = torch.tensor(source["acquisitions"][scene]["advantages"])
        pos_count, neg_count = int((advantages > 0).sum()), int((advantages < 0).sum())
        layers = {}
        for layer in range(config.num_hidden_layers):
            positive = raw[f"scene_{scene}_positive_layer_{layer}"]
            failed = raw[f"scene_{scene}_failed_layer_{layer}"]
            layers[layer] = (pos_count * positive + neg_count * failed) / (pos_count + neg_count)
        layers[24] = raw[f"scene_{scene}_contrastive_layer_24"]
        quality[scene] = layers

    archive_coordinates = []
    for scene in archive_ids:
        _, inputs = reset_inputs(scene)
        archive_coordinates.append(prompt_coordinate(inputs))
    archive_coordinates = torch.stack(archive_coordinates)

    def memory_from_weights(weights, context_id):
        mixture = {
            layer: sum(
                float(weight) * quality[scene][layer]
                for weight, scene in zip(weights, archive_ids, strict=True)
            )
            for layer in range(config.num_hidden_layers)
        }
        return KVPrefixMemory.from_flattened(
            mixture, kv_heads=config.num_key_value_heads, head_dim=head_dim,
            context_id=context_id, value_scale=0.25,
        )

    eos_ids = tuple(value for value in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(value, int))
    trials = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for scene in eval_ids:
        conditions = ["base", "population", "uniform"]
        if scene in archive_ids:
            conditions.insert(1, "matched")
        for seed in args.seeds:
            for condition in conditions:
                initial_events, inputs = reset_inputs(scene)
                query = prompt_coordinate(inputs)
                similarities = archive_coordinates @ query
                posterior = torch.softmax(similarities / args.temperature, dim=0)
                if condition == "base":
                    memory = None
                    used_weights = torch.zeros_like(posterior)
                elif condition == "uniform":
                    used_weights = torch.full_like(posterior, 1 / len(posterior))
                    memory = memory_from_weights(used_weights, f"{scene}:matched")
                elif condition == "matched":
                    used_weights = torch.zeros_like(posterior)
                    used_weights[archive_ids.index(scene)] = 1
                    memory = memory_from_weights(used_weights, f"{scene}:matched")
                else:
                    used_weights = posterior
                    memory = memory_from_weights(used_weights, f"{scene}:matched")
                ids = generate_with_kv_prefix(
                    model, inputs, memory,
                    context_id=f"{scene}:matched" if memory else None,
                    max_new_tokens=args.max_new_tokens, temperature=0.9, top_p=0.95,
                    eos_token_ids=eos_ids,
                    generator=torch.Generator(device=model.device).manual_seed(seed),
                )
                text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
                try:
                    parsed = action_parser.parse(text)
                    valid = True
                    try:
                        events = client.step(
                            code=parsed.exec_code,
                            programs=programs + "\n\n" + parsed.program_code,
                        )
                        score = verifier_score_events(scenarios[scene].milestones, events)
                        error = None
                    except Exception as exc:
                        score, error = 0.0, f"{type(exc).__name__}: {exc}"
                except ValueError as exc:
                    valid, score, error = False, 0.0, str(exc)
                trials.append({
                    "scene_id": scene, "seed": seed, "condition": condition,
                    "score": score, "parser_valid": valid, "error": error,
                    "posterior": posterior.tolist(), "used_weights": used_weights.tolist(),
                    "retrieved": archive_ids[int(posterior.argmax())],
                    "generated_tokens": int(ids.shape[-1]), "text": text,
                })
                args.output.write_text(json.dumps({
                    "protocol": "latent-population-kv-v1-partial", "trials": trials,
                }, indent=2) + "\n")

    condition_names = sorted({x["condition"] for x in trials})
    summary = {}
    for condition in condition_names:
        subset = [x for x in trials if x["condition"] == condition]
        summary[condition] = {
            "full_successes": sum(x["score"] == 1 for x in subset),
            "mean_score": sum(x["score"] for x in subset) / len(subset),
            "parser_valid": sum(x["parser_valid"] for x in subset),
            "trials": len(subset),
        }
    comparisons = {}
    indexed = {(x["scene_id"], x["seed"], x["condition"]): x for x in trials}
    for condition in condition_names:
        if condition == "base":
            continue
        pairs = [
            (scene, seed) for scene in eval_ids for seed in args.seeds
            if (scene, seed, condition) in indexed
        ]
        other = [indexed[(s, seed, condition)]["score"] == 1 for s, seed in pairs]
        base = [indexed[(s, seed, "base")]["score"] == 1 for s, seed in pairs]
        left, right, p = exact_mcnemar(other, base)
        comparisons[f"{condition}_vs_base"] = {
            "condition_only": left, "base_only": right, "p": p,
        }
    retrieval_accuracy = None
    if set(eval_ids).issubset(set(archive_ids)):
        population_trials = [x for x in trials if x["condition"] == "population"]
        retrieval_accuracy = sum(x["retrieved"] == x["scene_id"] for x in population_trials) / len(population_trials)
    report = {
        "protocol": "latent-population-kv-v1",
        "archive_ids": archive_ids, "scene_ids": eval_ids, "seeds": args.seeds,
        "temperature": args.temperature, "summary": summary,
        "comparisons": comparisons, "retrieval_accuracy": retrieval_accuracy,
        "trials": trials, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "Prompt-latent retrieval over four acquired native-KV memories; held-out-task retrieval has no oracle matched memory.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
