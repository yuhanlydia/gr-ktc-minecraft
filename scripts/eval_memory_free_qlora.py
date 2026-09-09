#!/usr/bin/env python3
"""Paired memory-free evaluation of base, BC+DPO, and GR-KTC adapters."""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from evaluation.parser_validity import VoyagerActionParser
from evaluation.statistics import exact_mcnemar, wilson_interval
from gr_ktc.generation import generate_with_kv_prefix
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.collect_mineexplorer_pilot import verifier_score_events
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument(
        "--suite", choices=("retention", "heldout"), default="retention"
    )
    argument_parser.add_argument(
        "--refresh-full", action="store_true",
        help="Reuse checkpointed base/control trials and rerun only the full adapter.",
    )
    argument_parser.add_argument(
        "--scene-ids", nargs="+",
        help="Override the built-in retention/heldout scene set.",
    )
    argument_parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="Override the built-in evaluation seeds.",
    )
    argument_parser.add_argument("--output", type=Path)
    argument_parser.add_argument(
        "--full-adapter", type=Path,
        default=ROOT / "results/qlora_full_100",
    )
    argument_parser.add_argument(
        "--control-adapter", type=Path,
        default=ROOT / "results/qlora_control_100",
    )
    argument_parser.add_argument(
        "--system-style", choices=("enhanced", "raw"), default="enhanced",
    )
    args = argument_parser.parse_args()
    if args.output is not None:
        output_path = args.output
    elif args.suite == "retention":
        output_path = ROOT / "results/memory_free_qlora.json"
    else:
        output_path = ROOT / "results/memory_free_qlora_heldout.json"
    if args.scene_ids is not None:
        scene_ids = args.scene_ids
    elif args.suite == "retention":
        scene_ids = ["0281", "0299"]
    else:
        scene_ids = ["0391", "0499", "0532", "0549"]
    if args.seeds is not None:
        seeds = args.seeds
    elif args.suite == "retention":
        seeds = list(range(212, 220))
    else:
        seeds = list(range(220, 224))
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    base, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="nf4"
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        base, args.full_adapter, adapter_name="full"
    )
    model.load_adapter(args.control_adapter, adapter_name="control")
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    client = VoyagerHTTPClient(timeout_seconds=120)
    parser = VoyagerActionParser(
        ROOT / "third_party/voyager/voyager/env/mineflayer"
    )
    programs = primitive_programs()
    system = SYSTEM
    if args.system_style == "enhanced":
        system = system.replace(
            "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
            "Complete every subgoal in order using as many helper calls as needed. "
            "mineBlock already searches and mines. Keep the answer under 300 tokens.",
        )
    eos_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))
    trials = []
    if (args.resume or args.refresh_full) and output_path.exists():
        trials = json.loads(output_path.read_text()).get("trials", [])
    if args.refresh_full:
        trials = [item for item in trials if item["condition"] != "full"]
    completed = {
        (item["scene_id"], item["seed"], item["condition"]) for item in trials
    }

    def reset_inputs(scenario):
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

    started = time.perf_counter()
    for scene_id in scene_ids:
        scenario = scenarios[scene_id]
        for seed in seeds:
            for condition in ("base", "control", "full"):
                if (scene_id, seed, condition) in completed:
                    continue
                events, inputs = reset_inputs(scenario)
                if condition == "base":
                    context = model.disable_adapter()
                else:
                    model.set_adapter(condition)
                    context = nullcontext()
                with context:
                    ids = generate_with_kv_prefix(
                        model, inputs, None, context_id=None,
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
                output_path.write_text(json.dumps({
                    "protocol": "memory-free-qlora-v1-partial",
                    "scene_ids": scene_ids, "seeds": seeds,
                    "completed_trials": len(trials), "trials": trials,
                }, indent=2) + "\n")

    summary = {}
    for condition in ("base", "control", "full"):
        subset = [item for item in trials if item["condition"] == condition]
        successes = sum(item["score"] == 1.0 for item in subset)
        summary[condition] = {
            "mean_score": sum(item["score"] for item in subset) / len(subset),
            "full_successes": successes,
            "trials": len(subset),
            "success_wilson_95ci": wilson_interval(successes, len(subset)),
            "parser_valid": sum(item["parser_valid"] for item in subset),
        }
    paired = {
        (scene, seed): {
            condition: next(item for item in trials if item["scene_id"] == scene
                            and item["seed"] == seed and item["condition"] == condition)
            for condition in ("base", "control", "full")
        }
        for scene in scene_ids for seed in seeds
    }
    comparisons = {}
    for other in ("base", "control"):
        full = [row["full"]["score"] == 1.0 for row in paired.values()]
        comparison = [row[other]["score"] == 1.0 for row in paired.values()]
        full_only, other_only, p_value = exact_mcnemar(full, comparison)
        comparisons[f"full_vs_{other}"] = {
            "full_only": full_only, "other_only": other_only,
            "mcnemar_two_sided_p": p_value,
        }
    report = {
        "protocol": "memory-free-qlora-v1",
        "full_adapter": str(args.full_adapter),
        "control_adapter": str(args.control_adapter),
        "system_style": args.system_style,
        "scene_ids": scene_ids,
        "seeds": seeds,
        "summary": summary,
        "comparisons": comparisons,
        "trials": trials,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "suite": args.suite,
        "scope_warning": (
            "MineExplorer memory-free generalization; not PEAM held-out success."
            if args.suite == "heldout" or args.scene_ids is not None else
            "Training-context retention gate; not held-out PEAM success."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
