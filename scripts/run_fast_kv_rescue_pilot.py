#!/usr/bin/env python3
"""Paired real-Minecraft fast KV rescue pilot on one mixed context."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

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
    scenario = next(
        item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id == "0391"
    )
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

    def reset_inputs():
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

    def execute(text: str, initial_events):
        try:
            parsed = parser.parse(text)
        except ValueError as exc:
            return False, 0.0, str(exc), initial_events
        try:
            events = client.step(
                code=parsed.exec_code,
                programs=programs + "\n\n" + parsed.program_code,
            )
            return True, verifier_score_events(scenario.milestones, events), None, events
        except Exception as exc:
            return True, 0.0, f"{type(exc).__name__}: {exc}", initial_events

    layer_count = model.config.text_config.num_hidden_layers
    trajectories = []
    acquisition = []
    for seed in (100, 101, 102, 103):
        events, inputs = reset_inputs()
        torch.manual_seed(seed)
        generated = generate_with_final_kv(
            model, inputs, layer_ids=list(range(layer_count)),
            max_new_tokens=384, temperature=0.9, top_p=0.95,
        )
        text = processor.tokenizer.decode(
            generated.all_generated_token_ids[0], skip_special_tokens=True
        )
        valid, score, error, _ = execute(text, events)
        trajectories.append(generated.kv_by_layer)
        acquisition.append({
            "seed": seed, "parser_valid": valid, "score": score,
            "error": error, "text": text,
        })
    scores = torch.tensor([item["score"] for item in acquisition])
    advantages = group_relative_advantage(scores)
    if not (advantages > 0).any() or not (advantages < 0).any():
        raise RuntimeError(f"acquisition was not mixed: {scores.tolist()}")

    def merged_for(mask: torch.Tensor):
        selected = [trajectory for trajectory, keep in zip(
            trajectories, mask, strict=True
        ) if bool(keep)]
        return merge_raw_kv_trajectories(
            {
                layer: [trajectory[layer] for trajectory in selected]
                for layer in range(layer_count)
            },
            torch.ones(len(selected)),
            memory_tokens=4,
        )

    positive_flat = merged_for(advantages > 0)
    failed_flat = merged_for(advantages < 0)
    random_generator = torch.Generator().manual_seed(991)
    random_flat = {
        layer: states[:, torch.randperm(
            states.shape[1], generator=random_generator
        )]
        for layer, states in positive_flat.items()
    }
    text_config = model.config.text_config
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )

    def make_memory(flattened, label):
        return KVPrefixMemory.from_flattened(
            flattened,
            kv_heads=text_config.num_key_value_heads,
            head_dim=head_dim,
            context_id=f"{scenario.scene_id}:matched",
            value_scale=0.25,
        )

    memories = {
        "no_memory": None,
        "positive_memory": make_memory(positive_flat, "positive"),
        "failed_memory": make_memory(failed_flat, "failed"),
        "randomized_memory": make_memory(random_flat, "random"),
    }
    eos_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))
    trials = []
    started = time.perf_counter()
    for seed in (200, 201, 202, 203):
        for condition, memory in memories.items():
            events, inputs = reset_inputs()
            ids = generate_with_kv_prefix(
                model, inputs, memory,
                context_id=(f"{scenario.scene_id}:matched" if memory else None),
                max_new_tokens=384,
                temperature=0.9,
                top_p=0.95,
                eos_token_ids=eos_ids,
                generator=torch.Generator(device=model.device).manual_seed(seed),
            )
            text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
            valid, score, error, _ = execute(text, events)
            trials.append({
                "seed": seed, "condition": condition,
                "parser_valid": valid, "score": score,
                "error": error, "generated_tokens": int(ids.shape[-1]),
                "text": text,
            })
    summary = {}
    for condition in memories:
        subset = [trial for trial in trials if trial["condition"] == condition]
        summary[condition] = {
            "mean_score": sum(trial["score"] for trial in subset) / len(subset),
            "full_successes": sum(trial["score"] == 1.0 for trial in subset),
            "parser_valid": sum(trial["parser_valid"] for trial in subset),
        }
    report = {
        "protocol": "real-fast-kv-rescue-pilot-v1",
        "scene_id": scenario.scene_id,
        "task": scenario.task_text,
        "memory_tokens": 4,
        "value_scale": 0.25,
        "acquisition": acquisition,
        "advantages": advantages.tolist(),
        "summary": summary,
        "trials": trials,
        "evaluation_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "One-context pilot; Gate 2 requires more paired contexts.",
    }
    output = ROOT / "results/fast_kv_rescue_pilot.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
