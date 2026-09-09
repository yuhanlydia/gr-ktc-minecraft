#!/usr/bin/env python3
"""Export real hindsight/privileged-future latent targets from successful rollouts.

The student sees only the current observation and task.  The frozen teacher is
the same Qwen model but additionally receives the successful future action and
verified terminal score in its prompt.  Both are teacher-forced on the same
successful action tokens, so their response-token hidden difference is a
well-defined future-information correction rather than a stronger-model label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.future_distillation import future_kl_loss, future_state_loss
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM


def _forward_response(model, prompt, response, layer: int):
    prompt_len = prompt["input_ids"].shape[-1]
    ids = torch.cat((prompt["input_ids"], response), dim=-1).to(model.device)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        output = model(
            input_ids=ids, attention_mask=attention, output_hidden_states=True,
            use_cache=False, return_dict=True,
        )
    length = response.shape[-1]
    start = prompt_len
    return (
        output.hidden_states[layer][0, start:start + length].float().cpu(),
        output.hidden_states[layer + 1][0, start:start + length].float().cpu(),
        output.logits[0, start:start + length].float().cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "results/fast_kv_four_context_acquisition.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/privileged_future_four_contexts.safetensors",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "results/privileged_future_four_contexts.json",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16",
    )
    model.eval()
    client = VoyagerHTTPClient(timeout_seconds=120)
    tensors = {}
    records = []
    torch.cuda.reset_peak_memory_stats()

    for scene in scene_ids:
        scenario = scenarios[scene]
        events = client.reset(
            hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands,
        )
        observation = compact_observation(final_observation(events))
        best = max(
            source["acquisitions"][scene]["records"],
            key=lambda item: (item["score"], item["parser_valid"]),
        )
        response = processor.tokenizer(
            best["text"], add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        current = f"Observation: {observation}\nTask: {scenario.task_text}"
        privileged = (
            current
            + "\nPrivileged successful future (training only):\n"
            + best["text"]
            + f"\nEnvironment verifier terminal score: {best['score']:.3f}"
        )

        def make_prompt(user_text):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]
            return processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )

        student_input, student_output, student_logits = _forward_response(
            model, make_prompt(current), response, args.layer,
        )
        _, teacher_output, teacher_logits = _forward_response(
            model, make_prompt(privileged), response, args.layer,
        )
        take = min(student_output.shape[0], teacher_output.shape[0])
        target = teacher_output[:take] - student_output[:take]
        tensors[f"features_{scene}"] = student_input[:take].contiguous()
        tensors[f"future_effect_{scene}"] = target.contiguous()
        records.append({
            "scene_id": scene,
            "tokens": take,
            "verified_score": best["score"],
            "hidden_loss": float(future_state_loss(
                teacher_output[:take], student_output[:take],
            )),
            "logit_kl": float(future_kl_loss(
                teacher_logits[:take], student_logits[:take],
            )),
            "effect_norm": float(target.norm()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    report = {
        "protocol": "real-qwen-privileged-future-v1",
        "model": "Qwen3-VL-8B-Instruct",
        "layer": args.layer,
        "teacher_privilege": "successful future action plus environment verifier terminal score",
        "student_information": "current observation and task only",
        "contexts": records,
        "mean_hidden_loss": sum(x["hidden_loss"] for x in records) / len(records),
        "mean_logit_kl": sum(x["logit_kl"] for x in records) / len(records),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "Frozen same-model teacher target export; no student adapter optimization or held-out behavior claim yet.",
    }
    args.metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
