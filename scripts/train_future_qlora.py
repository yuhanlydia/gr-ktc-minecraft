#!/usr/bin/env python3
"""Train a memory-free QLoRA student on same-model privileged-future targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.future_distillation import future_state_loss
from gr_ktc.lora_objectives import anchor_kl, dpo_loss, weighted_causal_lm_loss
from gr_ktc.lora_setup import attach_grktc_lora
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM
from scripts.train_grktc_qlora import sequence_logp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--future-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "results/fast_kv_four_context_acquisition.json",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "results/privileged_future_four_contexts.safetensors",
    )
    parser.add_argument(
        "--adapter-output", type=Path,
        default=ROOT / "results/qlora_future_100",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results/qlora_future_100.json",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.future_weight < 0 or args.anchor_weight < 0:
        raise ValueError("invalid training hyperparameters")

    source = json.loads(args.source.read_text())
    targets = load_file(args.targets)
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    chosen, rejected = {}, {}
    for scene in scene_ids:
        records = source["acquisitions"][scene]["records"]
        chosen[scene] = max(records, key=lambda x: (x["score"], x["parser_valid"]))["text"]
        rejected[scene] = min(records, key=lambda x: (x["score"], x["parser_valid"]))["text"]

    base, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", training=True, precision="nf4",
    )
    model = attach_grktc_lora(base, rank=args.rank, alpha=2 * args.rank, dropout=0.05)
    client = VoyagerHTTPClient(timeout_seconds=120)
    samples = {}
    for scene in scene_ids:
        events = client.reset(
            hard=True, kill_on_hard_reset=False,
            setup_commands=scenarios[scene].commands,
        )
        observation = compact_observation(final_observation(events))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {observation}\nTask: {scenarios[scene].task_text}"
            )}]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )["input_ids"]
        chosen_ids = processor.tokenizer(
            chosen[scene], add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        rejected_ids = processor.tokenizer(
            rejected[scene], add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        samples[scene] = {
            "prompt_tokens": prompt.shape[-1],
            "chosen_response": chosen_ids,
            "rejected_response": rejected_ids,
            "chosen_input": torch.cat((prompt, chosen_ids), dim=-1),
            "rejected_input": torch.cat((prompt, rejected_ids), dim=-1),
        }

    references = {}
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for scene, sample in samples.items():
            chosen_input = sample["chosen_input"].to(model.device)
            rejected_input = sample["rejected_input"].to(model.device)
            chosen_output = model(
                input_ids=chosen_input, attention_mask=torch.ones_like(chosen_input),
                output_hidden_states=True, use_cache=False,
            )
            rejected_output = model(
                input_ids=rejected_input, attention_mask=torch.ones_like(rejected_input),
                use_cache=False,
            )
            start = sample["prompt_tokens"]
            length = sample["chosen_response"].shape[-1]
            references[scene] = {
                "chosen_logp": sequence_logp(
                    chosen_output.logits, start,
                    sample["chosen_response"].to(model.device),
                ).detach(),
                "rejected_logp": sequence_logp(
                    rejected_output.logits, start,
                    sample["rejected_response"].to(model.device),
                ).detach(),
                "hidden": chosen_output.hidden_states[25][:, start:start + length].detach(),
                "prompt_logits": chosen_output.logits[:, :start].detach(),
            }

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4,
    )
    warmup = max(1, math.ceil(args.steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: ((step + 1) / warmup if step < warmup else
                      0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                         max(args.steps - warmup, 1)))),
    )
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(args.steps):
        scene = scene_ids[step % len(scene_ids)]
        sample = samples[scene]
        start = sample["prompt_tokens"]
        chosen_input = sample["chosen_input"].to(model.device)
        rejected_input = sample["rejected_input"].to(model.device)
        chosen_response = sample["chosen_response"].to(model.device)
        rejected_response = sample["rejected_response"].to(model.device)
        labels = chosen_input.clone()
        labels[:, :start] = -100
        optimizer.zero_grad(set_to_none=True)
        chosen_output = model(
            input_ids=chosen_input, attention_mask=torch.ones_like(chosen_input),
            output_hidden_states=True, use_cache=False,
        )
        rejected_output = model(
            input_ids=rejected_input, attention_mask=torch.ones_like(rejected_input),
            use_cache=False,
        )
        bc = weighted_causal_lm_loss(
            chosen_output.logits, labels, torch.ones(1, device=model.device),
        )
        dpo = dpo_loss(
            sequence_logp(chosen_output.logits, start, chosen_response),
            sequence_logp(rejected_output.logits, start, rejected_response),
            references[scene]["chosen_logp"], references[scene]["rejected_logp"],
            beta=0.1,
        )
        length = chosen_response.shape[-1]
        student = chosen_output.hidden_states[25][:, start:start + length].float()
        target_effect = targets[f"future_effect_{scene}"].to(model.device)
        take = min(student.shape[1], target_effect.shape[0])
        teacher = references[scene]["hidden"][:, :take].float() + target_effect[:take].unsqueeze(0)
        future = future_state_loss(teacher, student[:, :take])
        anchor = anchor_kl(
            references[scene]["prompt_logits"].float(),
            chosen_output.logits[:, :start].float(),
        )
        total = bc + dpo + args.future_weight * future + args.anchor_weight * anchor
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0,
        )
        optimizer.step()
        scheduler.step()
        if step % 5 == 0 or step == args.steps - 1:
            history.append({
                "step": step + 1, "scene_id": scene,
                "loss/total": float(total.detach()), "loss/bc": float(bc.detach()),
                "loss/dpo": float(dpo.detach()), "loss/future": float(future.detach()),
                "loss/anchor": float(anchor.detach()), "lr": scheduler.get_last_lr()[0],
            })

    args.adapter_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_output)
    report = {
        "protocol": "future-privileged-qlora-v1",
        "steps": args.steps, "scene_ids": scene_ids, "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "adapter_directory": str(
            args.adapter_output.resolve().relative_to(ROOT.resolve())
            if args.adapter_output.resolve().is_relative_to(ROOT.resolve())
            else args.adapter_output.resolve()
        ),
        "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.05},
        "loss_weights": {"bc": 1.0, "dpo": 1.0,
                         "future": args.future_weight, "anchor": args.anchor_weight},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
