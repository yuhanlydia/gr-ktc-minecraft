#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.lora_objectives import weighted_causal_lm_loss
from gr_ktc.lora_setup import attach_grktc_lora
from gr_ktc.model_loader import load_qwen3_vl_24gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results/qlora_smoke.json")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    successful = json.loads((ROOT / "results/local_qwen_action_retry2.json").read_text())
    prompt = (
        "You are a Minecraft Mineflayer code policy. Return one complete async "
        "function taking bot.\nTask: Collect 4 oak logs\nAssistant:\n"
    )
    response = successful["response"]

    torch.cuda.reset_peak_memory_stats()
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", training=True, precision="nf4"
    )
    model = attach_grktc_lora(model, rank=32, alpha=64, dropout=0.05)
    tokenizer = processor.tokenizer
    full = tokenizer(prompt + response, return_tensors="pt", add_special_tokens=True)
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ].shape[-1]
    input_ids = full["input_ids"].to(model.device)
    attention_mask = full["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[:, :prompt_tokens] = -100

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4,
    )
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = weighted_causal_lm_loss(
            output.logits, labels, torch.ones(1, device=output.logits.device)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    adapter_dir = args.output.with_suffix("")
    model.save_pretrained(adapter_dir)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "protocol": "qlora-consolidation-smoke-v1",
        "steps": args.steps,
        "losses": losses,
        "elapsed_seconds": time.perf_counter() - started,
        "sequence_tokens": int(input_ids.shape[-1]),
        "trainable_parameters": trainable,
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "adapter_directory": str(adapter_dir.relative_to(ROOT)),
        "scope_warning": "One successful real rollout BC smoke; not the full 100-step GRTA/DPO run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
