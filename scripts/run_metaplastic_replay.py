#!/usr/bin/env python3
"""Run the first MetaPlastic structural gate on real MineExplorer rollouts.

This is an immutable-rollout replay gate, not a claim of online held-out TSR.
It verifies real Qwen gradients, global rank reallocation, and 24 GB feasibility
before paying for fresh environment interaction at every lifetime loop.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.group_advantage import group_relative_advantage
from gr_ktc.lora_setup import attach_grktc_lora
from gr_ktc.metaplastic import (
    decay_peft_plastic_weights,
    group_relative_policy_loss,
    project_peft_global_rank_budget,
)
from gr_ktc.model_loader import load_qwen3_vl_24gb


def mixed_groups(pool_roots: list[Path]):
    groups = []
    for pool in pool_roots:
        for group_dir in sorted((pool / "groups").glob("*")):
            group = json.loads((group_dir / "group.json").read_text())
            records = [
                json.loads(path.read_text())
                for path in sorted(group_dir.glob("trajectories/*/metadata.json"))
            ]
            if len(records) == 4 and len({r["verifier_score"] for r in records}) > 1:
                groups.append((group, records))
    return groups


def batch(tokenizer, group, records, device):
    prefix = (
        "You are a Minecraft Mineflayer code policy. Return one complete async "
        "function taking bot.\nTask: " + group["context"]["task_text"] + "\nAssistant:\n"
    )
    texts = [prefix + r["terminal_state"]["response"] for r in records]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512,
                        return_tensors="pt", add_special_tokens=True)
    prompt_len = tokenizer(prefix, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ].shape[-1]
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[encoded["attention_mask"] == 0] = -100
    scores = torch.tensor([r["verifier_score"] for r in records])
    return (
        encoded["input_ids"].to(device), encoded["attention_mask"].to(device),
        labels.to(device), group_relative_advantage(scores).to(device), scores,
    )


@torch.no_grad()
def probe_loss(model, tokenizer, heldout):
    model.eval()
    values = []
    for group, records in heldout:
        input_ids, mask, labels, advantages, _ = batch(
            tokenizer, group, records, model.device
        )
        output = model(input_ids=input_ids, attention_mask=mask)
        values.append(float(group_relative_policy_loss(
            output.logits, labels, advantages
        ).detach()))
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--candidate-rank", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--method", choices=("metaplastic", "fixed"), default="metaplastic")
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/metaplastic_replay.json")
    args = parser.parse_args()
    if args.loops < 1 or args.budget < 1 or args.candidate_rank < 1:
        raise ValueError("loops, budget, and candidate-rank must be positive")

    pools = [Path(p) for p in glob.glob(str(ROOT / "data/acquisition/mineexplorer_composite*"))]
    groups = mixed_groups(pools)
    if len(groups) < args.loops + 2:
        raise RuntimeError(f"need {args.loops + 2} mixed groups, found {len(groups)}")
    train_groups, heldout = groups[: args.loops], groups[-2:]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", training=True, precision="nf4"
    )
    # Candidate rank is per module storage capacity, not the plasticity budget.
    attached_rank = args.candidate_rank if args.method == "metaplastic" else args.budget
    model = attach_grktc_lora(
        model, rank=attached_rank, alpha=attached_rank, dropout=0.0
    )
    history = []
    probe_history = [{"loop": 0, "group_relative_nll": probe_loss(
        model, processor.tokenizer, heldout
    )}]
    previous_allocation = set()
    for loop, (group, records) in enumerate(train_groups, 1):
        if args.method == "metaplastic":
            decay_peft_plastic_weights(model, args.retention)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr
        )
        input_ids, attention_mask, labels, advantages, scores = batch(
            processor.tokenizer, group, records, model.device
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = group_relative_policy_loss(output.logits, labels, advantages)
        loss.backward()
        optimizer.step()
        report = (
            project_peft_global_rank_budget(model, args.budget)
            if args.method == "metaplastic" else None
        )
        by_layer = ({k: v for k, v in report.ranks.items() if v} if report else {})
        allocation = {(name, slot) for name, rank in by_layer.items() for slot in range(rank)}
        history.append({
            "loop": loop,
            "scene_id": group["context"]["scene_id"],
            "scores": scores.tolist(),
            "advantages": advantages.cpu().tolist(),
            "loss": float(loss.detach()),
            "active_rank": report.active_rank if report else None,
            "retained_energy": report.retained_energy if report else None,
            "rank_allocation": by_layer,
            "directions_entered": len(allocation - previous_allocation),
            "directions_pruned": len(previous_allocation - allocation),
        })
        previous_allocation = allocation
        del output, loss, optimizer
        torch.cuda.empty_cache()
        probe_history.append({"loop": loop, "group_relative_nll": probe_loss(
            model, processor.tokenizer, heldout
        )})

    result = {
        "protocol": "metaplastic-immutable-rollout-replay-v1",
        "method": args.method,
        "scope": "structural/gradient gate on real MineExplorer outcomes; not online TSR",
        "model": "Qwen3-VL-8B-Instruct NF4",
        "budget": args.budget,
        "candidate_rank_per_module": args.candidate_rank,
        "retention": args.retention,
        "loops": args.loops,
        "mixed_groups_available": len(groups),
        "history": history,
        "heldout_scene_ids": [g["context"]["scene_id"] for g, _ in heldout],
        "probe_history": probe_history,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
