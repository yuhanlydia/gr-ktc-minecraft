#!/usr/bin/env python3
"""Analyze individual/shared rank reachability on exported real Qwen effects."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.reachability import fit_individual_and_shared, rank_scaling, split_state_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "results/real_reachability_data.safetensors",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/real_reachability_analysis.json",
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--heterogeneity-rank", type=int, default=8)
    parser.add_argument("--target-prefix", default="kv_effect")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    tensors = load_file(args.input)
    scene_ids = sorted(key.removeprefix("features_") for key in tensors if key.startswith("features_"))
    contexts = [
        (tensors[f"features_{scene}"], tensors[f"{args.target_prefix}_{scene}"])
        for scene in scene_ids
    ]
    curve = rank_scaling(contexts, tuple(args.ranks), ridge=args.ridge)
    individual = []
    for scene, (features, target) in zip(scene_ids, contexts, strict=True):
        reachable, unreachable, fit = split_state_correction(
            features, target, rank=max(args.ranks), ridge=args.ridge
        )
        individual.append({
            "scene_id": scene,
            "rho": fit.rho,
            "target_norm": float(target.norm()),
            "reachable_norm": float(reachable.norm()),
            "unreachable_norm": float(unreachable.norm()),
        })
    shared = fit_individual_and_shared(contexts, rank=max(args.ranks), ridge=args.ridge)
    context_scaling = []
    for count in range(1, len(contexts) + 1):
        values = []
        for indices in itertools.combinations(range(len(contexts)), count):
            subset = [contexts[index] for index in indices]
            values.append(float(fit_individual_and_shared(
                subset, rank=args.heterogeneity_rank, ridge=args.ridge,
            )["shared"].rho))
        context_scaling.append({
            "contexts": count,
            "rank": args.heterogeneity_rank,
            "rho_mean": sum(values) / len(values),
            "rho_min": min(values),
            "rho_max": max(values),
            "subsets": len(values),
        })
    report = {
        "protocol": "real-qwen-state-weight-reachability-proxy-v1",
        "contexts": scene_ids,
        "target_prefix": args.target_prefix,
        "rank_curve": curve,
        "context_heterogeneity": context_scaling,
        "individual_at_max_rank": individual,
        "rho_individual_mean_at_max_rank": float(shared["rho_individual_mean"]),
        "rho_joint_at_max_rank": float(shared["shared"].rho),
        "scope_warning": f"Real {args.target_prefix} targets with a residual-stream linear proxy; not yet an optimized multi-module QLoRA Jacobian fit.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
