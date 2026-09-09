#!/usr/bin/env python3
"""Unified real-data mechanism report for the four research directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.conditional_lora import conditional_reachability_score, fit_contextual_lora_basis
from gr_ktc.future_distillation import backward_future_targets, future_state_loss
from gr_ktc.latent_population import LatentPopulation, PopulationItem
from gr_ktc.reachability import fit_individual_and_shared, rank_scaling, split_state_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/real_reachability_data.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_four_ideas_analysis.json")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    torch.set_num_threads(4)
    tensors = load_file(args.input)
    scenes = sorted(key.removeprefix("features_") for key in tensors if key.startswith("features_"))
    contexts = [(tensors[f"features_{s}"], tensors[f"kv_effect_{s}"]) for s in scenes]

    reachability = rank_scaling(contexts, tuple(args.ranks))
    decomposition = []
    for rank in args.ranks:
        reconstruct = []
        for features, target in contexts:
            reachable, unreachable, fit = split_state_correction(features, target, rank=rank)
            reconstruct.append(float((reachable + unreachable - target).norm() / target.norm().clamp_min(1e-12)))
        decomposition.append({"rank": rank, "max_relative_reconstruction_error": max(reconstruct)})

    # Coordinates are causal teacher-effect summaries, not semantic task IDs.
    pooled = torch.stack([target.mean(0) for _, target in contexts])
    _, _, coordinate_basis = torch.pca_lowrank(pooled, q=min(len(contexts), 2), center=True)
    coordinates = pooled @ coordinate_basis
    conditional = []
    for bases in (1, 2):
        model = fit_contextual_lora_basis(
            coordinates, contexts, num_bases=bases, rank=8,
            temperature=0.01,
        )
        conditional.append({
            "bases": bases,
            "rho": conditional_reachability_score(model, coordinates, contexts),
            "coefficients": model.coefficients(coordinates).tolist(),
        })

    future = []
    for horizon in (1, 2, 4, 8):
        losses = []
        for _, target in contexts:
            privileged = backward_future_targets(target, horizon=horizon)
            losses.append(float(future_state_loss(privileged, target)))
        future.append({"horizon": horizon, "current_to_privileged_loss": sum(losses) / len(losses)})

    population = LatentPopulation(max_size=len(contexts))
    for scene, (_, target) in zip(scenes, contexts, strict=True):
        population.add(PopulationItem(target, 1.0, scene))
    retrieval = []
    for scene, (_, target) in zip(scenes, contexts, strict=True):
        weights = population.retrieval_weights(target.mean(0), temperature=1.0)
        retrieval.append({"scene_id": scene, "weights": weights.tolist(), "top_item": population.items[int(weights.argmax())].item_id})

    report = {
        "protocol": "real-qwen-four-ideas-mechanism-v1",
        "contexts": scenes,
        "state_weight_reachability": reachability,
        "reachable_unreachable_decomposition": decomposition,
        "kv_conditioned_lora_basis": conditional,
        "privileged_future": future,
        "latent_population": retrieval,
        "scope_warning": f"Uses {len(contexts)} real causal quality-KV teacher contexts. Conditional/future/population results are mechanism checks; broader behavior evaluation remains required.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
