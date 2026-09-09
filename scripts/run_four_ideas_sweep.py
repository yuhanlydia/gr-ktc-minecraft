#!/usr/bin/env python3
"""Run the preregistered reachability/heterogeneity sweep.

This is the inexpensive, API-free stage of the long goal.  It uses every
MineExplorer scenario for the conditional-basis and population analyses and a
fixed 32-context probe for repeated shared-LoRA fits, which keeps the 24GB
GPU/model free for later causal runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.conditional_lora import fit_conditional_lora_basis
from gr_ktc.future_distillation import future_privileged_target, future_state_loss
from gr_ktc.latent_population import LatentPopulation, PopulationItem
from gr_ktc.reachability import fit_individual_and_shared, rank_scaling
from scripts.run_new_ideas import _make_contexts


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(max(1, args.torch_threads))
    scenarios = list(load_mineexplorer(args.scenarios))
    contexts, coordinates, trajectories, scores, metadata = _make_contexts(scenarios, seed=args.seed)
    probe = contexts[: min(args.probe_contexts, len(contexts))]

    rank_curve = rank_scaling(probe, tuple(args.ranks), ridge=args.ridge)
    heterogeneity = []
    for count in args.context_counts:
        count = min(count, len(probe))
        fit = fit_individual_and_shared(probe[:count], rank=args.rank, ridge=args.ridge)
        heterogeneity.append({
            "contexts": count,
            "rho_individual_mean": float(fit["rho_individual_mean"]),
            "rho_joint": float(fit["shared"].rho),
        })

    pooled_features = torch.stack([x.mean(0) for x, _ in contexts])
    pooled_targets = torch.stack([y.mean(0) for _, y in contexts])
    basis_sweep = []
    for count in args.basis_counts:
        basis = fit_conditional_lora_basis(
            coordinates, pooled_features, pooled_targets,
            num_bases=count, rank=args.rank, ridge=args.ridge,
        )
        weights = basis.coefficients(coordinates)
        basis_sweep.append({
            "bases": basis.num_bases,
            "coefficient_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean()),
        })

    current = torch.stack([trajectory[0] for trajectory in trajectories])
    future_sweep = []
    for horizon in args.horizons:
        future = torch.stack([
            torch.stack([state + 0.05 * h * torch.ones_like(state) for h in range(horizon)])
            for state in current
        ])
        target = future_privileged_target(current, future)
        before = current + 0.2 * torch.randn(current.shape, generator=torch.Generator().manual_seed(args.seed + horizon))
        after = 0.5 * before + 0.5 * target
        future_sweep.append({
            "horizon": horizon,
            "loss_before": float(future_state_loss(target, before)),
            "loss_after": float(future_state_loss(target, after)),
        })

    population_sweep = []
    for size in args.population_sizes:
        population = LatentPopulation(max_size=size)
        for i, (trajectory, score) in enumerate(zip(trajectories, scores, strict=True)):
            population.add(PopulationItem(trajectory, float(score), metadata[i]["scene_id"]))
        query = torch.zeros(trajectories[0].shape[-1])
        query[: coordinates.shape[-1]] = coordinates[0]
        retrieved = population.retrieve(query, top_k=min(args.top_k, len(population)))
        population_sweep.append({
            "population_size": size,
            "retained": len(population),
            "retrieved_tokens": retrieved.shape[0],
        })

    result = {
        "protocol": "four-ideas-sweep-api-free-v1",
        "scenario_count": len(scenarios),
        "probe_contexts": len(probe),
        "rank_scaling": rank_curve,
        "context_heterogeneity": heterogeneity,
        "conditional_lora_basis": basis_sweep,
        "future_horizon": future_sweep,
        "population_scaling": population_sweep,
        "scope_warning": "Mechanism sweep over deterministic activation-shaped tensors; causal model claims require recorded KV/hidden effects.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("data/MineExplorer-Benchmark/benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/four_ideas_sweep.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-contexts", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--context-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--basis-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

