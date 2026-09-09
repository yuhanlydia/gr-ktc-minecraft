#!/usr/bin/env python3
"""Run the four new research ideas over the complete MineExplorer task index.

This runner is intentionally API-free.  It consumes the task/context records,
creates deterministic activation-shaped tensors for a mechanism study, and
writes the exact same result schema that a real KV recorder can later fill.
Use ``--real-artifacts`` in a future run to replace the tensor factory with
saved model activations without changing the analysis code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.conditional_lora import fit_conditional_lora_basis
from gr_ktc.future_distillation import future_kl_loss, future_privileged_target, future_state_loss
from gr_ktc.latent_population import LatentPopulation, PopulationItem
from gr_ktc.reachability import fit_individual_and_shared, fit_reachability, rank_scaling, split_state_correction


def _difficulty(scenario) -> float:
    graph = scenario.reasoning_graph or {}
    hops = len(graph.get("edges", ())) or len(scenario.selected_tasks) or 1
    return min(1.0, 0.15 + 0.08 * hops + 0.04 * len(scenario.milestones))


def _make_contexts(scenarios, *, seed: int, feature_dim: int = 24, hidden_dim: int = 16):
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(hidden_dim, feature_dim, generator=generator)
    contexts = []
    coordinates = []
    trajectories = []
    scores = []
    metadata = []
    for index, scenario in enumerate(scenarios):
        difficulty = _difficulty(scenario)
        local = torch.Generator().manual_seed(seed + 10007 * (index + 1))
        sites = 8
        features = torch.randn(sites, feature_dim, generator=local)
        coordinate = torch.randn(8, generator=local)
        # The first term is globally reachable; the second is a context-specific
        # state correction, making joint reachability degrade with heterogeneity.
        context_update = torch.randn(hidden_dim, feature_dim, generator=local)
        context_update = context_update / context_update.norm().clamp_min(1e-6)
        target = features @ (shared + difficulty * context_update).T
        target += 0.03 * torch.randn(target.shape, generator=local)
        contexts.append((features, target))
        coordinates.append(coordinate)
        trajectories.append(target.float())
        scores.append(float(1.0 - difficulty + 0.05 * torch.randn((), generator=local)))
        metadata.append({"scene_id": scenario.scene_id, "difficulty": difficulty, "task_text": scenario.task_text})
    return contexts, torch.stack(coordinates), trajectories, torch.tensor(scores), metadata


def run(args: argparse.Namespace) -> dict:
    # Small per-context SVDs are substantially faster without BLAS spawning a
    # fresh team for every scenario.  The setting is local to this process and
    # does not affect the 24GB model-serving workers.
    torch.set_num_threads(max(1, args.torch_threads))
    all_scenarios = list(load_mineexplorer(args.scenarios))
    if args.max_contexts:
        scenarios = all_scenarios[: args.max_contexts]
    else:
        scenarios = all_scenarios
    contexts, coordinates, trajectories, scores, metadata = _make_contexts(
        scenarios, seed=args.seed, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim
    )

    # Idea 1: individual-vs-shared state-to-weight reachability and rank curve.
    probe_count = min(args.reachability_probe, len(contexts))
    probe = contexts[:probe_count]
    reachability = fit_individual_and_shared(probe, rank=args.rank, ridge=args.ridge)
    curve = rank_scaling(probe, tuple(args.ranks), ridge=args.ridge)
    per_context = []
    for i in range(len(contexts)):
        _, unreachable, fit = split_state_correction(
            contexts[i][0], contexts[i][1], rank=args.rank, ridge=args.ridge
        )
        per_context.append({
            "scene_id": metadata[i]["scene_id"],
            "difficulty": metadata[i]["difficulty"],
            "rho": fit.rho,
            "reachable_energy_fraction": float(fit.reachable_component.float().square().sum() / contexts[i][1].float().square().sum().clamp_min(1e-12)),
            "unreachable_energy_fraction": float(unreachable.float().square().sum() / contexts[i][1].float().square().sum().clamp_min(1e-12)),
        })

    # Idea 2: continuous latent-coordinate conditioned LoRA basis.
    pooled_features = torch.stack([features.mean(0) for features, _ in contexts])
    pooled_targets = torch.stack([target.mean(0) for _, target in contexts])
    basis = fit_conditional_lora_basis(
        coordinates, pooled_features, pooled_targets,
        num_bases=args.num_bases, rank=min(args.rank, args.feature_dim, args.hidden_dim),
        ridge=args.ridge, temperature=args.temperature,
    )
    coeff = basis.coefficients(coordinates)
    conditional = {
        "num_bases": basis.num_bases,
        "coefficient_entropy": float((-(coeff * coeff.clamp_min(1e-12).log()).sum(-1)).mean()),
        "coefficient_shape": list(coeff.shape),
    }

    # Idea 3: privileged successful future -> current-state target.
    horizon = 4
    teacher_future = torch.stack([
        torch.stack([trajectory.mean(0) + 0.05 * h * torch.ones(args.hidden_dim) for h in range(horizon)])
        for trajectory in trajectories
    ])
    current = torch.stack([trajectory[0] for trajectory in trajectories])
    privileged = future_privileged_target(current, teacher_future, horizon_decay=args.horizon_decay)
    student_before = current + 0.2 * torch.randn(current.shape, generator=torch.Generator().manual_seed(args.seed + 9))
    student_after = 0.5 * student_before + 0.5 * privileged
    future = {
        "state_loss_before": float(future_state_loss(privileged, student_before)),
        "state_loss_after": float(future_state_loss(privileged, student_after)),
        "kl_self_check": float(future_kl_loss(teacher_future, teacher_future)),
        "horizon": horizon,
    }

    # Idea 4: population archive, selection and latent retrieval.
    population = LatentPopulation(max_size=args.population_size)
    for i, (trajectory, score) in enumerate(zip(trajectories, scores, strict=True)):
        population.add(PopulationItem(trajectory, float(score), metadata[i]["scene_id"]))
    before = len(population)
    if population:
        query = torch.zeros(trajectories[0].shape[-1])
        query[: min(query.numel(), coordinates.shape[-1])] = coordinates[0][: query.numel()]
        retrieved = population.retrieve(query, temperature=args.temperature, top_k=min(8, len(population)))
        population.evolve([], elite_fraction=0.5, mutation_std=0.0, seed=args.seed)
        population_result = {
            "size_before_after": [before, len(population)],
            "retrieved_shape": list(retrieved.shape),
            "top_score": max(item.score for item in population.items),
        }
    else:
        population_result = {"size_before_after": [0, 0]}

    result = {
        "protocol": "four-ideas-api-free-mechanism-v1",
        "seed": args.seed,
        "scenario_file": str(args.scenarios),
        "scenario_count": len(scenarios),
        "available_scenario_count": len(all_scenarios),
        "scope_warning": "Tensor mechanism study; replace synthetic activation factory with recorded model KV/hidden states for causal claims.",
        "state_weight_reachability": {
            "probe_contexts": probe_count,
            "rank": args.rank,
            "rho_individual_mean": float(reachability["rho_individual_mean"]),
            "rho_shared": float(reachability["shared"].rho),
            "rank_curve": curve,
            "per_context": per_context,
        },
        "conditional_lora_basis": conditional,
        "future_privileged_distillation": future,
        "latent_population": population_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("data/MineExplorer-Benchmark/benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/four_new_ideas.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-contexts", type=int, default=0, help="0 means all scenarios")
    parser.add_argument("--reachability-probe", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--horizon-decay", type=float, default=0.9)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
