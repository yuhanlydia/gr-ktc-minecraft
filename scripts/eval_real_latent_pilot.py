#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.correctness_subspace import (
    fit_correctness_subspace,
    fit_group_relative_covariance,
)
from gr_ktc.group_advantage import group_relative_advantage
from gr_ktc.whitening import fit_whitening


def load_groups(pools: list[Path], layer: int):
    groups = []
    group_dirs = [
        group_dir for pool in pools
        for group_dir in sorted((pool / "groups").iterdir())
    ]
    for group_dir in group_dirs:
        rollouts = []
        for trajectory_dir in sorted((group_dir / "trajectories").iterdir()):
            metadata = json.loads((trajectory_dir / "metadata.json").read_text())
            kv = load_file(trajectory_dir / "kv.safetensors")[f"layer_{layer}_kv"].float()
            rollouts.append((int(metadata["rollout_index"]), float(metadata["verifier_score"]), kv))
        rollouts.sort()
        if len(rollouts) != 4:
            raise ValueError(f"incomplete group: {group_dir}")
        scores = torch.tensor([rollout[1] for rollout in rollouts])
        if float(scores.std(unbiased=False)) > 0:
            groups.append({
                "group_id": group_dir.name,
                "scores": scores,
                "residuals": [rollout[2][1:] - rollout[2][:-1] for rollout in rollouts],
            })
    return groups


def ranking_accuracy(metric: torch.Tensor, outcomes: torch.Tensor) -> tuple[int, int]:
    correct = total = 0
    for i in range(len(outcomes)):
        for j in range(len(outcomes)):
            if outcomes[i] <= outcomes[j]:
                continue
            total += 1
            correct += int(metric[i] > metric[j])
    return correct, total


def _unit_mean(sequence: torch.Tensor) -> torch.Tensor:
    unit = sequence / sequence.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return unit.mean(0)


def _direction(train_residual_groups, train_advantages) -> torch.Tensor:
    terms = []
    weights = []
    for residuals, advantages in zip(
        train_residual_groups, train_advantages, strict=True
    ):
        for residual, advantage in zip(residuals, advantages, strict=True):
            if float(advantage) == 0:
                continue
            terms.append(_unit_mean(residual) * advantage)
            weights.append(abs(float(advantage)))
    direction = torch.stack(terms).sum(0) / max(sum(weights), 1e-8)
    return direction / direction.norm().clamp_min(1e-8)


def _centroids(train_residual_groups, train_advantages):
    positive, negative = [], []
    for residuals, advantages in zip(
        train_residual_groups, train_advantages, strict=True
    ):
        for residual, advantage in zip(residuals, advantages, strict=True):
            summary = _unit_mean(residual)
            if float(advantage) > 0:
                positive.append(summary)
            elif float(advantage) < 0:
                negative.append(summary)
    return torch.stack(positive).mean(0), torch.stack(negative).mean(0)


def _bootstrap_group_accuracy(folds, metric_name: str, samples: int = 10_000):
    generator = random.Random(42)
    estimates = []
    for _ in range(samples):
        selected = [generator.choice(folds) for _ in folds]
        correct = sum(fold["metrics"][metric_name]["correct_pairs"] for fold in selected)
        pairs = sum(fold["metrics"][metric_name]["comparable_pairs"] for fold in selected)
        estimates.append(correct / pairs if pairs else 0.5)
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def _outcome_permutation_null(folds, metric_name: str, samples: int = 10_000):
    """Group-preserving random-label control for the final ranking statistic."""
    generator = random.Random(43)
    observed_correct = sum(
        fold["metrics"][metric_name]["correct_pairs"] for fold in folds
    )
    null_accuracies = []
    greater_or_equal = 0
    for _ in range(samples):
        correct = pairs = 0
        for fold in folds:
            shuffled = list(fold["scores"])
            generator.shuffle(shuffled)
            fold_correct, fold_pairs = ranking_accuracy(
                torch.tensor(fold["metrics"][metric_name]["values"]),
                torch.tensor(shuffled),
            )
            correct += fold_correct
            pairs += fold_pairs
        null_accuracies.append(correct / pairs if pairs else 0.5)
        greater_or_equal += correct >= observed_correct
    return {
        "mean_accuracy": sum(null_accuracies) / len(null_accuracies),
        "one_sided_p": (greater_or_equal + 1) / (samples + 1),
        "samples": samples,
    }


def evaluate_layer(groups, rank: int = 8):
    folds = []
    totals = {
        name: {"correct_pairs": 0, "comparable_pairs": 0}
        for name in ("signed_energy", "direction_cosine", "centroid_margin")
    }
    for heldout_index, heldout in enumerate(groups):
        train = [group for index, group in enumerate(groups) if index != heldout_index]
        train_values = torch.cat([
            residual for group in train for residual in group["residuals"]
        ])
        whitening = fit_whitening(
            train_values, max_rank=64, randomized=True, random_seed=42
        )
        train_residual_groups = [
            [whitening.transform(residual) for residual in group["residuals"]]
            for group in train
        ]
        train_advantages = [group_relative_advantage(group["scores"]) for group in train]
        covariance = fit_group_relative_covariance(
            train_residual_groups, train_advantages
        )
        subspace = fit_correctness_subspace(covariance, rank=rank)
        direction = _direction(train_residual_groups, train_advantages)
        positive_centroid, negative_centroid = _centroids(
            train_residual_groups, train_advantages
        )
        metric_values = {name: [] for name in totals}
        for residual in heldout["residuals"]:
            projected = whitening.transform(residual).double()
            positive = (projected @ subspace.positive).square().mean()
            negative = (projected @ subspace.negative).square().mean()
            summary = _unit_mean(projected)
            metric_values["signed_energy"].append(positive - negative)
            metric_values["direction_cosine"].append(
                torch.nn.functional.cosine_similarity(summary, direction, dim=0)
            )
            metric_values["centroid_margin"].append(
                -(summary - positive_centroid).square().sum()
                + (summary - negative_centroid).square().sum()
            )
        fold_metrics = {}
        for name, values in metric_values.items():
            metric = torch.stack(values).float()
            correct, pairs = ranking_accuracy(metric, heldout["scores"])
            totals[name]["correct_pairs"] += correct
            totals[name]["comparable_pairs"] += pairs
            fold_metrics[name] = {
                "values": metric.tolist(),
                "correct_pairs": correct,
                "comparable_pairs": pairs,
                "ranking_accuracy": correct / pairs if pairs else None,
            }
        folds.append({
            "heldout_group": heldout["group_id"],
            "scores": heldout["scores"].tolist(),
            "metrics": fold_metrics,
        })
    summary = {}
    for name, total in totals.items():
        pairs = total["comparable_pairs"]
        summary[name] = {
            **total,
            "ranking_accuracy": total["correct_pairs"] / pairs if pairs else None,
            "group_bootstrap_95ci": _bootstrap_group_accuracy(folds, name),
            "random_label_control": _outcome_permutation_null(folds, name),
        }
    return {"folds": folds, "metrics": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_latent_pilot.json")
    args = parser.parse_args()
    pools = args.pools or sorted(
        path.parent for path in (ROOT / "data/acquisition").glob(
            "mineexplorer_composite*/manifest.json"
        )
    )
    pools = [pool if pool.is_absolute() else ROOT / pool for pool in pools]
    total_groups = sum(len(list((pool / "groups").iterdir())) for pool in pools)
    mixed_groups = len(load_groups(pools, 24))
    result = {
        "protocol": "real-kv-leave-one-group-out-v2",
        "whitening": "train-fold-only randomized PCA rank 64, seed 42",
        "pools": [str(pool.relative_to(ROOT)) for pool in pools],
        "total_groups": total_groups,
        "total_trajectories": total_groups * 4,
        "mixed_groups": mixed_groups,
        "scope_warning": (
            "Gate 1 requires the 30-group/120-trajectory pilot, ranking above "
            "60%, a group-bootstrap CI excluding 0.5, and a random-label control."
        ),
        "layers": {},
    }
    for layer in (24, 35):
        groups = load_groups(pools, layer)
        if len(groups) < 2:
            raise RuntimeError("at least two mixed groups are required")
        result["layers"][str(layer)] = evaluate_layer(groups)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
