#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gr_ktc.correctness_subspace import (
    fit_correctness_subspace,
    fit_group_relative_covariance,
)
from gr_ktc.group_advantage import group_relative_advantage
from gr_ktc.lora_closed_form import fit_low_rank_delta
from gr_ktc.merge_a_residual import merge_a
from gr_ktc.merge_b_softdtw import merge_b
from gr_ktc.merge_c_sinkhorn import merge_c
from gr_ktc.schrodinger_bridge import merge_schrodinger_bridge


def pairwise_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).float().mean() + 0.5 * (comparisons == 0).float().mean())


def run(seed: int, groups: int = 90, rollouts: int = 4, dim: int = 32) -> dict:
    generator = torch.Generator().manual_seed(seed)
    true_direction = F.normalize(torch.randn(dim, generator=generator), dim=0)
    residual_groups: list[list[torch.Tensor]] = []
    advantage_groups: list[torch.Tensor] = []
    score_groups: list[torch.Tensor] = []

    for _ in range(groups):
        scores = torch.randint(0, 2, (rollouts,), generator=generator).float()
        # Force mixed supervision so the mechanism test is identifiable.
        if scores.min() == scores.max():
            scores[0] = 1 - scores[0]
        sequences = []
        for score in scores:
            length = int(torch.randint(12, 25, (), generator=generator))
            noise = 0.35 * torch.randn(length, dim, generator=generator)
            phase = torch.linspace(0.3, 1.0, length).unsqueeze(1)
            signal = float(score) * phase * true_direction
            sequences.append(noise + signal)
        residual_groups.append(sequences)
        score_groups.append(scores)
        advantage_groups.append(group_relative_advantage(scores))

    covariance = fit_group_relative_covariance(residual_groups, advantage_groups)
    subspace = fit_correctness_subspace(covariance, rank=8)
    # Covariance fitting intentionally accumulates in float64 for numerical
    # stability, whereas synthetic trajectories follow the model's float32
    # activation dtype.  Cast the learned basis at this boundary, just as the
    # runtime injector casts stored memories to the active model dtype.
    positive_basis = subspace.positive.to(dtype=residual_groups[0][0].dtype)
    positive_scores, negative_scores = [], []
    for sequences, scores in zip(residual_groups, score_groups, strict=True):
        for sequence, score in zip(sequences, scores, strict=True):
            energy = (sequence @ positive_basis).square().mean()
            (positive_scores if score else negative_scores).append(energy)
    auc = pairwise_auc(torch.stack(positive_scores), torch.stack(negative_scores))

    mixed_sequences = residual_groups[0]
    advantages = advantage_groups[0]
    projected = [sequence @ positive_basis for sequence in mixed_sequences]
    projected_truth = F.normalize(true_direction @ positive_basis, dim=0)
    a_target = merge_a(projected, advantages)
    a_alignment = abs(float(F.cosine_similarity(a_target, projected_truth, dim=0)))

    b_target = merge_b(projected, advantages, phases=8, steps=5)
    b_direction = F.normalize(b_target.mean(dim=0), dim=0)
    b_alignment = abs(float(F.cosine_similarity(b_direction, projected_truth, dim=0)))

    c_target = merge_c(
        projected, advantages, phases=4, support_points=2, steps=3
    )
    c_direction = F.normalize(c_target.mean(dim=(0, 1)), dim=0)
    c_alignment = abs(float(F.cosine_similarity(c_direction, projected_truth, dim=0)))

    d_target, bridge = merge_schrodinger_bridge(
        projected, advantages, phases=8, epsilon=0.2, temporal_cost=2.0
    )
    d_direction = F.normalize(d_target.mean(dim=0), dim=0)
    d_alignment = abs(float(F.cosine_similarity(d_direction, projected_truth, dim=0)))

    x = torch.randn(128, 24, generator=generator)
    teacher_delta = torch.randn(24, 12, generator=generator)
    u, s, vh = torch.linalg.svd(teacher_delta, full_matrices=False)
    teacher_delta = (u[:, :4] * s[:4]) @ vh[:4]
    delta_y = x @ teacher_delta
    a, b = fit_low_rank_delta(x, delta_y, rank=4, ridge=1e-6)
    student_delta_y = x @ (b @ a).T
    relative_error = float((student_delta_y - delta_y).norm() / delta_y.norm())

    return {
        "seed": seed,
        "groups": groups,
        "rollouts_per_group": rollouts,
        "latent_ranking_auc": auc,
        "merge_alignment": {
            "A": a_alignment,
            "B": b_alignment,
            "C": c_alignment,
            "D-SB": d_alignment,
        },
        "schrodinger_control_energy": float(
            (bridge.source_marginal[:, None] * bridge.control.square()).sum()
        ),
        "closed_form_lora_relative_error": relative_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_gate.json"))
    args = parser.parse_args()
    results = [run(seed) for seed in (42, 43, 44)]
    summary = {
        "protocol": "synthetic-mechanism-gate-v1",
        "runs": results,
        "mean_latent_ranking_auc": sum(r["latent_ranking_auc"] for r in results) / len(results),
        "mean_lora_relative_error": sum(r["closed_form_lora_relative_error"] for r in results) / len(results),
        "scope_warning": "Mechanism sanity check only; not Minecraft task success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
