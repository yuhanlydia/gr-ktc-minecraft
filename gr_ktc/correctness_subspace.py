from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CorrectnessSubspace:
    positive: torch.Tensor
    negative: torch.Tensor
    positive_eigenvalues: torch.Tensor
    negative_eigenvalues: torch.Tensor


def fit_group_relative_covariance(
    residual_groups: list[list[torch.Tensor]],
    advantage_groups: list[torch.Tensor],
) -> torch.Tensor:
    if len(residual_groups) != len(advantage_groups):
        raise ValueError("residual and advantage group counts differ")
    covariance: torch.Tensor | None = None
    normalizer = 0.0
    for residuals, advantages in zip(residual_groups, advantage_groups, strict=True):
        if len(residuals) != advantages.numel():
            raise ValueError("rollout and advantage counts differ within a group")
        for sequence, advantage in zip(residuals, advantages, strict=True):
            if sequence.ndim != 2:
                raise ValueError("each residual sequence must be rank 2")
            r = sequence.double()
            if covariance is None:
                covariance = torch.zeros((r.shape[1], r.shape[1]), dtype=torch.float64)
            covariance.add_(r.T @ r, alpha=float(advantage))
            normalizer += abs(float(advantage)) * r.shape[0]
    if covariance is None:
        raise ValueError("no residual sequences supplied")
    return covariance / max(normalizer, 1.0)


def fit_correctness_subspace(covariance: torch.Tensor, rank: int) -> CorrectnessSubspace:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if rank <= 0:
        raise ValueError("rank must be positive")
    eigenvalues, eigenvectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    positive_idx = torch.where(eigenvalues > 0)[0][-rank:].flip(0)
    negative_idx = torch.where(eigenvalues < 0)[0][:rank]
    return CorrectnessSubspace(
        positive=eigenvectors[:, positive_idx],
        negative=eigenvectors[:, negative_idx],
        positive_eigenvalues=eigenvalues[positive_idx],
        negative_eigenvalues=eigenvalues[negative_idx],
    )

