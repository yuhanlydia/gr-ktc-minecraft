"""Grassmann geometry for cross-context native-KV memory subspaces."""

from __future__ import annotations

import itertools
import math

import torch


def latent_subspace(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Return an orthonormal basis for the matrix's latent feature directions.

    ``matrix`` is ``[samples_or_phases, latent_dim]``.  The returned right
    singular basis is ``[latent_dim, effective_rank]``; comparing the left
    singular vectors would compare token indices rather than K/V directions.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be non-empty and rank-2")
    if rank < 1:
        raise ValueError("rank must be positive")
    effective = min(rank, *matrix.shape)
    _, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return vh[:effective].T.contiguous()


def principal_angles(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Principal angles in radians between equal-rank orthonormal bases."""
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("bases must be rank-2")
    if first.shape != second.shape:
        raise ValueError("bases must have equal ambient dimension and rank")
    singular = torch.linalg.svdvals(first.float().T @ second.float())
    return torch.acos(singular.clamp(-1, 1))


def grassmann_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Canonical geodesic distance, the L2 norm of principal angles."""
    return float(principal_angles(first, second).square().sum().sqrt())


def consensus_spectrum(bases: list[torch.Tensor]) -> torch.Tensor:
    """Nonzero eigenvalues of mean projection C_N without a dense projector."""
    if not bases:
        raise ValueError("at least one basis is required")
    shape = bases[0].shape
    if any(basis.shape != shape for basis in bases):
        raise ValueError("all bases must have equal shape")
    stacked = torch.cat([basis.float() for basis in bases], dim=1) / math.sqrt(len(bases))
    values = torch.linalg.svdvals(stacked).square().clamp(0, 1)
    return values.sort(descending=True).values


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, matching the standard Spearman definition."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2
        position = end
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("x and y must be equal vectors with at least two values")
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = rx.norm() * ry.norm()
    return 0.0 if float(denominator) == 0 else float((rx @ ry) / denominator)


def exact_permutation_pvalue(x: torch.Tensor, y: torch.Tensor) -> tuple[float, int]:
    """Exact two-sided permutation p-value for small paired vectors."""
    observed = abs(spearman_correlation(x, y))
    exceed, total = 0, 0
    for permutation in itertools.permutations(range(len(y))):
        statistic = abs(spearman_correlation(x, y[list(permutation)]))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total, total


def permutation_pvalue(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, int, bool]:
    """Two-sided permutation p-value; exact up to eight pairs, Monte Carlo above."""
    if len(x) <= 8:
        p, count = exact_permutation_pvalue(x, y)
        return p, count, True
    if samples < 1:
        raise ValueError("samples must be positive")
    observed = abs(spearman_correlation(x, y))
    generator = torch.Generator().manual_seed(seed)
    exceed = 0
    for _ in range(samples):
        statistic = abs(spearman_correlation(x, y[torch.randperm(len(y), generator=generator)]))
        exceed += statistic >= observed - 1e-12
    # Add-one correction makes the Monte Carlo estimate finite and valid.
    return (exceed + 1) / (samples + 1), samples, False


def context_bootstrap_spearman_interval(
    pair_values: dict[tuple[str, str], tuple[float, float]],
    contexts: list[str],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, int]:
    """Cluster bootstrap contexts, preserving pair dependence.

    A resampled context may occur multiple times. Pairs between distinct
    original contexts are repeated according to their bootstrap multiplicity;
    self-pairs are omitted because their distance/interference is structurally
    zero and was not part of the observed sample.
    """
    if len(contexts) < 3 or samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid context bootstrap settings")
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(samples):
        sampled = [contexts[index] for index in torch.randint(
            len(contexts), (len(contexts),), generator=generator,
        ).tolist()]
        x, y = [], []
        for first_index in range(len(sampled)):
            for second_index in range(first_index + 1, len(sampled)):
                first, second = sampled[first_index], sampled[second_index]
                if first == second:
                    continue
                key = tuple(sorted((first, second)))
                distance, interference = pair_values[key]
                x.append(distance)
                y.append(interference)
        if len(x) >= 3 and len(set(x)) >= 2 and len(set(y)) >= 2:
            estimates.append(spearman_correlation(torch.tensor(x), torch.tensor(y)))
    if not estimates:
        raise ValueError("all context bootstrap samples were degenerate")
    estimates_tensor = torch.tensor(estimates)
    tail = (1 - confidence) / 2
    return (
        float(estimates_tensor.quantile(tail)),
        float(estimates_tensor.quantile(1 - tail)),
        len(estimates),
    )


def bootstrap_spearman_interval(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    samples: int = 10_000,
    confidence: float = 0.90,
    seed: int = 42,
) -> tuple[float, float]:
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    # Degenerate resamples have no defined rank correlation; discarding them
    # is explicit and preferable to silently treating them as evidence for 0.
    for _ in range(samples):
        index = torch.randint(len(x), (len(x),), generator=generator)
        if x[index].unique().numel() < 2 or y[index].unique().numel() < 2:
            continue
        estimates.append(spearman_correlation(x[index], y[index]))
    if not estimates:
        raise ValueError("all bootstrap samples were degenerate")
    estimates = torch.tensor(estimates)
    tail = (1 - confidence) / 2
    return float(estimates.quantile(tail)), float(estimates.quantile(1 - tail))
