"""First-order state-to-weight reachability analysis.

The runtime approximation used here is the same linearization as a LoRA
projection: ``delta_y ~= X @ delta_W.T``.  ``X`` can be hidden-state features
or Jacobian rows collected at the intervention sites.  The module deliberately
keeps the analysis model-agnostic so it can run on CPU over saved activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_closed_form import fit_low_rank_delta


@dataclass(frozen=True)
class ReachabilityFit:
    """Low-rank fit and normalized explained energy for one or more contexts."""

    delta_weight: torch.Tensor
    prediction: torch.Tensor
    residual: torch.Tensor
    rho: float
    target_energy: float
    lora_a: torch.Tensor
    lora_b: torch.Tensor

    @property
    def reachable_component(self) -> torch.Tensor:
        return self.prediction

    @property
    def unreachable_component(self) -> torch.Tensor:
        return -self.residual


def _validate_pair(features: torch.Tensor, target: torch.Tensor) -> None:
    if features.ndim != 2 or target.ndim != 2:
        raise ValueError("features and target must be rank-2 tensors")
    if features.shape[0] != target.shape[0]:
        raise ValueError("features and target must have the same sample count")
    if features.shape[0] == 0:
        raise ValueError("at least one intervention site is required")


def fit_reachability(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    ridge: float = 1e-3,
) -> ReachabilityFit:
    """Fit the best available rank-r LoRA effect for one context or a pool.

    ``rho`` is the fraction of target squared norm explained by the fitted
    low-rank update, clipped to [0, 1] for numerical robustness.
    """

    _validate_pair(features, target)
    a, b = fit_low_rank_delta(features, target, rank=rank, ridge=ridge)
    delta_weight = b @ a
    prediction = features @ delta_weight.T
    residual = prediction - target
    target_energy = float(target.float().square().sum())
    error = float(residual.float().square().sum())
    rho = 1.0 if target_energy == 0 else 1.0 - error / (target_energy + 1e-12)
    return ReachabilityFit(
        delta_weight=delta_weight,
        prediction=prediction,
        residual=residual,
        rho=float(min(1.0, max(0.0, rho))),
        target_energy=target_energy,
        lora_a=a,
        lora_b=b,
    )


def split_state_correction(
    features: torch.Tensor,
    kv_correction: torch.Tensor,
    *,
    rank: int,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, ReachabilityFit]:
    """Orthogonally split a useful KV correction into slow and fast parts.

    The fitted prediction is the rank-r parameter-reachable component.  The
    residual is the context-specific component that remains in fast memory.
    ``kv_correction = reachable + unreachable`` up to floating point error.
    """

    fit = fit_reachability(features, kv_correction, rank=rank, ridge=ridge)
    reachable = fit.reachable_component
    unreachable = kv_correction - reachable
    return reachable, unreachable, fit


def fit_individual_and_shared(
    contexts: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    rank: int,
    ridge: float = 1e-3,
) -> dict[str, object]:
    """Compare per-context and shared reachability.

    Every tuple is ``(features, target_state_shift)``.  The shared fit is
    computed on the concatenated intervention sites, while individual fits
    expose the context heterogeneity that a single adapter cannot represent.
    """

    if not contexts:
        raise ValueError("contexts must not be empty")
    individual = [fit_reachability(x, y, rank=rank, ridge=ridge) for x, y in contexts]
    features = torch.cat([x for x, _ in contexts], dim=0)
    target = torch.cat([y for _, y in contexts], dim=0)
    shared = fit_reachability(features, target, rank=rank, ridge=ridge)
    return {
        "individual": individual,
        "shared": shared,
        "rho_individual_mean": sum(f.rho for f in individual) / len(individual),
        "rho_shared": shared.rho,
        "context_count": len(contexts),
    }


def rank_scaling(
    contexts: list[tuple[torch.Tensor, torch.Tensor]],
    ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    *,
    ridge: float = 1e-3,
) -> list[dict[str, float | int]]:
    """Return shared and individual reachability as rank increases."""

    output: list[dict[str, float | int]] = []
    for rank in ranks:
        result = fit_individual_and_shared(contexts, rank=rank, ridge=ridge)
        output.append({
            "rank": rank,
            "rho_shared": float(result["shared"].rho),
            "rho_individual_mean": float(result["rho_individual_mean"]),
        })
    return output
