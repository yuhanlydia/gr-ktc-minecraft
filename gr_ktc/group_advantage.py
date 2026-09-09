from __future__ import annotations

import torch


def group_relative_advantage(
    scores: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardize verifier scores within one matched rollout group.

    Constant-outcome groups intentionally return zero: they contain no
    group-relative preference information.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be rank 1, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("scores must be non-empty")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")

    scores = scores.to(dtype=torch.float32)
    std = scores.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / (std + eps)


def positive_softmax_weights(
    advantages: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return normalized positive-only weights in the original shape."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = advantages > 0
    weights = torch.zeros_like(advantages, dtype=torch.float32)
    if mask.any():
        weights[mask] = torch.softmax(advantages[mask].float() / temperature, dim=0)
    return weights

