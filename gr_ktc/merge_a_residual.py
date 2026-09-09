from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_direction(
    directions: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    weights = torch.softmax(logits.float() / temperature, dim=0).to(directions)
    return (weights[:, None] * directions).sum(dim=0)


def merge_a(
    residual_sequences: list[torch.Tensor],
    advantages: torch.Tensor,
    temperature: float = 1.0,
    negative_scale: float = 1.0,
) -> torch.Tensor:
    """Advantage-weighted residual direction barycenter."""
    if len(residual_sequences) != advantages.numel():
        raise ValueError("sequence and advantage counts differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive = advantages > 0
    if not positive.any():
        raise ValueError("merge A requires at least one positive-advantage rollout")

    means = []
    for sequence in residual_sequences:
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            raise ValueError("residual sequences must be non-empty rank-2 tensors")
        means.append(F.normalize(sequence, dim=-1, eps=1e-6).mean(dim=0))
    mean_directions = torch.stack(means)
    d_pos = _weighted_direction(
        mean_directions[positive], advantages[positive], temperature
    )

    negative = advantages < 0
    if negative.any():
        d_neg = _weighted_direction(
            mean_directions[negative], -advantages[negative], temperature
        )
    else:
        d_neg = torch.zeros_like(d_pos)
    return F.normalize(d_pos - negative_scale * d_neg, dim=0, eps=1e-6)

