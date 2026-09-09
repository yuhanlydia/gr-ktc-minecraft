from __future__ import annotations

import torch


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Differentiable soft-DTW with squared Euclidean local cost."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be [time, same_features]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    distances = torch.cdist(x, y).square()
    rows: list[list[torch.Tensor]] = []
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    rows.append([zero] + [inf] * y.shape[0])
    for i in range(1, x.shape[0] + 1):
        row = [inf]
        for j in range(1, y.shape[0] + 1):
            previous = torch.stack((rows[i - 1][j], row[j - 1], rows[i - 1][j - 1]))
            soft_min = -gamma * torch.logsumexp(-previous / gamma, dim=0)
            row.append(distances[i - 1, j - 1] + soft_min)
        rows.append(row)
    return rows[-1][-1]


def linear_resample(sequence: torch.Tensor, phases: int) -> torch.Tensor:
    if sequence.ndim != 2 or sequence.shape[0] == 0 or phases < 2:
        raise ValueError("invalid sequence or phase count")
    if sequence.shape[0] == 1:
        return sequence.expand(phases, -1).clone()
    positions = torch.linspace(0, sequence.shape[0] - 1, phases, device=sequence.device)
    left = positions.floor().long()
    right = positions.ceil().long()
    alpha = (positions - left).unsqueeze(1).to(sequence.dtype)
    return sequence[left] * (1 - alpha) + sequence[right] * alpha


def soft_dtw_barycenter(
    sequences: list[torch.Tensor],
    weights: torch.Tensor,
    phases: int = 16,
    gamma: float = 0.1,
    steps: int = 30,
    learning_rate: float = 0.05,
) -> torch.Tensor:
    if len(sequences) != weights.numel() or len(sequences) == 0:
        raise ValueError("sequence and weight counts must match and be non-empty")
    normalized = weights.float() / weights.sum().clamp_min(1e-12)
    initial = sum(
        float(weight) * linear_resample(sequence, phases)
        for sequence, weight in zip(sequences, normalized, strict=True)
    )
    barycenter = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([barycenter], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(
            weight * soft_dtw(barycenter, sequence, gamma)
            for sequence, weight in zip(sequences, normalized, strict=True)
        )
        loss.backward()
        optimizer.step()
    return barycenter.detach()


def merge_b(
    sequences: list[torch.Tensor],
    advantages: torch.Tensor,
    phases: int = 16,
    gamma: float = 0.1,
    temperature: float = 1.0,
    negative_scale: float = 1.0,
    steps: int = 30,
) -> torch.Tensor:
    positive = advantages > 0
    if not positive.any():
        raise ValueError("merge B requires a positive-advantage rollout")
    pos_sequences = [s for s, keep in zip(sequences, positive, strict=True) if bool(keep)]
    pos_weights = torch.softmax(advantages[positive] / temperature, dim=0)
    positive_center = soft_dtw_barycenter(
        pos_sequences, pos_weights, phases, gamma, steps
    )
    negative = advantages < 0
    if not negative.any():
        return positive_center[1:] - positive_center[:-1]
    neg_sequences = [s for s, keep in zip(sequences, negative, strict=True) if bool(keep)]
    neg_weights = torch.softmax(-advantages[negative] / temperature, dim=0)
    negative_center = soft_dtw_barycenter(
        neg_sequences, neg_weights, phases, gamma, steps
    )
    return (positive_center[1:] - positive_center[:-1]) - negative_scale * (
        negative_center[1:] - negative_center[:-1]
    )

