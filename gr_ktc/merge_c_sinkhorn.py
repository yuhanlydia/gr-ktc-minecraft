from __future__ import annotations

import torch

from .merge_b_softdtw import linear_resample


def sinkhorn_cost(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.1,
    iterations: int = 50,
) -> torch.Tensor:
    """Entropic balanced OT cost between uniformly weighted token clouds."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("clouds must be [support, same_features]")
    cost = torch.cdist(source, target).square()
    log_a = torch.full(
        (source.shape[0],), -torch.log(torch.tensor(float(source.shape[0]))),
        device=source.device, dtype=source.dtype,
    )
    log_b = torch.full(
        (target.shape[0],), -torch.log(torch.tensor(float(target.shape[0]))),
        device=target.device, dtype=target.dtype,
    )
    log_kernel = -cost / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    transport = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return (transport * cost).sum()


def _phase_clouds(sequence: torch.Tensor, phases: int) -> list[torch.Tensor]:
    boundaries = torch.linspace(0, sequence.shape[0], phases + 1).round().long()
    clouds = []
    for index in range(phases):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        if end <= start:
            clouds.append(linear_resample(sequence, phases)[index : index + 1])
        else:
            clouds.append(sequence[start:end])
    return clouds


def sinkhorn_phase_barycenter(
    sequences: list[torch.Tensor],
    weights: torch.Tensor,
    phases: int = 16,
    support_points: int = 4,
    epsilon: float = 0.1,
    steps: int = 30,
    learning_rate: float = 0.05,
) -> torch.Tensor:
    """Learn a fixed-support Wasserstein barycenter independently per phase."""
    weights = weights.float() / weights.sum().clamp_min(1e-12)
    grouped = [_phase_clouds(sequence, phases) for sequence in sequences]
    outputs = []
    for phase in range(phases):
        initial_centroid = sum(
            weight * clouds[phase].mean(dim=0)
            for clouds, weight in zip(grouped, weights, strict=True)
        )
        support = initial_centroid.expand(support_points, -1).clone()
        support += 1e-3 * torch.randn_like(support)
        support.requires_grad_(True)
        optimizer = torch.optim.Adam([support], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                weight * sinkhorn_cost(support, clouds[phase], epsilon)
                for clouds, weight in zip(grouped, weights, strict=True)
            )
            loss.backward()
            optimizer.step()
        outputs.append(support.detach())
    return torch.stack(outputs)


def merge_c(
    sequences: list[torch.Tensor],
    advantages: torch.Tensor,
    phases: int = 16,
    support_points: int = 4,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    steps: int = 30,
) -> torch.Tensor:
    positive = advantages > 0
    if not positive.any():
        raise ValueError("merge C requires a positive-advantage rollout")
    selected = [s for s, keep in zip(sequences, positive, strict=True) if bool(keep)]
    weights = torch.softmax(advantages[positive] / temperature, dim=0)
    return sinkhorn_phase_barycenter(
        selected, weights, phases, support_points, epsilon, steps
    )

