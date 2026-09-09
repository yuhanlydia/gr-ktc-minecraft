"""Privileged-future / hindsight latent-state distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def future_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distill a future-informed teacher distribution into current-state student."""

    if teacher_logits.shape != student_logits.shape or temperature <= 0:
        raise ValueError("logits must match and temperature must be positive")
    teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(student_log, teacher, reduction="none").sum(dim=-1)
    if token_mask is not None:
        if token_mask.shape != per_token.shape:
            raise ValueError("token_mask shape mismatch")
        per_token = per_token * token_mask
        return per_token.sum() / token_mask.sum().clamp_min(1)
    return per_token.mean()


def future_state_loss(
    teacher_states: torch.Tensor,
    student_states: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
    cosine_weight: float = 0.5,
) -> torch.Tensor:
    """Align current latent states with successful-future privileged states."""

    if teacher_states.shape != student_states.shape:
        raise ValueError("teacher and student states must have equal shapes")
    mse = (student_states.float() - teacher_states.float()).square().mean(dim=-1)
    cosine = (
        1 - F.cosine_similarity(
            student_states.float(), teacher_states.float(), dim=-1
        )
    ).clamp_min(0)
    per_token = mse + cosine_weight * cosine
    if token_mask is not None:
        if token_mask.shape != per_token.shape:
            raise ValueError("token_mask shape mismatch")
        return (per_token * token_mask).sum() / token_mask.sum().clamp_min(1)
    return per_token.mean()


def future_privileged_target(
    current_states: torch.Tensor,
    successful_future_states: torch.Tensor,
    *,
    horizon_decay: float = 0.9,
) -> torch.Tensor:
    """Build a causal backward target by discounted future pooling.

    Inputs are ``[batch, horizon, hidden]``.  The returned target is the
    discounted mean future state, suitable for a student that only sees the
    current observation. ``current_states`` is accepted for shape/device
    validation and to make the causal boundary explicit.
    """

    if current_states.ndim != 2 or successful_future_states.ndim != 3:
        raise ValueError("current_states=[batch, hidden], future_states=[batch, horizon, hidden]")
    if current_states.shape[0] != successful_future_states.shape[0] or current_states.shape[1] != successful_future_states.shape[2]:
        raise ValueError("current and future state dimensions do not match")
    if not 0 < horizon_decay <= 1:
        raise ValueError("horizon_decay must be in (0, 1]")
    horizon = successful_future_states.shape[1]
    weights = horizon_decay ** torch.arange(horizon, device=successful_future_states.device, dtype=successful_future_states.dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    return (successful_future_states * weights.view(1, -1, 1)).sum(dim=1)


# Backward-compatible descriptive alias for callers that prefer the adjective
# first; both names are part of the small public analysis API.
privileged_future_target = future_privileged_target


def backward_future_targets(
    states: torch.Tensor,
    *,
    horizon: int = 4,
    horizon_decay: float = 0.9,
) -> torch.Tensor:
    """Give every trajectory position its discounted privileged future target."""
    if states.ndim != 2 or horizon < 1 or not 0 < horizon_decay <= 1:
        raise ValueError("states=[time, hidden], positive horizon and valid decay required")
    targets = []
    for index in range(states.shape[0]):
        future = states[index:min(states.shape[0], index + horizon)]
        weights = horizon_decay ** torch.arange(
            future.shape[0], device=states.device, dtype=states.dtype
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        targets.append((future * weights[:, None]).sum(0))
    return torch.stack(targets)
