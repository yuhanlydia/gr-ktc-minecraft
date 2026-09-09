from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-example weighted next-token cross entropy."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must be [batch, time, vocab] / [batch, time]")
    if sample_weights.shape != (logits.shape[0],):
        raise ValueError("sample_weights must have one value per batch item")
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels,
        reduction="none", ignore_index=ignore_index,
    )
    mask = shifted_labels != ignore_index
    per_sample = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    weights = sample_weights.float().clamp_min(0)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    if not (
        chosen_logps.shape == rejected_logps.shape
        == reference_chosen_logps.shape == reference_rejected_logps.shape
    ):
        raise ValueError("all DPO log-prob tensors must have equal shapes")
    student_margin = chosen_logps - rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (student_margin - reference_margin)).mean()


def grta_loss(
    student_effect: torch.Tensor,
    teacher_effect: torch.Tensor,
    positive_basis: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine alignment of student and memory-teacher effects in U+ space."""
    if student_effect.shape != teacher_effect.shape:
        raise ValueError("student and teacher effects must have equal shapes")
    projected_student = student_effect @ positive_basis
    projected_teacher = teacher_effect @ positive_basis
    losses = 1 - F.cosine_similarity(
        projected_student.float(), projected_teacher.float(), dim=-1, eps=1e-6
    )
    if token_mask is None:
        return losses.mean()
    if token_mask.shape != losses.shape:
        raise ValueError("token mask shape does not match effects")
    return (losses * token_mask).sum() / token_mask.sum().clamp_min(1)


def negative_subspace_penalty(
    student_effect: torch.Tensor,
    negative_basis: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    projected_energy = (
        student_effect @ negative_basis
    ).float().square().sum(dim=-1)
    total_energy = student_effect.float().square().sum(dim=-1)
    # A dimensionless fraction is comparable to cosine GRTA across layers and
    # quantization modes. The previous raw energy could exceed 400 and dominate
    # every other objective despite lambda_-=0.1.
    energy = projected_energy / total_energy.clamp_min(1e-12)
    if token_mask is None:
        return energy.mean()
    return (energy * token_mask).sum() / token_mask.sum().clamp_min(1)


def anchor_kl(
    base_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if base_logits.shape != student_logits.shape:
        raise ValueError("anchor logits must have equal shapes")
    base_log_probs = F.log_softmax(base_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    per_token = F.kl_div(
        student_log_probs, base_log_probs, log_target=True, reduction="none"
    ).sum(dim=-1)
    if token_mask is None:
        return per_token.mean()
    return (per_token * token_mask).sum() / token_mask.sum().clamp_min(1)


@dataclass(frozen=True)
class SlowLossWeights:
    dpo: float = 1.0
    trajectory: float = 0.5
    negative: float = 0.1
    anchor: float = 0.01


def combine_slow_losses(
    *,
    bc: torch.Tensor,
    dpo: torch.Tensor,
    trajectory: torch.Tensor,
    negative: torch.Tensor,
    anchor: torch.Tensor,
    weights: SlowLossWeights = SlowLossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    total = (
        bc + weights.dpo * dpo + weights.trajectory * trajectory
        + weights.negative * negative + weights.anchor * anchor
    )
    metrics = {
        "loss/total": float(total.detach()),
        "loss/bc": float(bc.detach()),
        "loss/dpo": float(dpo.detach()),
        "loss/trajectory": float(trajectory.detach()),
        "loss/negative": float(negative.detach()),
        "loss/anchor": float(anchor.detach()),
    }
    return total, metrics
