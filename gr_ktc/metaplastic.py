"""Fixed-budget structural plasticity for PEFT LoRA adapters.

The projection is deliberately task agnostic: every attached LoRA module is a
candidate and the globally largest singular directions share one rank budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PlasticFactors:
    a: torch.Tensor  # [capacity, d_in]
    b: torch.Tensor  # [d_out, capacity]
    scale: float = 1.0


@dataclass(frozen=True)
class ProjectionReport:
    budget: int
    active_rank: int
    ranks: dict[str, int]
    singular_values: dict[str, list[float]]
    retained_energy: float


def group_relative_policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ignore_index: int = -100,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Signed GRPO-style sequence loss without a learned reward model."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must be [batch,time,vocab]/[batch,time]")
    if advantages.shape != (logits.shape[0],):
        raise ValueError("advantages must have one value per rollout")
    token_nll = F.cross_entropy(
        logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none",
        ignore_index=ignore_index,
    )
    mask = labels[:, 1:] != ignore_index
    sequence_nll = (token_nll * mask).sum(1)
    if length_normalize:
        sequence_nll = sequence_nll / mask.sum(1).clamp_min(1)
    # Detach because MineExplorer scores and their normalization are outcomes,
    # not differentiable reward-model predictions.
    return (sequence_nll * advantages.detach().to(sequence_nll)).mean()


def _compact_svd(factors: PlasticFactors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD(scale * B@A) through a capacity-sized core."""
    a = factors.a.float()
    b = factors.b.float()
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
    u_c, s, vh_c = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    return q_b @ u_c, s * abs(float(factors.scale)), vh_c @ q_a.T


@torch.no_grad()
def project_global_rank_budget(
    modules: Mapping[str, PlasticFactors], budget: int
) -> tuple[dict[str, PlasticFactors], ProjectionReport]:
    """Keep the globally top-``budget`` singular directions across modules."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    decompositions = {}
    candidates: list[tuple[float, str, int]] = []
    total_energy = 0.0
    for name, factors in modules.items():
        if factors.a.ndim != 2 or factors.b.ndim != 2:
            raise ValueError(f"{name}: factors must be matrices")
        if factors.a.shape[0] != factors.b.shape[1]:
            raise ValueError(f"{name}: incompatible factor capacity")
        u, s, vh = _compact_svd(factors)
        decompositions[name] = (u, s, vh)
        for index, value in enumerate(s.tolist()):
            candidates.append((value, name, index))
            total_energy += value * value
    selected = sorted(candidates, reverse=True)[:budget]
    selected_by_name: dict[str, list[int]] = {name: [] for name in modules}
    retained_energy = 0.0
    for value, name, index in selected:
        if value > 0:
            selected_by_name[name].append(index)
            retained_energy += value * value

    projected: dict[str, PlasticFactors] = {}
    spectra: dict[str, list[float]] = {}
    for name, old in modules.items():
        u, s, vh = decompositions[name]
        indices = selected_by_name[name]
        capacity = old.a.shape[0]
        new_a = torch.zeros_like(old.a)
        new_b = torch.zeros_like(old.b)
        for slot, index in enumerate(indices[:capacity]):
            root = s[index].clamp_min(0).sqrt()
            # Store the scale in factors themselves and normalize PEFT scale to 1.
            new_a[slot].copy_((root * vh[index]).to(new_a))
            new_b[:, slot].copy_((root * u[:, index]).to(new_b))
        # A zero B row has no functional effect, while retaining its A probe
        # lets a future gradient create a new direction. Zeroing both factors
        # would make pruning irreversible and silently disable ``grow``.
        for slot in range(len(indices), capacity):
            new_a[slot].copy_(old.a[slot])
        projected[name] = PlasticFactors(new_a, new_b, 1.0)
        spectra[name] = [float(s[i]) for i in indices]
    ranks = {name: len(indices) for name, indices in selected_by_name.items()}
    report = ProjectionReport(
        budget=budget,
        active_rank=sum(ranks.values()),
        ranks=ranks,
        singular_values=spectra,
        retained_energy=retained_energy / max(total_energy, 1e-12),
    )
    return projected, report


def peft_plastic_factors(model, adapter_name: str = "default") -> dict[str, PlasticFactors]:
    """Extract every PEFT LoRA layer; no transformer layer is hand selected."""
    result = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or adapter_name not in module.lora_A:
            continue
        scale = float(module.scaling[adapter_name])
        result[name] = PlasticFactors(
            module.lora_A[adapter_name].weight.detach().clone(),
            module.lora_B[adapter_name].weight.detach().clone(), scale,
        )
    if not result:
        raise ValueError("model contains no PEFT LoRA layers")
    return result


@torch.no_grad()
def load_peft_plastic_factors(model, factors: Mapping[str, PlasticFactors], adapter_name="default"):
    modules = dict(model.named_modules())
    for name, value in factors.items():
        module = modules[name]
        module.lora_A[adapter_name].weight.copy_(value.a)
        module.lora_B[adapter_name].weight.copy_(value.b)
        # PEFT stores scale separately. The projected factors already include it.
        module.scaling[adapter_name] = float(value.scale)


def project_peft_global_rank_budget(model, budget: int, adapter_name="default") -> ProjectionReport:
    projected, report = project_global_rank_budget(
        peft_plastic_factors(model, adapter_name), budget
    )
    load_peft_plastic_factors(model, projected, adapter_name)
    return report


@torch.no_grad()
def decay_peft_plastic_weights(model, retention: float, adapter_name="default") -> None:
    """Apply gamma * Delta-W before the next experience gradient."""
    if not 0 <= retention <= 1:
        raise ValueError("retention must be in [0, 1]")
    for module in model.modules():
        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
            module.lora_B[adapter_name].weight.mul_(retention)
