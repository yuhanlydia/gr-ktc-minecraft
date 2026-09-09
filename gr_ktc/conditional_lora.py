"""Latent-state-conditioned low-rank adapter basis.

This is a lightweight, deterministic baseline for the proposed continuous
KV-coordinate -> LoRA mapping.  It clusters latent coordinates only to obtain
initial basis regions, then uses soft distance weights at inference; no task
ID or semantic category router is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_closed_form import fit_low_rank_delta


@dataclass(frozen=True)
class ConditionalLoRABasis:
    basis_a: tuple[torch.Tensor, ...]
    basis_b: tuple[torch.Tensor, ...]
    centroids: torch.Tensor
    temperature: float = 1.0

    @property
    def num_bases(self) -> int:
        return len(self.basis_a)

    def coefficients(self, coordinate: torch.Tensor) -> torch.Tensor:
        if coordinate.ndim == 1:
            coordinate = coordinate.unsqueeze(0)
        if coordinate.ndim != 2 or coordinate.shape[-1] != self.centroids.shape[-1]:
            raise ValueError("coordinate has incompatible shape")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        distances = torch.cdist(coordinate.float(), self.centroids.float()).square()
        return torch.softmax(-distances / self.temperature, dim=-1)

    def delta_weight(self, coordinate: torch.Tensor) -> torch.Tensor:
        single = coordinate.ndim == 1
        weights = self.coefficients(coordinate)
        updates = torch.stack([b @ a for a, b in zip(self.basis_a, self.basis_b)])
        mixed = torch.einsum("nb,bod->nod", weights, updates)
        return mixed[0] if single else mixed


def _kmeans(coordinates: torch.Tensor, clusters: int, steps: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    n = coordinates.shape[0]
    clusters = min(clusters, n)
    # Farthest-point initialization is deterministic and avoids a dependency
    # on sklearn for the CPU analysis path.
    centers = [coordinates[0]]
    for _ in range(1, clusters):
        distances = torch.cdist(coordinates, torch.stack(centers)).min(dim=1).values
        centers.append(coordinates[distances.argmax()])
    centroids = torch.stack(centers).clone()
    for _ in range(steps):
        labels = torch.cdist(coordinates, centroids).argmin(dim=1)
        updated = []
        for index in range(clusters):
            members = coordinates[labels == index]
            updated.append(members.mean(0) if len(members) else centroids[index])
        new_centroids = torch.stack(updated)
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels


def fit_conditional_lora_basis(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bases: int = 4,
    rank: int = 8,
    ridge: float = 1e-3,
    temperature: float = 1.0,
) -> ConditionalLoRABasis:
    """Fit a latent-coordinate-conditioned collection of LoRA updates."""

    if coordinates.ndim != 2 or features.ndim != 2 or targets.ndim != 2:
        raise ValueError("coordinates, features and targets must be rank-2")
    if not (coordinates.shape[0] == features.shape[0] == targets.shape[0]):
        raise ValueError("all inputs must have the same number of contexts")
    if num_bases < 1 or rank < 1:
        raise ValueError("num_bases and rank must be positive")
    centroids, labels = _kmeans(coordinates.float(), num_bases)
    actual = centroids.shape[0]
    basis_a: list[torch.Tensor] = []
    basis_b: list[torch.Tensor] = []
    for index in range(actual):
        mask = labels == index
        x = features[mask] if mask.any() else features
        y = targets[mask] if mask.any() else targets
        a, b = fit_low_rank_delta(x, y, rank=rank, ridge=ridge)
        basis_a.append(a)
        basis_b.append(b)
    return ConditionalLoRABasis(tuple(basis_a), tuple(basis_b), centroids, temperature)


def fit_contextual_lora_basis(
    coordinates: torch.Tensor,
    contexts: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_bases: int = 4,
    rank: int = 8,
    ridge: float = 1e-3,
    temperature: float = 1.0,
) -> ConditionalLoRABasis:
    """Fit bases from variable-length token sequences grouped by latent state."""
    if coordinates.ndim != 2 or coordinates.shape[0] != len(contexts):
        raise ValueError("one coordinate is required per context")
    if not contexts:
        raise ValueError("contexts must not be empty")
    centroids, labels = _kmeans(coordinates.float(), num_bases)
    basis_a, basis_b = [], []
    for index in range(centroids.shape[0]):
        selected = [pair for pair, label in zip(contexts, labels, strict=True) if int(label) == index]
        if not selected:
            selected = contexts
        x = torch.cat([pair[0] for pair in selected])
        y = torch.cat([pair[1] for pair in selected])
        a, b = fit_low_rank_delta(x, y, rank=rank, ridge=ridge)
        basis_a.append(a)
        basis_b.append(b)
    return ConditionalLoRABasis(tuple(basis_a), tuple(basis_b), centroids, temperature)


def conditional_reachability_score(
    basis: ConditionalLoRABasis,
    coordinates: torch.Tensor,
    contexts: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Explained target energy under continuous per-context basis mixtures."""
    updates = basis.delta_weight(coordinates)
    error = target_energy = 0.0
    for update, (features, target) in zip(updates, contexts, strict=True):
        prediction = features @ update.T
        error += float((prediction.float() - target.float()).square().sum())
        target_energy += float(target.float().square().sum())
    return float(min(1.0, max(0.0, 1 - error / (target_energy + 1e-12))))
