"""Population memory for many-loop latent trajectory evolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PopulationItem:
    trajectory: torch.Tensor
    score: float
    item_id: str

    @property
    def embedding(self) -> torch.Tensor:
        return self.trajectory.float().mean(dim=0)


class LatentPopulation:
    def __init__(self, items: list[PopulationItem] | None = None, *, max_size: int = 32):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.items: list[PopulationItem] = list(items or [])[:max_size]

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: PopulationItem) -> None:
        if item.trajectory.ndim != 2 or item.trajectory.shape[0] == 0:
            raise ValueError("trajectory must be [time, hidden] and non-empty")
        self.items.append(item)
        self.items.sort(key=lambda x: x.score, reverse=True)
        del self.items[self.max_size:]

    def retrieval_weights(self, context: torch.Tensor, *, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        if not self.items:
            raise ValueError("cannot retrieve from an empty population")
        if context.ndim != 1 or temperature <= 0:
            raise ValueError("context must be a vector and temperature positive")
        candidates = self.items
        embeddings = torch.stack([item.embedding for item in candidates])
        distances = (embeddings - context.float()).square().sum(dim=-1)
        if top_k is not None:
            top_k = max(1, min(top_k, len(candidates)))
            keep = distances.topk(top_k, largest=False).indices
            logits = -distances[keep] / temperature
            weights = torch.softmax(logits, dim=0)
            result = torch.zeros(len(candidates), dtype=weights.dtype)
            result[keep] = weights
            return result
        return torch.softmax(-distances / temperature, dim=0)

    def retrieve(self, context: torch.Tensor, *, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        weights = self.retrieval_weights(context, temperature=temperature, top_k=top_k)
        # Phase-resample trajectories before mixture, retaining a single API
        # for variable-length population members.
        length = max(item.trajectory.shape[0] for item in self.items)
        support = []
        for item in self.items:
            index = torch.linspace(0, item.trajectory.shape[0] - 1, length, device=item.trajectory.device)
            left = index.floor().long().clamp_max(item.trajectory.shape[0] - 1)
            right = index.ceil().long().clamp_max(item.trajectory.shape[0] - 1)
            fraction = (index - left).unsqueeze(-1)
            support.append(item.trajectory[left] * (1 - fraction) + item.trajectory[right] * fraction)
        return torch.einsum("n,ntd->td", weights.to(support[0].device), torch.stack(support))

    def evolve(
        self,
        candidates: list[PopulationItem],
        *,
        elite_fraction: float = 0.5,
        mutation_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Selection + optional mutation update, preserving a score-ranked archive."""
        if not 0 < elite_fraction <= 1 or mutation_std < 0:
            raise ValueError("invalid elite_fraction or mutation_std")
        merged = sorted(self.items + candidates, key=lambda x: x.score, reverse=True)
        elite_count = max(1, int(round(len(merged) * elite_fraction)))
        generator = torch.Generator().manual_seed(seed)
        next_items: list[PopulationItem] = []
        for index, item in enumerate(merged[:elite_count]):
            trajectory = item.trajectory
            if mutation_std:
                trajectory = trajectory + mutation_std * torch.randn(trajectory.shape, generator=generator, device=trajectory.device, dtype=trajectory.dtype)
            next_items.append(PopulationItem(trajectory, item.score, item.item_id))
        # Fill remaining capacity with high-scoring non-elites to avoid a
        # single-lineage collapse when scores are close.
        for item in merged[elite_count:]:
            if len(next_items) >= self.max_size:
                break
            if item.item_id not in {x.item_id for x in next_items}:
                next_items.append(item)
        self.items = next_items[:self.max_size]

