from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WhiteningTransform:
    mean: torch.Tensor
    matrix: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) @ self.matrix.to(values)


def fit_whitening(
    values: torch.Tensor,
    eps: float = 1e-5,
    max_rank: int | None = None,
    *,
    randomized: bool = False,
    random_seed: int = 0,
    randomized_iterations: int = 4,
) -> WhiteningTransform:
    """Fit PCA whitening without constructing a dense feature covariance."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must have shape [samples >= 2, features]")
    x = values.float() if randomized else values.double()
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    maximum = min(centered.shape)
    rank = maximum if max_rank is None else min(max_rank, maximum)
    if randomized:
        with torch.random.fork_rng():
            torch.manual_seed(random_seed)
            _, singular_values, vectors = torch.pca_lowrank(
                centered,
                q=rank,
                center=False,
                niter=randomized_iterations,
            )
        components = vectors
    else:
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[:rank].T
    scales = ((singular_values[:rank] ** 2) / max(x.shape[0] - 1, 1) + eps).rsqrt()
    matrix = components[:, :rank] * scales.unsqueeze(0)
    return WhiteningTransform(mean.squeeze(0).float(), matrix.float())
