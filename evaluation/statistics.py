from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z / denominator * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> tuple[int, int, float]:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("paired outcomes must be non-empty and equal length")
    left_only = sum(bool(a) and not bool(b) for a, b in zip(left, right, strict=True))
    right_only = sum(not bool(a) and bool(b) for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2 ** discordant)
    return left_only, right_only, min(1.0, 2 * probability)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def paired_bootstrap_median_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("paired samples must be non-empty and equal length")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(resamples, len(differences)))
    estimates = np.median(differences[indices], axis=1)
    return (
        float(np.median(differences)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )

