from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch

from .group_advantage import group_relative_advantage

Trajectory = TypeVar("Trajectory")
Memory = TypeVar("Memory")


@dataclass(frozen=True)
class VerifiedRollout(Generic[Trajectory]):
    trajectory: Trajectory
    score: float
    parser_valid: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class RetryCycle(Generic[Trajectory, Memory]):
    index: int
    rollouts: list[VerifiedRollout[Trajectory]]
    advantages: torch.Tensor
    memory: Memory | None
    best_score: float
    improved: bool


@dataclass
class RetryResult(Generic[Trajectory, Memory]):
    cycles: list[RetryCycle[Trajectory, Memory]]
    best_rollout: VerifiedRollout[Trajectory] | None
    stopped_reason: str


def run_verified_retry_loop(
    rollout_fn: Callable[[Memory | None, int, int], VerifiedRollout[Trajectory]],
    merge_fn: Callable[[list[Trajectory], torch.Tensor], Memory],
    *,
    rollouts_per_cycle: int = 4,
    max_cycles: int = 4,
    success_score: float = 1.0,
    patience: int = 2,
) -> RetryResult[Trajectory, Memory]:
    """Run bounded verifier-closed-loop fast-memory refinement.

    Memory is updated only for mixed-outcome groups with a positive relative
    signal. Invalid parser outputs remain in metrics but cannot become teachers.
    """
    if rollouts_per_cycle < 2 or max_cycles < 1 or patience < 1:
        raise ValueError("invalid retry-loop budget")
    cycles: list[RetryCycle[Trajectory, Memory]] = []
    memory: Memory | None = None
    best: VerifiedRollout[Trajectory] | None = None
    stale_cycles = 0

    for cycle_index in range(max_cycles):
        rollouts = [
            rollout_fn(memory, cycle_index, rollout_index)
            for rollout_index in range(rollouts_per_cycle)
        ]
        valid_scores = torch.tensor(
            [item.score if item.parser_valid else 0.0 for item in rollouts],
            dtype=torch.float32,
        )
        advantages = group_relative_advantage(valid_scores)
        cycle_best = max(rollouts, key=lambda item: item.score if item.parser_valid else -1)
        improved = best is None or (
            cycle_best.parser_valid and cycle_best.score > best.score
        )
        if improved:
            best = cycle_best
            stale_cycles = 0
        else:
            stale_cycles += 1

        positive = advantages > 0
        if positive.any() and (advantages < 0).any():
            valid_trajectories = [item.trajectory for item in rollouts]
            memory = merge_fn(valid_trajectories, advantages)

        cycles.append(
            RetryCycle(
                index=cycle_index,
                rollouts=rollouts,
                advantages=advantages,
                memory=memory,
                best_score=float(cycle_best.score),
                improved=improved,
            )
        )
        if best is not None and best.parser_valid and best.score >= success_score:
            return RetryResult(cycles, best, "verified_success")
        if stale_cycles >= patience:
            return RetryResult(cycles, best, "no_improvement")
    return RetryResult(cycles, best, "budget_exhausted")

