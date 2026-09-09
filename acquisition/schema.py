from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


@dataclass(frozen=True)
class ContextKey:
    task_text: str
    scene_id: str
    seed: int
    inventory: tuple[tuple[str, int], ...] = ()
    biome: str | None = None
    time_bucket: Literal["day", "dusk", "night", "unknown"] = "unknown"
    location_grid: tuple[int, int, int] | None = None

    @property
    def context_id(self) -> str:
        return stable_id("ctx", asdict(self))


@dataclass(frozen=True)
class AcquisitionGroup:
    group_id: str
    context: ContextKey
    setup_commands: tuple[str, ...]
    milestones: tuple[dict[str, Any], ...]
    rollout_count: int = 4
    source: str = "mineexplorer"

    @classmethod
    def create(
        cls,
        context: ContextKey,
        setup_commands: tuple[str, ...],
        milestones: tuple[dict[str, Any], ...],
        rollout_count: int = 4,
        source: str = "mineexplorer",
    ) -> "AcquisitionGroup":
        if rollout_count < 2:
            raise ValueError("group-relative acquisition requires at least two rollouts")
        payload = {
            "context": asdict(context),
            "commands": setup_commands,
            "milestones": milestones,
            "rollout_count": rollout_count,
            "source": source,
        }
        return cls(stable_id("grp", payload), context, setup_commands, milestones, rollout_count, source)


@dataclass(frozen=True)
class RolloutMetadata:
    trajectory_id: str
    group_id: str
    rollout_index: int
    seed: int
    verifier_score: float
    parser_valid: bool
    task_success: bool
    generated_tokens: int
    latency_seconds: float
    model_id: str
    selected_layers: tuple[int, ...]
    terminal_state: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.rollout_index < 0 or self.generated_tokens < 0:
            raise ValueError("negative rollout index or token count")
        if not 0.0 <= self.verifier_score <= 1.0:
            raise ValueError("verifier score must be within [0, 1]")
        if self.latency_seconds < 0:
            raise ValueError("latency must be non-negative")
        if len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError("selected layers must be unique")

