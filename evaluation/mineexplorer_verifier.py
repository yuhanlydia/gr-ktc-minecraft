from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class WorldObject:
    kind: str
    name: str
    position: Vec3


@dataclass(frozen=True)
class VerifierState:
    inventory: Mapping[str, int]
    player_position: Vec3
    player_facing: Vec3
    spawn_position: Vec3
    objects: tuple[WorldObject, ...]


@dataclass(frozen=True)
class VerificationResult:
    milestone_success_rate: float
    task_success: bool
    milestone_results: dict[str, bool]
    rule_results: dict[str, tuple[bool, ...]]


def _world_point(point: Iterable[float], frame: str, spawn: Vec3) -> Vec3:
    values = tuple(float(value) for value in point)
    if len(values) != 3:
        raise ValueError("coordinates must contain x, y, z")
    if frame == "spawn_relative":
        return tuple(a + b for a, b in zip(values, spawn, strict=True))  # type: ignore[return-value]
    if frame in {"world", "absolute"}:
        return values  # type: ignore[return-value]
    raise ValueError(f"unsupported coordinate frame: {frame}")


def _inside(position: Vec3, minimum: Vec3, maximum: Vec3) -> bool:
    return all(low <= value <= high for value, low, high in zip(position, minimum, maximum, strict=True))


def _evaluate_rule(rule: Mapping[str, Any], state: VerifierState) -> bool:
    kind = rule["type"]
    params = rule["params"]
    if kind == "inventory_has":
        return int(state.inventory.get(params["item"], 0)) >= int(params["min_count"])

    frame = params.get("coordinate_frame", "world")
    if kind == "position_inside_box":
        minimum = _world_point(params["min"], frame, state.spawn_position)
        maximum = _world_point(params["max"], frame, state.spawn_position)
        return _inside(state.player_position, minimum, maximum)

    if kind in {"count_in_box_at_least", "count_in_box_at_most"}:
        minimum = _world_point(params["min"], frame, state.spawn_position)
        maximum = _world_point(params["max"], frame, state.spawn_position)
        count = sum(
            obj.kind == params["kind"]
            and obj.name.removeprefix("minecraft:") == params["object"].removeprefix("minecraft:")
            and _inside(obj.position, minimum, maximum)
            for obj in state.objects
        )
        if kind == "count_in_box_at_least":
            return count >= int(params["min_count"])
        return count <= int(params["max_count"])

    if kind == "position_near_with_facing":
        target = _world_point(params["target"], frame, state.spawn_position)
        delta = tuple(t - p for t, p in zip(target, state.player_position, strict=True))
        distance = math.sqrt(sum(component * component for component in delta))
        if distance > float(params["max_distance"]):
            return False
        facing_norm = math.sqrt(sum(component * component for component in state.player_facing))
        if distance < 1e-9:
            return True
        if facing_norm < 1e-9:
            return False
        cosine = sum(a * b for a, b in zip(delta, state.player_facing, strict=True)) / (
            distance * facing_norm
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        return angle <= float(params["facing_tolerance"])

    raise ValueError(f"unsupported MineExplorer rule type: {kind}")


def verify_milestones(
    milestones: Iterable[Mapping[str, Any]],
    state: VerifierState,
) -> VerificationResult:
    milestone_results: dict[str, bool] = {}
    rule_results: dict[str, tuple[bool, ...]] = {}
    for milestone in milestones:
        milestone_id = str(milestone["milestone_id"])
        outcomes = tuple(_evaluate_rule(rule, state) for rule in milestone["rules"])
        rule_results[milestone_id] = outcomes
        milestone_results[milestone_id] = bool(outcomes) and all(outcomes)
    count = len(milestone_results)
    completed = sum(milestone_results.values())
    msr = completed / count if count else 0.0
    return VerificationResult(msr, count > 0 and completed == count, milestone_results, rule_results)

