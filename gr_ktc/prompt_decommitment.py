"""Prompt-only event decommitment for closed-loop MineExplorer agents.

The method is intentionally training-free and does not alter attention, KV
caches, or model weights.  It maintains a discrete ledger of the agent's own
past decisions and lets the model explicitly revoke previously committed
thought/action events when current visual evidence makes them obsolete.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Iterable, Mapping, Sequence


class PromptMode(str, Enum):
    BASE = "base"
    REFLECTION = "reflection"
    DECOMMIT = "decommit"
    DECOMMIT_IGNORE = "decommit_ignore"


@dataclass(frozen=True)
class ControlResponse:
    revoke_event_ids: tuple[str, ...]
    current_belief: str


@dataclass(frozen=True)
class EventBlock:
    event_id: str
    step: int
    thought: str
    action: dict


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _find_json_object(text: str) -> dict:
    for candidate in reversed(_JSON_FENCE.findall(text)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("response contains no parseable JSON object")


def parse_control_response(text: str) -> ControlResponse:
    payload = _find_json_object(text)
    raw_ids = payload.get("revoke_event_ids", [])
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, list):
        raise ValueError("revoke_event_ids must be a JSON list")
    ids = tuple(str(value).strip() for value in raw_ids if str(value).strip())
    belief = str(payload.get("current_belief", "")).strip()
    return ControlResponse(ids, belief)


def _normalized_number(value: float) -> float | int:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return int(round(numeric))
    return round(numeric, 4)


def action_signature(action: Mapping[str, object]) -> str:
    """Canonicalize a low-level MineExplorer action for repetition metrics."""
    kept: dict[str, object] = {}
    for key, value in action.items():
        if isinstance(value, bool):
            if value:
                kept[key] = 1
            continue
        if isinstance(value, (int, float)):
            if abs(float(value)) > 1e-12:
                kept[key] = _normalized_number(float(value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized = [
                _normalized_number(float(item)) if isinstance(item, (int, float)) else item
                for item in value
            ]
            if any(item not in (0, 0.0, None, False) for item in normalized):
                kept[key] = normalized
            continue
        if value not in (None, "", [], {}):
            kept[key] = value
    return json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class DecommitmentState:
    def __init__(self, *, ignore_revocations: bool = False) -> None:
        self.ignore_revocations = bool(ignore_revocations)
        self._events: list[EventBlock] = []
        self._revoked: set[str] = set()

    @property
    def revoked_event_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._revoked, key=lambda value: int(value[1:])))

    @property
    def events(self) -> tuple[EventBlock, ...]:
        return tuple(self._events)

    def observe_decision(self, *, step: int, thought: str, action: Mapping[str, object]) -> str:
        if step < 1:
            raise ValueError("step must be positive")
        event_id = f"E{step}"
        self._events.append(EventBlock(event_id, step, str(thought), dict(action)))
        return event_id

    def apply_revocations(self, event_ids: Iterable[str], *, current_step: int) -> tuple[str, ...]:
        if self.ignore_revocations:
            return ()
        existing = {event.event_id: event for event in self._events}
        applied: list[str] = []
        for raw in event_ids:
            event_id = str(raw).strip()
            event = existing.get(event_id)
            if event is None:
                continue
            if event.step >= current_step:
                continue
            if event_id not in self._revoked:
                self._revoked.add(event_id)
                applied.append(event_id)
        return tuple(applied)

    def render_ledger(self, *, max_events: int = 12) -> str:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        selected = self._events[-max_events:]
        if not selected:
            return "No prior decision events."
        lines = []
        for event in selected:
            status = "REVOKED" if event.event_id in self._revoked else "ACTIVE"
            lines.append(
                f"{event.event_id} [{status}] thought={event.thought!r} "
                f"action={action_signature(event.action)}"
            )
        return "\n".join(lines)


def build_control_instruction(mode: PromptMode | str) -> str:
    mode = PromptMode(mode)
    if mode is PromptMode.BASE:
        return ""
    if mode is PromptMode.REFLECTION:
        return (
            "Before choosing the next action, reconsider the previous plan against the "
            "latest visual evidence. If a prior assumption or action is no longer valid, "
            "revise your plan rather than repeating it blindly. Do not invent hidden "
            "environment facts."
        )
    return (
        "You maintain a discrete ledger of your own prior decision events E1, E2, ... . "
        "Before acting, explicitly identify any PRIOR event whose thought/plan has become "
        "contradicted, completed, or obsolete according to the visible observation history. "
        "Return `revoke_event_ids` as a JSON list of those event IDs. Revocation applies only "
        "to the old thought/action commitment; the historical image remains evidence. Never "
        "revoke an event merely because progress is slow, and never use hidden verifier or "
        "benchmark information. Then write `current_belief`, a concise description of the "
        "current world state using current visual evidence plus non-revoked commitments, and "
        "choose the next action from that belief. Your JSON must contain exactly these five "
        "top-level keys: `revoke_event_ids`, `current_belief`, `thought`, `action`, "
        "`memory_update`."
    )
