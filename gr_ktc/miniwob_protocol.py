"""Pure protocol rules for the MiniWoB++ LatentSkill reliability gate."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import statistics
from typing import Sequence


SUCCESS_THRESHOLD = 0.5
MIN_SUCCESSES = 3
MIN_FAILURES = 3
MIN_REWARD_STD = 0.15
MODES = ("base", "context", "positive_all", "clsc")
ALLOWED_ACTIONS = frozenset(
    {
        "clear",
        "click",
        "fill",
        "focus",
        "noop",
        "press",
        "scroll",
        "select_option",
    }
)


@dataclass(frozen=True)
class MiniwobPhase:
    name: str
    rollouts_per_instance: int
    test_instances: int
    target_families: int
    modes: tuple[str, ...]


@dataclass(frozen=True)
class SignalQualification:
    qualified: bool
    successes: int
    failures: int
    reward_std: float
    reason: str


def phase_spec(name: str) -> MiniwobPhase:
    specs = {
        "smoke": MiniwobPhase("smoke", 2, 1, 1, MODES),
        "quick": MiniwobPhase("quick", 10, 8, 2, MODES),
    }
    try:
        return specs[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown phase {name!r}; choose smoke or quick") from exc


def _finite_raw_reward(raw_reward: object) -> float:
    try:
        value = float(raw_reward)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("MiniWoB requires a finite raw reward") from exc
    if not math.isfinite(value):
        raise ValueError("MiniWoB requires a finite raw reward")
    return value


def quality_reward(raw_reward: object) -> float:
    """Return the preregistered continuous quality value in ``[0, 1]``."""
    return min(1.0, max(0.0, _finite_raw_reward(raw_reward)))


def is_terminal_success(raw_reward: object) -> bool:
    """Avoid BrowserGym's unsafe ``raw_reward > 0`` success conversion."""
    return _finite_raw_reward(raw_reward) >= SUCCESS_THRESHOLD


def qualify_signal(rewards: Sequence[object]) -> SignalQualification:
    values = [quality_reward(raw) for raw in rewards]
    successes = sum(is_terminal_success(raw) for raw in rewards)
    failures = len(values) - successes
    reward_std = statistics.pstdev(values) if len(values) > 1 else 0.0
    reasons: list[str] = []
    if successes < MIN_SUCCESSES:
        reasons.append(f"requires at least {MIN_SUCCESSES} terminal successes")
    if failures < MIN_FAILURES:
        reasons.append(f"requires at least {MIN_FAILURES} terminal failures")
    if reward_std < MIN_REWARD_STD:
        reasons.append(f"reward std must be at least {MIN_REWARD_STD}")
    return SignalQualification(
        qualified=not reasons,
        successes=successes,
        failures=failures,
        reward_std=float(reward_std),
        reason="; ".join(reasons) if reasons else "qualified",
    )


def protocol_seed(namespace: str, family: str, index: int, base_seed: int) -> int:
    payload = f"miniwob-clsc-v1|{namespace}|{family}|{index}|{base_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _literal_only(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool, type(None)))
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_literal_only(item) for item in node.elts)
    return False


def _validated_action(candidate: str) -> str | None:
    try:
        parsed = ast.parse(candidate.strip(), mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    if call.func.id not in ALLOWED_ACTIONS:
        return None
    if not all(_literal_only(arg) for arg in call.args):
        return None
    if not all(keyword.arg and _literal_only(keyword.value) for keyword in call.keywords):
        return None
    return candidate.strip()


def extract_single_action(text: str) -> str:
    """Extract one literal-only high-level action from model output."""
    raw = str(text).replace("```python", "").replace("```", "").strip()
    candidates: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            candidates.append(stripped.split(":", 1)[1].strip())
    if not candidates:
        candidates.append(raw)
    valid = [action for item in candidates if (action := _validated_action(item))]
    if len(valid) != 1:
        raise ValueError("expected a single allowed BrowserGym action")
    return valid[0]
