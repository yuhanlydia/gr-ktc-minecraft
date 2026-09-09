from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import AcquisitionGroup


def normalize_task(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class LeakageFinding:
    group_id: str
    reason: str


def audit_evaluation_leakage(
    groups: Iterable[AcquisitionGroup],
    *,
    held_out_task_texts: Iterable[str],
    held_out_context_ids: Iterable[str] = (),
) -> list[LeakageFinding]:
    held_out_tasks = {normalize_task(text) for text in held_out_task_texts}
    held_out_contexts = set(held_out_context_ids)
    findings = []
    for group in groups:
        if normalize_task(group.context.task_text) in held_out_tasks:
            findings.append(LeakageFinding(group.group_id, "exact normalized task match"))
        if group.context.context_id in held_out_contexts:
            findings.append(LeakageFinding(group.group_id, "exact context match"))
    return findings

