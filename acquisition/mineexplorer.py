from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schema import AcquisitionGroup, ContextKey


@dataclass(frozen=True)
class MineExplorerScenario:
    scene_id: str
    mode: str
    task_text: str
    scene_name: str
    scene_description: str
    commands: tuple[str, ...]
    selected_tasks: tuple[str, ...]
    milestones: tuple[dict[str, Any], ...]
    reasoning_graph: dict[str, Any] | None

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "MineExplorerScenario":
        required = {
            "scene_id", "mode", "task_text", "scene_name", "scene_description",
            "commands", "selected_tasks", "milestones",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"MineExplorer record missing fields: {sorted(missing)}")
        return cls(
            scene_id=str(record["scene_id"]),
            mode=str(record["mode"]),
            task_text=str(record["task_text"]),
            scene_name=str(record["scene_name"]),
            scene_description=str(record["scene_description"]),
            commands=tuple(map(str, record["commands"])),
            selected_tasks=tuple(map(str, record["selected_tasks"])),
            milestones=tuple(record["milestones"]),
            reasoning_graph=record.get("reasoning_graph"),
        )

    def acquisition_group(self, seed: int, rollout_count: int = 4) -> AcquisitionGroup:
        inventory = tuple(sorted(_inventory_from_commands(self.commands).items()))
        context = ContextKey(
            task_text=self.task_text,
            scene_id=self.scene_id,
            seed=seed,
            inventory=inventory,
        )
        return AcquisitionGroup.create(
            context=context,
            setup_commands=self.commands,
            milestones=self.milestones,
            rollout_count=rollout_count,
        )


_GIVE_PATTERN = re.compile(r"^/give\s+@\w+\s+([\w:.-]+)(?:\s+(\d+))?$")


def _inventory_from_commands(commands: tuple[str, ...]) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for command in commands:
        match = _GIVE_PATTERN.match(command.strip())
        if match:
            item = match.group(1).removeprefix("minecraft:")
            inventory[item] = inventory.get(item, 0) + int(match.group(2) or 1)
    return inventory


def load_mineexplorer(path: str | Path) -> Iterator[MineExplorerScenario]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield MineExplorerScenario.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid MineExplorer record at line {line_number}") from exc

