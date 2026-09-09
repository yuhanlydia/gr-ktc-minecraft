from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PEAMTask:
    task_id: str
    category: str
    instruction: str
    verifier_kind: str
    target: str
    count: int = 1
    setup_commands: tuple[str, ...] = ()


PEAM_TASKS: tuple[PEAMTask, ...] = (
    PEAMTask("T1", "craft", "Craft a crafting table", "inventory", "crafting_table"),
    PEAMTask("T2", "craft", "Craft a wooden pickaxe", "inventory", "wooden_pickaxe"),
    PEAMTask("T3", "craft", "Craft a stone pickaxe", "inventory", "stone_pickaxe"),
    PEAMTask("T4", "craft", "Craft a furnace", "inventory", "furnace"),
    PEAMTask("T5", "craft", "Craft an iron pickaxe", "inventory", "iron_pickaxe"),
    PEAMTask("T6", "gather", "Collect 4 oak logs", "inventory", "oak_log", 4),
    PEAMTask("T7", "gather", "Mine 8 cobblestone", "inventory", "cobblestone", 8),
    PEAMTask("T8", "gather", "Mine 2 iron ore, including required processing", "inventory", "iron_ingot", 2),
    PEAMTask("T9", "gather", "Collect 4 coal", "inventory", "coal", 4),
    PEAMTask(
        "T10", "combat", "Defeat a zombie at night", "kill", "zombie", 1,
        ("/time set night",),
    ),
    PEAMTask(
        "T11", "combat", "Defeat a skeleton with bow", "kill", "skeleton", 1,
        ("/give @s minecraft:bow 1", "/give @s minecraft:arrow 16"),
    ),
)

PEAM_SEEDS: tuple[int, ...] = (42, 43, 44)
PEAM_MAX_AGENT_STEPS = 200


def paired_trials() -> tuple[tuple[PEAMTask, int], ...]:
    return tuple((task, seed) for task in PEAM_TASKS for seed in PEAM_SEEDS)


def verify_inventory(task: PEAMTask, inventory: dict[str, int]) -> bool:
    if task.verifier_kind != "inventory":
        raise ValueError(f"{task.task_id} requires a {task.verifier_kind} verifier")
    return int(inventory.get(task.target, 0)) >= task.count


def verify_events(task: PEAMTask, events: list[list[object]]) -> bool:
    """Environment-side verifier over authoritative Voyager observations."""
    if task.verifier_kind == "inventory":
        observations = [value for kind, value in events if kind == "observe"]
        if not observations:
            return False
        return verify_inventory(task, observations[-1].get("inventory", {}))  # type: ignore[union-attr]
    if task.verifier_kind == "kill":
        expected = f"{task.target}_killed"
        return any(
            kind == "onSave"
            and isinstance(value, dict)
            and value.get("onSave") == expected
            for kind, value in events
        )
    raise ValueError(f"unsupported verifier kind: {task.verifier_kind}")
