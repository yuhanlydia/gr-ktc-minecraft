from evaluation.peam_suite import (
    PEAM_MAX_AGENT_STEPS,
    PEAM_SEEDS,
    PEAM_TASKS,
    paired_trials,
    verify_events,
    verify_inventory,
)


def test_peam_protocol_has_11_tasks_and_33_paired_trials():
    assert [task.task_id for task in PEAM_TASKS] == [f"T{i}" for i in range(1, 12)]
    assert PEAM_SEEDS == (42, 43, 44)
    assert len(paired_trials()) == 33
    assert PEAM_MAX_AGENT_STEPS == 200
    assert {task.category for task in PEAM_TASKS} == {"craft", "gather", "combat"}


def test_inventory_verifier_is_environment_side_and_counted():
    oak_logs = next(task for task in PEAM_TASKS if task.task_id == "T6")
    assert not verify_inventory(oak_logs, {"oak_log": 3})
    assert verify_inventory(oak_logs, {"oak_log": 4})


def test_combat_verifier_requires_authoritative_kill_save_event():
    zombie = next(task for task in PEAM_TASKS if task.task_id == "T10")
    assert not verify_events(zombie, [["observe", {"inventory": {}}]])
    assert verify_events(zombie, [["onSave", {"onSave": "zombie_killed"}]])


def test_smelted_iron_task_verifies_ingots_not_raw_ore():
    iron = next(task for task in PEAM_TASKS if task.task_id == "T8")
    assert iron.target == "iron_ingot"
    assert not verify_inventory(iron, {"iron_ore": 2})
    assert verify_inventory(iron, {"iron_ingot": 2})
