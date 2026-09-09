from evaluation.mineexplorer_verifier import (
    VerifierState,
    WorldObject,
    verify_milestones,
)
from acquisition.mineexplorer import load_mineexplorer


def test_inventory_and_box_rules_produce_msr_and_tsr():
    milestones = [
        {
            "milestone_id": "have_logs",
            "rules": [{"type": "inventory_has", "params": {"item": "oak_log", "min_count": 2}}],
        },
        {
            "milestone_id": "place_dirt",
            "rules": [{
                "type": "count_in_box_at_least",
                "params": {
                    "kind": "block", "object": "dirt", "min": [0, 0, 0],
                    "max": [2, 2, 2], "min_count": 1,
                    "coordinate_frame": "spawn_relative",
                },
            }],
        },
    ]
    state = VerifierState(
        inventory={"oak_log": 2},
        player_position=(10, 64, 10),
        player_facing=(1, 0, 0),
        spawn_position=(10, 64, 10),
        objects=(WorldObject("block", "dirt", (11, 64, 11)),),
    )
    result = verify_milestones(milestones, state)
    assert result.milestone_success_rate == 1.0
    assert result.task_success


def test_position_facing_and_partial_completion():
    milestones = [
        {
            "milestone_id": "look",
            "rules": [{
                "type": "position_near_with_facing",
                "params": {
                    "target": [5, 0, 0], "max_distance": 10,
                    "facing_tolerance": 15, "coordinate_frame": "spawn_relative",
                },
            }],
        },
        {
            "milestone_id": "reach",
            "rules": [{
                "type": "position_inside_box",
                "params": {
                    "min": [4, -1, -1], "max": [6, 1, 1],
                    "coordinate_frame": "spawn_relative",
                },
            }],
        },
    ]
    state = VerifierState({}, (0, 0, 0), (1, 0, 0), (0, 0, 0), ())
    result = verify_milestones(milestones, state)
    assert result.milestone_success_rate == 0.5
    assert not result.task_success


def test_verifier_supports_every_rule_in_full_dataset():
    state = VerifierState({}, (0, 0, 0), (1, 0, 0), (0, 0, 0), ())
    for scenario in load_mineexplorer("data/MineExplorer-Benchmark/benchmark.jsonl"):
        result = verify_milestones(scenario.milestones, state)
        assert 0.0 <= result.milestone_success_rate <= 1.0
