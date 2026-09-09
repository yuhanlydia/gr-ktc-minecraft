from scripts.collect_mineexplorer_pilot import verifier_score_events


def test_intermediate_consumed_inventory_milestone_is_latched():
    milestones = (
        {"milestone_id": "planks", "rules": [{
            "type": "inventory_has", "params": {"item": "oak_planks", "min_count": 1}
        }]},
        {"milestone_id": "stairs", "rules": [{
            "type": "inventory_has", "params": {"item": "oak_stairs", "min_count": 1}
        }]},
    )
    common = {
        "status": {"position": {"x": 0, "y": 64, "z": 0}},
        "voxels": [],
    }
    events = [
        ["observe", {**common, "inventory": {"oak_planks": 4}}],
        ["observe", {**common, "inventory": {"oak_stairs": 4}}],
    ]
    assert verifier_score_events(milestones, events) == 1.0
