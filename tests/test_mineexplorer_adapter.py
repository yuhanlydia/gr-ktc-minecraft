from pathlib import Path

from acquisition.mineexplorer import load_mineexplorer


DATA = Path(__file__).parents[1] / "data/MineExplorer-Benchmark/benchmark.jsonl"


def test_all_mineexplorer_scenarios_parse_and_have_unique_ids():
    scenarios = list(load_mineexplorer(DATA))
    assert len(scenarios) == 813
    assert len({scenario.scene_id for scenario in scenarios}) == 813


def test_group_is_stable_and_extracts_initial_inventory():
    scenario = next(load_mineexplorer(DATA))
    left = scenario.acquisition_group(seed=42)
    right = scenario.acquisition_group(seed=42)
    assert left.group_id == right.group_id
    assert left.context.inventory == (("oak_log", 10),)
    assert left.rollout_count == 4


def test_seed_changes_context_and_group_identity():
    scenario = next(load_mineexplorer(DATA))
    seed_42 = scenario.acquisition_group(seed=42)
    seed_43 = scenario.acquisition_group(seed=43)
    assert seed_42.context.context_id != seed_43.context.context_id
    assert seed_42.group_id != seed_43.group_id

