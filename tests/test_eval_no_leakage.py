from acquisition.leakage import audit_evaluation_leakage
from acquisition.mineexplorer import load_mineexplorer


def test_exact_task_leakage_is_detected():
    scenario = next(load_mineexplorer("data/MineExplorer-Benchmark/benchmark.jsonl"))
    group = scenario.acquisition_group(seed=42)
    findings = audit_evaluation_leakage(
        [group], held_out_task_texts=[scenario.task_text.upper() + "!!!"]
    )
    assert findings and findings[0].reason == "exact normalized task match"

