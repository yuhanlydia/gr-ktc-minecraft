from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.statistics import exact_mcnemar, wilson_interval


def _exact_sign_test(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(count, index) for index in range(tail + 1)) / 2**count
    return min(1.0, 2 * probability)


def build_fast_kv_report(source: str | Path) -> dict:
    payload = json.loads(Path(source).read_text())
    paired = defaultdict(dict)
    for trial in payload["trials"]:
        paired[(trial["scene_id"], trial["seed"])][trial["condition"]] = trial
    quality_name = "layer24_quality"
    comparisons = {}
    for other in (
        "no_memory", "context_only", "positive_all_layers", "failed_all_layers"
    ):
        quality_success = [
            row[quality_name]["score"] == 1.0 for row in paired.values()
        ]
        other_success = [row[other]["score"] == 1.0 for row in paired.values()]
        quality_only, other_only, mcnemar_p = exact_mcnemar(
            quality_success, other_success
        )
        score_wins = sum(
            row[quality_name]["score"] > row[other]["score"]
            for row in paired.values()
        )
        score_losses = sum(
            row[quality_name]["score"] < row[other]["score"]
            for row in paired.values()
        )
        comparisons[other] = {
            "quality_only_successes": quality_only,
            "other_only_successes": other_only,
            "mcnemar_two_sided_p": mcnemar_p,
            "quality_score_wins": score_wins,
            "quality_score_losses": score_losses,
            "score_sign_test_two_sided_p": _exact_sign_test(
                score_wins, score_losses
            ),
        }
    failed = [row["failed_all_layers"]["score"] == 1.0 for row in paired.values()]
    baseline = [row["no_memory"]["score"] == 1.0 for row in paired.values()]
    failed_only, baseline_only, failed_p = exact_mcnemar(failed, baseline)
    conditions = {}
    for condition, summary in payload["summary"].items():
        conditions[condition] = {
            **summary,
            "success_wilson_95ci": wilson_interval(
                summary["full_successes"], summary["trials"]
            ),
        }
    return {
        "protocol": "real-fast-kv-gate2-statistics-v1",
        "paired_trials": len(paired),
        "contexts": sorted({key[0] for key in paired}),
        "conditions": conditions,
        "quality_comparisons": comparisons,
        "failed_vs_no_memory": {
            "failed_only_successes": failed_only,
            "no_memory_only_successes": baseline_only,
            "mcnemar_two_sided_p": failed_p,
        },
        "gate2_pass": (
            comparisons["no_memory"]["mcnemar_two_sided_p"] < 0.05
            and comparisons["failed_all_layers"]["mcnemar_two_sided_p"] < 0.05
            and failed_p >= 0.05
            and payload["summary"][quality_name]["parser_valid"]
            == payload["summary"][quality_name]["trials"]
        ),
        "scope_warning": "Mechanism gate on two contexts; not PEAM held-out success.",
    }


if __name__ == "__main__":
    report = build_fast_kv_report(ROOT / "results/layer24_quality_control.json")
    output = ROOT / "results/fast_kv_gate2_statistics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
