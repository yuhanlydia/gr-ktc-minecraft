import json

from evaluation.report_fast_kv import build_fast_kv_report


def test_fast_kv_report_uses_paired_context_seed_trials(tmp_path):
    trials = []
    for seed, base, quality, failed in (
        (1, 0, 1, 0), (2, 0, 1, 0), (3, 1, 1, 1), (4, 0, 1, 0),
    ):
        for condition, score in (
            ("no_memory", base), ("context_only", base),
            ("positive_all_layers", quality), ("failed_all_layers", failed),
            ("layer24_quality", quality),
        ):
            trials.append({
                "scene_id": "x", "seed": seed, "condition": condition,
                "score": float(score), "parser_valid": True,
            })
    summary = {
        condition: {
            "mean_score": 0.0,
            "full_successes": sum(
                item["score"] == 1 for item in trials if item["condition"] == condition
            ),
            "parser_valid": 4, "trials": 4,
        }
        for condition in {item["condition"] for item in trials}
    }
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"trials": trials, "summary": summary}))
    report = build_fast_kv_report(source)
    assert report["paired_trials"] == 4
    assert report["quality_comparisons"]["no_memory"]["quality_only_successes"] == 3
