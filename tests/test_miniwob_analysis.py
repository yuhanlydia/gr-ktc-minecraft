from scripts.analyze_miniwob_latentskill import analyze


MODES = ("base", "context", "positive_all", "clsc")


def _summary(patterns, *, qualified=("family-a", "family-b")):
    evaluation = []
    for family, by_mode in patterns.items():
        for mode in MODES:
            for index, success in enumerate(by_mode[mode]):
                evaluation.append(
                    {
                        "family": family,
                        "instance_id": f"eval-{index:02d}",
                        "mode": mode,
                        "success": bool(success),
                        "goal_fingerprint": f"{family}-{index}",
                        "model_seed": index + 100,
                    }
                )
    return {
        "protocol": "latentskill-miniwob-raw-reward-v1",
        "target_qualified_families": 2,
        "test_instances_per_family": 8,
        "modes": list(MODES),
        "families": {
            family: {"status": "qualified" if family in qualified else "no_quality_signal"}
            for family in patterns
        },
        "evaluation": evaluation,
    }


def test_positive_gate_requires_aggregate_and_per_family_paired_wins():
    summary = _summary(
        {
            "family-a": {
                "base": [1, 1, 0, 0, 0, 0, 0, 0],
                "context": [1, 1, 1, 0, 0, 0, 0, 0],
                "positive_all": [1, 1, 0, 0, 0, 0, 0, 0],
                "clsc": [1, 1, 1, 1, 1, 0, 0, 0],
            },
            "family-b": {
                "base": [1, 0, 0, 0, 0, 0, 0, 0],
                "context": [1, 1, 0, 0, 0, 0, 0, 0],
                "positive_all": [1, 0, 0, 0, 0, 0, 0, 0],
                "clsc": [1, 1, 1, 1, 0, 0, 0, 0],
            },
        }
    )
    report = analyze(summary)
    assert report["status"] == "complete"
    assert report["pass"] is True
    assert report["mode_metrics"]["clsc"]["success_rate"] == 9 / 16
    assert report["paired"]["base"]["wins"] == 6
    assert report["per_family"]["family-a"]["clsc_minus_base"] > 0
    assert report["per_family"]["family-b"]["clsc_minus_base"] > 0


def test_complete_tie_is_a_method_level_negative_result():
    pattern = [1, 1, 0, 0, 0, 0, 0, 0]
    summary = _summary(
        {
            family: {mode: list(pattern) for mode in MODES}
            for family in ("family-a", "family-b")
        }
    )
    report = analyze(summary)
    assert report["status"] == "complete"
    assert report["pass"] is False
    assert any("Base" in reason for reason in report["reasons"])


def test_insufficient_signal_is_not_a_method_failure():
    patterns = {
        "family-a": {mode: [0] * 8 for mode in MODES},
        "family-b": {mode: [0] * 8 for mode in MODES},
    }
    report = analyze(_summary(patterns, qualified=("family-a",)))
    assert report["status"] == "insufficient_quality_signal"
    assert report["pass"] is None


def test_incomplete_paired_modes_do_not_issue_a_decision():
    patterns = {
        family: {mode: [0] * 8 for mode in MODES}
        for family in ("family-a", "family-b")
    }
    summary = _summary(patterns)
    summary["evaluation"].pop()
    report = analyze(summary)
    assert report["status"] == "incomplete"
    assert report["pass"] is None
