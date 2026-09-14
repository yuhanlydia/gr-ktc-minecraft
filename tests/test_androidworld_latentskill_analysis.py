from scripts.analyze_androidworld_latentskill import _markdown, analyze, hard_gate


def _record(family, instance_id, mode, success, parser_valid=True, steps=5):
    return {
        "family": family,
        "instance_id": instance_id,
        "mode": mode,
        "success": bool(success),
        "parser_valid": bool(parser_valid),
        "steps": steps,
    }


def test_analysis_computes_paired_rates_and_hard_gate():
    families = {
        "A": {"status": "ready", "mixed_group_count": 2},
        "B": {"status": "ready", "mixed_group_count": 1},
        "C": {"status": "ready", "mixed_group_count": 1},
    }
    evaluation = []
    outcomes = {
        "A": {
            "base": [0, 0, 1, 0],
            "context": [0, 1, 1, 0],
            "positive_all": [0, 1, 1, 0],
            "failed": [0, 0, 0, 0],
            "quality_all": [1, 1, 1, 0],
            "clsc": [1, 1, 1, 1],
        },
        "B": {
            "base": [0, 1, 0, 0],
            "context": [0, 1, 0, 1],
            "positive_all": [1, 1, 0, 0],
            "failed": [0, 0, 0, 0],
            "quality_all": [1, 1, 0, 1],
            "clsc": [1, 1, 1, 1],
        },
        "C": {
            "base": [1, 1, 0, 0],
            "context": [1, 1, 0, 0],
            "positive_all": [1, 1, 0, 0],
            "failed": [0, 1, 0, 0],
            "quality_all": [1, 1, 1, 0],
            "clsc": [1, 1, 1, 0],
        },
    }
    for family, modes in outcomes.items():
        for mode, values in modes.items():
            for index, value in enumerate(values):
                evaluation.append(_record(family, f"{family}-{index}", mode, value))

    report = analyze({"families": families, "evaluation": evaluation})
    assert report["mode_metrics"]["clsc"]["success_rate"] > report["mode_metrics"]["base"]["success_rate"]
    assert report["paired"]["clsc_vs_base"]["clsc_only_wins"] > 0
    assert report["mixed_signal_family_count"] == 3
    assert report["positive_family_count"] >= 2

    decision = hard_gate(report)
    assert decision["pass"] is True
    assert all(decision["criteria"].values())


def test_hard_gate_rejects_no_advantage():
    summary = {
        "families": {
            "A": {"status": "ready", "mixed_group_count": 1},
            "B": {"status": "ready", "mixed_group_count": 1},
        },
        "evaluation": [],
    }
    for family in ("A", "B"):
        for instance_index in range(4):
            for mode in ("base", "context", "positive_all", "failed", "quality_all", "clsc"):
                summary["evaluation"].append(
                    _record(family, f"{family}-{instance_index}", mode, success=0)
                )
    report = analyze(summary)
    decision = hard_gate(report)
    assert decision["pass"] is False
    assert decision["criteria"]["clsc_beats_base_by_5pp"] is False


def test_partial_smoke_is_incomplete_instead_of_scientific_no_go():
    summary = {
        "phase": "smoke",
        "tasks": ["A", "B"],
        "modes": ["base", "clsc"],
        "test_instances_per_family": 1,
        "families": {"A": {"status": "ready", "mixed_group_count": 1}},
        "evaluation": [
            _record("A", "eval-00", "base", 1),
            _record("A", "eval-00", "clsc", 1),
        ],
    }
    gate = analyze(summary)["gate"]
    assert gate["status"] == "incomplete"
    assert gate["pass"] is None
    assert gate["decision"].startswith("INCOMPLETE")


def test_complete_smoke_is_integration_only_instead_of_scientific_no_go():
    summary = {
        "phase": "smoke",
        "tasks": ["A"],
        "modes": ["base", "clsc"],
        "test_instances_per_family": 1,
        "families": {"A": {"status": "ready", "mixed_group_count": 1}},
        "evaluation": [
            _record("A", "eval-00", "base", 1),
            _record("A", "eval-00", "clsc", 0),
        ],
    }
    gate = analyze(summary)["gate"]
    assert gate["status"] == "smoke_complete"
    assert gate["pass"] is None
    assert gate["decision"].startswith("SMOKE COMPLETE")


def test_smoke_markdown_marks_scientific_criteria_not_evaluated():
    summary = {
        "phase": "smoke",
        "tasks": ["A"],
        "modes": ["base", "clsc"],
        "test_instances_per_family": 1,
        "families": {"A": {"status": "ready", "mixed_group_count": 1}},
        "evaluation": [
            _record("A", "eval-00", "base", 1),
            _record("A", "eval-00", "clsc", 1),
        ],
    }

    markdown = _markdown(analyze(summary))

    assert "NOT EVALUATED — `clsc_beats_base_by_5pp`" in markdown
    assert "OBSERVED" not in markdown
