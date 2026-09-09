from scripts.eval_peam_compatible import summarize


def _trial(condition: str, success: bool) -> dict:
    return {
        "task_id": "T1",
        "seed": 42,
        "condition": condition,
        "task_success": success,
        "attempts": [{"parser_valid": True, "generated_tokens": 7}],
    }


def test_summary_uses_paired_outcomes_for_mcnemar() -> None:
    result = summarize(
        [_trial("base", False), _trial("control", False), _trial("full", True)],
        ["base", "control", "full"],
    )
    comparison = result["comparisons"]["full_vs_control"]
    assert comparison == {
        "pairs": 1,
        "left_only": 1,
        "right_only": 0,
        "mcnemar_two_sided_p": 1.0,
    }
    assert result["conditions"]["full"]["tokens"] == 7
