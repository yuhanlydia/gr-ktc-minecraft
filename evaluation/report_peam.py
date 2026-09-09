from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from evaluation.peam_suite import PEAM_SEEDS, PEAM_TASKS
from evaluation.statistics import exact_mcnemar, wilson_interval


CONDITIONS = ("base", "control", "full")


def build_report(payload: dict) -> dict:
    trials = payload.get("trials", [])
    expected = {
        (task.task_id, seed, condition)
        for task in PEAM_TASKS for seed in PEAM_SEEDS for condition in CONDITIONS
    }
    keys = [(row["task_id"], row["seed"], row["condition"]) for row in trials]
    counts = Counter(keys)
    actual = set(keys)
    errors = []
    if len(trials) != 99:
        errors.append(f"expected 99 trials, found {len(trials)}")
    if any(value != 1 for value in counts.values()):
        errors.append("trial keys are not unique")
    if actual != expected:
        errors.append(f"missing={len(expected-actual)}, unexpected={len(actual-expected)}")
    if payload.get("paired_world_restore") is not True:
        errors.append("paired world restoration was not recorded")
    if any(not 1 <= len(row.get("attempts", [])) <= 4 for row in trials):
        errors.append("one or more trials violate the 1..4 retry budget")
    lower_cap_hits = [
        (row["task_id"], row["seed"], row["condition"])
        for row in trials
        if int(row.get("generation_cap", 0)) < 2048
        and any(
            int(attempt["generated_tokens"]) >= int(row["generation_cap"])
            for attempt in row["attempts"]
        )
    ]
    if lower_cap_hits:
        errors.append(f"lower-cap truncations remain: {lower_cap_hits}")
    if errors:
        raise ValueError("; ".join(errors))

    methods = {}
    per_task = {}
    for condition in CONDITIONS:
        rows = [row for row in trials if row["condition"] == condition]
        calls = [attempt for row in rows for attempt in row["attempts"]]
        successes = sum(bool(row["task_success"]) for row in rows)
        methods[condition] = {
            "successes": successes,
            "trials": len(rows),
            "success_rate": successes / len(rows),
            "wilson_95ci": list(wilson_interval(successes, len(rows))),
            "parser_valid": sum(bool(call["parser_valid"]) for call in calls),
            "calls": len(calls),
            "parser_valid_rate": sum(bool(call["parser_valid"]) for call in calls) / len(calls),
            "median_call_latency_seconds": statistics.median(
                float(call["latency_seconds"]) for call in calls
            ),
            "tokens_per_task": sum(int(call["generated_tokens"]) for call in calls) / len(rows),
        }
    for task in PEAM_TASKS:
        per_task[task.task_id] = {
            condition: sum(
                bool(row["task_success"])
                for row in trials
                if row["task_id"] == task.task_id and row["condition"] == condition
            )
            for condition in CONDITIONS
        }

    comparisons = {}
    keyed = {(row["task_id"], row["seed"], row["condition"]): bool(row["task_success"])
             for row in trials}
    for left, right in (("full", "control"), ("full", "base"), ("control", "base")):
        left_values = [keyed[task.task_id, seed, left] for task in PEAM_TASKS for seed in PEAM_SEEDS]
        right_values = [keyed[task.task_id, seed, right] for task in PEAM_TASKS for seed in PEAM_SEEDS]
        left_only, right_only, p_value = exact_mcnemar(left_values, right_values)
        comparisons[f"{left}_vs_{right}"] = {
            "left_only": left_only, "right_only": right_only,
            "mcnemar_two_sided_p": p_value,
        }
    return {
        "protocol": "peam-compatible-local-qwen-report-v1",
        "precision": payload.get("precision", "unknown"),
        "audit": {
            "complete": True, "trials": 99, "unique_trials": 99,
            "paired_world_restore": True, "max_retries": 4,
            "generation_caps": sorted({int(row["generation_cap"]) for row in trials}),
            "lower_cap_truncations": 0,
            "peam_2048_cap_hits": sum(
                int(row["generation_cap"]) == 2048
                and any(int(attempt["generated_tokens"]) >= 2048 for attempt in row["attempts"])
                for row in trials
            ),
        },
        "methods": methods,
        "per_task_successes_out_of_3": per_task,
        "comparisons": comparisons,
        "reference_only": {"voyager_reported": "18/33", "peam_reported": "23/33"},
        "claim_boundary": (
            "Local Qwen-only stress test with published PEAM task/seed/retry/verifier structure. "
            "It is not an official PEAM reproduction and is not paired with the paper's GPT-4o tier."
        ),
    }


def markdown(report: dict) -> str:
    labels = {"base": "Local Qwen base", "control": "Shared QLoRA BC+DPO",
              "full": "GR-KTC slow (memory-free)"}
    precision = str(report.get("precision", "unknown")).upper()
    lines = [f"# Gate 4 — PEAM-compatible local-Qwen {precision} stress test", "", report["claim_boundary"], "",
             "| Method | Success | Wilson 95% CI | Parser valid | Median call | Tokens/task |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        row = report["methods"][condition]
        low, high = row["wilson_95ci"]
        lines.append(
            f"| {labels[condition]} | {row['successes']}/{row['trials']} "
            f"({100*row['success_rate']:.1f}%) | [{100*low:.1f}%, {100*high:.1f}%] | "
            f"{row['parser_valid']}/{row['calls']} | {row['median_call_latency_seconds']:.2f}s | "
            f"{row['tokens_per_task']:.1f} |"
        )
    lines.extend(["", "## Per-task successes / 3", "",
                  "| Task | Base | BC+DPO | GR-KTC |", "|---|---:|---:|---:|"])
    for task in PEAM_TASKS:
        row = report["per_task_successes_out_of_3"][task.task_id]
        lines.append(f"| {task.task_id} {task.instruction} | {row['base']} | {row['control']} | {row['full']} |")
    lines.extend(["", "## Paired tests", ""])
    for name, row in report["comparisons"].items():
        lines.append(
            f"- `{name}`: left-only={row['left_only']}, right-only={row['right_only']}, "
            f"two-sided exact McNemar p={row['mcnemar_two_sided_p']:.4g}."
        )
    lines.extend(["", "Published Voyager 18/33 and PEAM 23/33 are reference-only and are not used in paired tests.", ""])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/peam_compatible.json"
    report = build_report(json.loads(source.read_text()))
    (root / "results/peam_compatible_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "results/PEAM_COMPATIBLE_REPORT.md").write_text(markdown(report))


if __name__ == "__main__":
    main()
