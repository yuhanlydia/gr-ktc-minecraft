#!/usr/bin/env python3
"""Analyze the AndroidWorld LatentSkill survival gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


MODES = ("base", "context", "positive_all", "failed", "quality_all", "clsc")


def _mode_metrics(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    subset = [record for record in records if record.get("mode") == mode]
    if not subset:
        return {
            "trials": 0,
            "successes": 0,
            "success_rate": None,
            "parser_valid_rate": None,
            "mean_steps": None,
        }
    successes = sum(bool(record.get("success")) for record in subset)
    parser_valid = sum(bool(record.get("parser_valid", True)) for record in subset)
    step_values = [float(record.get("steps", 0)) for record in subset]
    return {
        "trials": len(subset),
        "successes": successes,
        "success_rate": successes / len(subset),
        "parser_valid_rate": parser_valid / len(subset),
        "mean_steps": mean(step_values) if step_values else None,
    }


def _paired(
    records: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        family = str(record.get("family", ""))
        instance = str(record.get("instance_id", ""))
        mode = str(record.get("mode", ""))
        if family and instance and mode:
            lookup[(family, instance, mode)] = record
    pairs = []
    keys = sorted({(family, instance) for family, instance, mode in lookup if mode == left})
    for family, instance in keys:
        a = lookup.get((family, instance, left))
        b = lookup.get((family, instance, right))
        if a is not None and b is not None:
            pairs.append((bool(a.get("success")), bool(b.get("success"))))
    return {
        "paired_trials": len(pairs),
        f"{left}_only_wins": sum(a and not b for a, b in pairs),
        f"{right}_only_wins": sum(b and not a for a, b in pairs),
        "both_success": sum(a and b for a, b in pairs),
        "both_fail": sum((not a) and (not b) for a, b in pairs),
        "paired_success_delta": (
            sum(int(a) - int(b) for a, b in pairs) / len(pairs)
            if pairs else None
        ),
    }


def _per_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    families = sorted({str(record.get("family", "")) for record in records if record.get("family")})
    for family in families:
        subset = [record for record in records if str(record.get("family")) == family]
        metrics = {mode: _mode_metrics(subset, mode) for mode in MODES}
        base_rate = metrics["base"]["success_rate"]
        clsc_rate = metrics["clsc"]["success_rate"]
        output[family] = {
            "mode_metrics": metrics,
            "clsc_minus_base": (
                clsc_rate - base_rate
                if clsc_rate is not None and base_rate is not None else None
            ),
        }
    return output


def analyze(summary: dict[str, Any]) -> dict[str, Any]:
    """Compute gate metrics from one runner summary."""
    records = list(summary.get("evaluation", []))
    families = dict(summary.get("families", {}))
    mode_metrics = {mode: _mode_metrics(records, mode) for mode in MODES}
    per_family = _per_family(records)
    mixed_signal_family_count = sum(
        int(info.get("mixed_group_count", 0)) > 0
        and str(info.get("status", "")) == "ready"
        for info in families.values()
    )
    positive_family_count = sum(
        info.get("clsc_minus_base") is not None
        and float(info["clsc_minus_base"]) > 0
        for info in per_family.values()
    )
    paired = {
        f"clsc_vs_{baseline}": _paired(records, "clsc", baseline)
        for baseline in ("base", "context", "positive_all", "failed", "quality_all")
    }
    report = {
        "protocol": "latentskill-androidworld-analysis-v1",
        "mode_metrics": mode_metrics,
        "per_family": per_family,
        "paired": paired,
        "mixed_signal_family_count": int(mixed_signal_family_count),
        "positive_family_count": int(positive_family_count),
        "family_status": families,
    }
    report["gate"] = hard_gate(report)
    return report


def hard_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered favorable-setting survival criterion."""
    metrics = report.get("mode_metrics", {})

    def rate(mode: str) -> float | None:
        value = metrics.get(mode, {}).get("success_rate")
        return None if value is None else float(value)

    base = rate("base")
    context = rate("context")
    positive = rate("positive_all")
    clsc = rate("clsc")
    criteria = {
        "clsc_beats_base_by_5pp": (
            base is not None and clsc is not None and clsc - base >= 0.05 - 1e-12
        ),
        "clsc_beats_positive_all": (
            positive is not None and clsc is not None and clsc > positive
        ),
        "clsc_beats_context": (
            context is not None and clsc is not None and clsc > context
        ),
        "positive_in_at_least_two_families": int(
            report.get("positive_family_count", 0)
        ) >= 2,
        "mixed_signal_in_at_least_two_families": int(
            report.get("mixed_signal_family_count", 0)
        ) >= 2,
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "clsc_minus_base": (
            clsc - base if clsc is not None and base is not None else None
        ),
        "decision": (
            "GO: scale LatentSkill to paper benchmarks"
            if all(criteria.values())
            else "NO-GO: stop the latent-KV skill direction; do not rescue with extra modules"
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LatentSkill AndroidWorld Gate Report",
        "",
        "| Mode | Success | Parser valid | Mean steps |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        metrics = report["mode_metrics"][mode]
        success = metrics["success_rate"]
        valid = metrics["parser_valid_rate"]
        steps = metrics["mean_steps"]
        lines.append(
            f"| {mode} | "
            f"{success:.3f}" if success is not None else f"| {mode} | n/a"
        )
        # Rebuild the row to avoid format branching ambiguity.
        lines[-1] = (
            f"| {mode} | "
            f"{success:.3f} | " if success is not None else f"| {mode} | n/a | "
        ) + (
            f"{valid:.3f} | " if valid is not None else "n/a | "
        ) + (
            f"{steps:.2f} |" if steps is not None else "n/a |"
        )
    gate = report["gate"]
    lines.extend([
        "",
        f"**Decision:** {gate['decision']}",
        "",
        "## Hard-gate criteria",
    ])
    for name, passed in gate["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    report = analyze(summary)
    output = args.output or args.summary.with_name("report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(_markdown(report))
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
