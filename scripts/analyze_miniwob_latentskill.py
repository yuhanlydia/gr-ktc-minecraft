#!/usr/bin/env python3
"""Analyze paired MiniWoB++ CLSC reliability-gate results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BASELINES = ("base", "context", "positive_all")


def _rate(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(bool(record.get("success")) for record in records) / len(records)


def _paired(
    records: Sequence[Mapping[str, Any]], baseline: str
) -> dict[str, int]:
    indexed = {
        (str(record["family"]), str(record["instance_id"]), str(record["mode"])): bool(
            record["success"]
        )
        for record in records
    }
    wins = losses = ties = 0
    instances = {(family, instance) for family, instance, _ in indexed}
    for family, instance in instances:
        clsc = indexed[(family, instance, "clsc")]
        other = indexed[(family, instance, baseline)]
        if clsc and not other:
            wins += 1
        elif other and not clsc:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def analyze(summary: Mapping[str, Any]) -> dict[str, Any]:
    target = int(summary.get("target_qualified_families", 2))
    expected_n = int(summary.get("test_instances_per_family", 8))
    modes = tuple(summary.get("modes", ()))
    families = summary.get("families", {})
    qualified = [
        str(family)
        for family, info in families.items()
        if info.get("status") == "qualified"
    ]
    if len(qualified) < target:
        return {
            "status": "insufficient_quality_signal",
            "pass": None,
            "qualified_families": qualified,
            "reasons": [f"found {len(qualified)} qualified families; require {target}"],
        }
    selected = qualified[:target]
    records = [
        record
        for record in summary.get("evaluation", [])
        if str(record.get("family")) in selected
    ]
    expected_modes = ("base", "context", "positive_all", "clsc")
    complete = True
    for family in selected:
        for mode in expected_modes:
            subset = [
                record
                for record in records
                if record.get("family") == family and record.get("mode") == mode
            ]
            if len(subset) != expected_n:
                complete = False
    if any(mode not in modes for mode in expected_modes):
        complete = False
    if not complete:
        return {
            "status": "incomplete",
            "pass": None,
            "qualified_families": selected,
            "reasons": ["paired evaluation does not contain every required mode and seed"],
        }

    for family in selected:
        for instance in {r["instance_id"] for r in records if r["family"] == family}:
            group = [
                record
                for record in records
                if record["family"] == family and record["instance_id"] == instance
            ]
            if len({record.get("goal_fingerprint") for record in group}) != 1:
                raise ValueError(f"unpaired goal fingerprints for {family}/{instance}")
            if len({record.get("model_seed") for record in group}) != 1:
                raise ValueError(f"unpaired model seeds for {family}/{instance}")

    mode_metrics = {}
    for mode in expected_modes:
        subset = [record for record in records if record["mode"] == mode]
        mode_metrics[mode] = {
            "successes": sum(bool(record["success"]) for record in subset),
            "episodes": len(subset),
            "success_rate": _rate(subset),
        }
    paired = {baseline: _paired(records, baseline) for baseline in BASELINES}
    per_family = {}
    positive_families = True
    for family in selected:
        subset = [record for record in records if record["family"] == family]
        by_mode = {
            mode: _rate([record for record in subset if record["mode"] == mode])
            for mode in expected_modes
        }
        family_pair = _paired(subset, "base")
        delta = by_mode["clsc"] - by_mode["base"]
        per_family[family] = {
            "mode_success_rates": by_mode,
            "clsc_minus_base": delta,
            "paired_clsc_vs_base": family_pair,
        }
        positive_families &= family_pair["wins"] > family_pair["losses"]

    rates = {mode: metrics["success_rate"] for mode, metrics in mode_metrics.items()}
    reasons: list[str] = []
    if rates["clsc"] - rates["base"] < 0.05:
        reasons.append("CLSC does not exceed Base by at least 5 percentage points")
    if rates["clsc"] <= rates["context"]:
        reasons.append("CLSC does not exceed Context")
    if rates["clsc"] <= rates["positive_all"]:
        reasons.append("CLSC does not exceed Positive-All")
    if not positive_families:
        reasons.append("CLSC lacks a positive paired Base difference in every family")
    return {
        "status": "complete",
        "pass": not reasons,
        "qualified_families": selected,
        "mode_metrics": mode_metrics,
        "paired": paired,
        "per_family": per_family,
        "reasons": reasons or ["all preregistered reliability criteria passed"],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# MiniWoB++ CLSC reliability gate", "", f"Status: **{report['status']}**"]
    if report.get("pass") is not None:
        lines.append(f"\nPass: **{report['pass']}**")
    if "mode_metrics" in report:
        lines.extend(["", "| Mode | Success | Rate |", "|---|---:|---:|"])
        for mode, metrics in report["mode_metrics"].items():
            lines.append(
                f"| {mode} | {metrics['successes']}/{metrics['episodes']} | "
                f"{metrics['success_rate']:.1%} |"
            )
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}" for reason in report.get("reasons", []))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    report = analyze(summary)
    output = args.output or args.summary.with_name("report.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(_markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
