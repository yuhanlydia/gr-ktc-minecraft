#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def summarize_mode(rows: list[dict], mode: str) -> dict:
    selected = [row for row in rows if row.get("mode") == mode]
    n = len(selected)
    return {
        "mode": mode,
        "n": n,
        "tsr": sum(bool(row.get("task_success")) for row in selected) / n if n else 0.0,
        "mean_msr": sum(float(row.get("msr", 0.0)) for row in selected) / n if n else 0.0,
    }


def paired_rows(rows: list[dict], mode_a: str, mode_b: str) -> list[tuple[dict, dict]]:
    by_scene: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), {})[str(row["mode"])] = row
    return [
        (modes[mode_a], modes[mode_b])
        for _, modes in sorted(by_scene.items())
        if mode_a in modes and mode_b in modes
    ]


def bootstrap_mean_ci(values: Iterable[float], *, samples: int = 10000, seed: int = 0) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        value = float(array[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p(pairs: list[tuple[dict, dict]]) -> dict:
    a_only = sum(bool(a["task_success"]) and not bool(b["task_success"]) for a, b in pairs)
    b_only = sum(bool(b["task_success"]) and not bool(a["task_success"]) for a, b in pairs)
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_exact": p}


def comparison(rows: list[dict], mode_a: str, mode_b: str, *, bootstrap_samples: int, seed: int) -> dict:
    pairs = paired_rows(rows, mode_a, mode_b)
    diffs = [float(a["msr"]) - float(b["msr"]) for a, b in pairs]
    ci = bootstrap_mean_ci(diffs, samples=bootstrap_samples, seed=seed)
    return {
        "a": mode_a,
        "b": mode_b,
        "n_pairs": len(pairs),
        "mean_msr_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "msr_diff_bootstrap95": list(ci),
        "mcnemar": exact_mcnemar_p(pairs),
    }


def summarize_by_hop(rows: list[dict], modes: list[str]) -> dict:
    hops = sorted({int(row.get("hop_count", 0)) for row in rows})
    result = {}
    for hop in hops:
        subset = [row for row in rows if int(row.get("hop_count", 0)) == hop]
        result[str(hop)] = {mode: summarize_mode(subset, mode) for mode in modes}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    rows = list(payload.get("results", []))
    modes = ["base", "reflection", "decommit_ignore", "decommit"]
    report = {
        "protocol": payload.get("protocol"),
        "overall": {mode: summarize_mode(rows, mode) for mode in modes},
        "by_hop": summarize_by_hop(rows, modes),
        "comparisons": [
            comparison(rows, "decommit", "reflection", bootstrap_samples=args.bootstrap_samples, seed=args.seed),
            comparison(rows, "decommit", "decommit_ignore", bootstrap_samples=args.bootstrap_samples, seed=args.seed + 1),
            comparison(rows, "decommit", "base", bootstrap_samples=args.bootstrap_samples, seed=args.seed + 2),
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
