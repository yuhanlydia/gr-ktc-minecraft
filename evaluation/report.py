from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .statistics import exact_mcnemar, paired_bootstrap_median_difference, wilson_interval


@dataclass(frozen=True)
class TrialResult:
    method: str
    task_id: str
    seed: int
    success: bool
    latency_seconds: float
    tokens: int
    parser_valid: bool

    @property
    def pair_key(self) -> tuple[str, int]:
        return self.task_id, self.seed


def load_results(path: str | Path) -> list[TrialResult]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(TrialResult(**json.loads(line)))
    return rows


def compare_methods(rows: list[TrialResult], left_method: str, right_method: str) -> dict:
    left = {row.pair_key: row for row in rows if row.method == left_method}
    right = {row.pair_key: row for row in rows if row.method == right_method}
    if set(left) != set(right) or not left:
        raise ValueError("methods do not contain identical non-empty task-seed pairs")
    keys = sorted(left)
    left_success = [left[key].success for key in keys]
    right_success = [right[key].success for key in keys]
    discordant_left, discordant_right, p_value = exact_mcnemar(left_success, right_success)
    latency = paired_bootstrap_median_difference(
        [left[key].latency_seconds for key in keys],
        [right[key].latency_seconds for key in keys],
    )
    left_count = sum(left_success)
    right_count = sum(right_success)
    return {
        "pairs": len(keys),
        left_method: {
            "successes": left_count,
            "rate": left_count / len(keys),
            "wilson_95": wilson_interval(left_count, len(keys)),
            "parser_valid_rate": sum(left[key].parser_valid for key in keys) / len(keys),
        },
        right_method: {
            "successes": right_count,
            "rate": right_count / len(keys),
            "wilson_95": wilson_interval(right_count, len(keys)),
            "parser_valid_rate": sum(right[key].parser_valid for key in keys) / len(keys),
        },
        "mcnemar": {
            "left_only": discordant_left,
            "right_only": discordant_right,
            "exact_two_sided_p": p_value,
        },
        "latency_median_difference": {
            "estimate": latency[0], "ci_low": latency[1], "ci_high": latency[2]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("left_method")
    parser.add_argument("right_method")
    args = parser.parse_args()
    print(json.dumps(compare_methods(load_results(args.results), args.left_method, args.right_method), indent=2))


if __name__ == "__main__":
    main()

