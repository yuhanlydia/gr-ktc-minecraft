import pytest

from evaluation.statistics import (
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap_median_difference,
    wilson_interval,
)


def test_wilson_known_bounds_and_mcnemar_symmetry():
    low, high = wilson_interval(23, 33)
    # Standard score Wilson interval (without continuity correction). PEAM's
    # printed [0.530, 0.834] differs slightly from this reproducible definition.
    assert low == pytest.approx(0.5266, abs=1e-3)
    assert high == pytest.approx(0.8262, abs=1e-3)
    assert exact_mcnemar([True, True, False], [False, True, True]) == (1, 1, 1.0)


def test_holm_is_monotonic_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] <= adjusted["b"]


def test_paired_bootstrap_is_deterministic():
    result = paired_bootstrap_median_difference([2, 3, 4], [1, 1, 1], resamples=1000, seed=7)
    assert result == paired_bootstrap_median_difference([2, 3, 4], [1, 1, 1], resamples=1000, seed=7)
    assert result[0] == 2.0
