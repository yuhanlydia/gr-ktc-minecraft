import pytest

from evaluation.report_peam import build_report


def test_report_rejects_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="expected 99 trials"):
        build_report({"trials": [], "paired_world_restore": True})
