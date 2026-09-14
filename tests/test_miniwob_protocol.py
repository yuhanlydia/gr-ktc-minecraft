import math
from pathlib import Path

import pytest
import yaml

from gr_ktc.miniwob_protocol import (
    extract_single_action,
    is_terminal_success,
    phase_spec,
    protocol_seed,
    qualify_signal,
    quality_reward,
)


def test_raw_reward_is_finite_clipped_and_thresholded():
    assert quality_reward(-1.0) == 0.0
    assert quality_reward(0.25) == 0.25
    assert quality_reward(2.0) == 1.0
    assert is_terminal_success(0.4999) is False
    assert is_terminal_success(0.5) is True
    for invalid in (None, "bad", math.nan, math.inf):
        with pytest.raises(ValueError, match="finite raw reward"):
            quality_reward(invalid)


def test_quick_phase_is_the_fixed_reliability_protocol():
    quick = phase_spec("quick")
    assert quick.rollouts_per_instance == 10
    assert quick.test_instances == 8
    assert quick.target_families == 2
    assert quick.modes == ("base", "context", "positive_all", "clsc")
    smoke = phase_spec("smoke")
    assert smoke.rollouts_per_instance == 2
    with pytest.raises(ValueError, match="smoke or quick"):
        phase_spec("full")


def test_signal_requires_three_successes_three_failures_and_variance():
    qualified = qualify_signal([0.9, 0.8, 0.7, 0.1, 0.0, 0.2, 0.9, 0.1, 0.8, 0.2])
    assert qualified.qualified is True
    assert qualified.successes == 5
    assert qualified.failures == 5
    assert qualified.reward_std >= 0.15

    too_few_failures = qualify_signal([0.9] * 8 + [0.1] * 2)
    assert too_few_failures.qualified is False
    assert "at least 3 terminal failures" in too_few_failures.reason

    low_variance = qualify_signal([0.51] * 5 + [0.49] * 5)
    assert low_variance.qualified is False
    assert "reward std" in low_variance.reason


def test_protocol_seeds_are_deterministic_and_namespace_separated():
    first = protocol_seed("acquisition", "miniwob.form-sequence", 0, 42)
    assert first == protocol_seed("acquisition", "miniwob.form-sequence", 0, 42)
    assert first != protocol_seed("evaluation", "miniwob.form-sequence", 0, 42)
    assert first != protocol_seed("acquisition", "miniwob.form-sequence", 1, 42)
    assert 0 <= first < 2**31


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('Reason: choose it\nAction: click("12")', 'click("12")'),
        ("fill('7', 'hello')", "fill('7', 'hello')"),
        ("```python\nselect_option('4', 'CA')\n```", "select_option('4', 'CA')"),
        ("scroll(0, 500)", "scroll(0, 500)"),
    ],
)
def test_extract_single_action_accepts_one_safe_browsergym_call(raw, expected):
    assert extract_single_action(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "click('1'); click('2')",
        "page.goto('https://example.com')",
        "click(target)",
        "__import__('os').system('id')",
        "Action: unknown('1')",
        "no action here",
    ],
)
def test_extract_single_action_rejects_code_and_unknown_calls(raw):
    with pytest.raises(ValueError, match="single allowed BrowserGym action"):
        extract_single_action(raw)


def test_yaml_matches_executable_quick_protocol():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/latentskill_miniwob_24gb.yaml").read_text()
    )
    quick = phase_spec("quick")
    assert config["benchmark"]["browsergym_commit"] == (
        "9e779f087de9a65668b6974d11f9ce9816026e96"
    )
    assert config["benchmark"]["miniwob_commit"] == (
        "7fd85d71a4b60325c6585396ec4f48377d049838"
    )
    assert tuple(config["scan"]["tasks"]) == tuple(
        [
            "miniwob.form-sequence",
            "miniwob.choose-date",
            "miniwob.email-inbox",
            "miniwob.login-user-popup",
            "miniwob.social-media-some",
            "miniwob.use-autocomplete",
            "miniwob.navigate-tree",
            "miniwob.book-flight-nodelay",
        ]
    )
    assert config["quick"]["rollouts_per_instance"] == quick.rollouts_per_instance
    assert config["quick"]["fresh_instances"] == quick.test_instances
    assert tuple(config["quick"]["modes"]) == quick.modes
