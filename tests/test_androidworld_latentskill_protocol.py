from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from scripts.run_androidworld_latentskill_gate import (
    configure_slow_emulator_a11y,
    default_task_names,
    generation_route,
    instance_seed,
    phase_spec,
    repair_crashed_a11y_forwarder,
    safe_episode_length,
    verify_task_snapshots,
)


def test_phase_specs_are_fixed_before_androidworld_results():
    assert phase_spec("smoke").train_instances == 1
    assert phase_spec("smoke").test_instances == 1
    assert phase_spec("pilot").train_instances == 2
    assert phase_spec("pilot").rollouts_per_instance == 10
    assert phase_spec("pilot").test_instances == 4
    assert phase_spec("full").train_instances == 4
    assert phase_spec("full").rollouts_per_instance == 10
    assert phase_spec("full").test_instances == 8


def test_yaml_phase_rollout_counts_match_executable_protocol():
    config = yaml.safe_load(
        Path("configs/latentskill_androidworld_24gb.yaml").read_text()
    )

    for name in ("smoke", "pilot", "full"):
        assert config["phases"][name]["rollouts_per_instance"] == (
            phase_spec(name).rollouts_per_instance
        )


def test_default_task_families_are_exactly_preregistered_set():
    assert set(default_task_names()) == {
        "ContactsAddContact",
        "SimpleCalendarAddOneEvent",
        "MarkorCreateNote",
        "SimpleSmsSend",
        "ClockTimerEntry",
        "ExpenseAddSingle",
        "RecipeAddSingleRecipe",
        "VlcCreatePlaylist",
    }


def test_acquisition_and_test_instance_seeds_are_disjoint_and_deterministic():
    family = "ContactsAddContact"
    acquisition = {instance_seed("acquisition", family, i, 42) for i in range(16)}
    evaluation = {instance_seed("evaluation", family, i, 42) for i in range(16)}
    assert acquisition.isdisjoint(evaluation)
    assert instance_seed("acquisition", family, 3, 42) == instance_seed(
        "acquisition", family, 3, 42
    )
    assert instance_seed("acquisition", family, 3, 42) != instance_seed(
        "acquisition", "SimpleSmsSend", 3, 42
    )


def test_generation_route_keeps_skill_out_of_acquisition_and_summaries():
    acquisition = generation_route(
        call_kind="action", acquisition=True, mode="base"
    )
    assert acquisition.capture_kv is True
    assert acquisition.inject_memory is False

    evaluation = generation_route(
        call_kind="action", acquisition=False, mode="clsc"
    )
    assert evaluation.capture_kv is False
    assert evaluation.inject_memory is True

    base = generation_route(
        call_kind="action", acquisition=False, mode="base"
    )
    assert base.capture_kv is False
    assert base.inject_memory is False

    summary = generation_route(
        call_kind="summary", acquisition=False, mode="clsc"
    )
    assert summary.capture_kv is False
    assert summary.inject_memory is False


def test_a11y_repair_keeps_bound_service_and_resynchronizes_grpc():
    status = "Bound services:{com.google.androidenv.accessibilityforwarder}\nCrashed services:{}"
    wrapper = mock.Mock()
    wrapper.get_port.return_value = 43210
    env = SimpleNamespace(controller=SimpleNamespace(env=wrapper))
    with mock.patch(
        "scripts.run_androidworld_latentskill_gate.subprocess.run",
        return_value=SimpleNamespace(stdout=status),
    ) as run:
        assert repair_crashed_a11y_forwarder(env, "/opt/android/adb") is False
    assert run.call_count == 5
    wrapper.get_port.assert_called_once_with()


def test_a11y_repair_toggles_crashed_service_and_reconfigures_grpc():
    crashed = (
        "Bound services:{}\n"
        "Crashed services:{{com.google.androidenv.accessibilityforwarder/"
        "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder}}"
    )
    healthy = (
        "Bound services:{Service[label=com.google.androidenv.accessibilityforwarder]}\n"
        "Crashed services:{}"
    )
    dumpsys_count = 0

    def response(command, **unused_kwargs):
        nonlocal dumpsys_count
        if command[-2:] == ["dumpsys", "accessibility"]:
            dumpsys_count += 1
            return SimpleNamespace(stdout=crashed if dumpsys_count == 1 else healthy)
        return SimpleNamespace(stdout="")

    wrapper = mock.Mock()
    wrapper.get_port.return_value = 43210
    env = SimpleNamespace(controller=SimpleNamespace(env=wrapper))
    with (
        mock.patch(
            "scripts.run_androidworld_latentskill_gate.subprocess.run",
            side_effect=response,
        ) as run,
        mock.patch("scripts.run_androidworld_latentskill_gate.time.sleep"),
    ):
        assert repair_crashed_a11y_forwarder(env, "/opt/android/adb") is True
    commands = [call.args[0] for call in run.call_args_list]
    assert any(command[-2:] == ["keyevent", "KEYCODE_WAKEUP"] for command in commands)
    assert any("--async" in command for command in commands)
    wrapper.get_port.assert_called_once_with()


def test_slow_emulator_a11y_uses_patient_tree_retries():
    original = mock.Mock(return_value="forest")
    module = SimpleNamespace(get_a11y_tree=original)
    original_start = mock.Mock(return_value="started")
    original_type = mock.Mock(return_value=None)
    original_generic = mock.Mock(return_value="response")
    adb_utils = SimpleNamespace(
        start_activity=original_start,
        type_text=original_type,
        issue_generic_request=original_generic,
    )
    runtime = SimpleNamespace(
        android_world_controller=module,
        adb_utils=adb_utils,
    )
    configure_slow_emulator_a11y(runtime)
    assert runtime.transition_pause_seconds == 8.0

    assert module.get_a11y_tree("env") == "forest"
    original.assert_called_once_with("env", max_retries=30, sleep_duration=2.0)
    assert adb_utils.start_activity("activity", [], "env", timeout_sec=5) == "started"
    original_start.assert_called_once_with(
        "activity", [], "env", timeout_sec=60.0
    )
    adb_utils.type_text("hello", "env", timeout_sec=10)
    original_type.assert_called_once_with("hello", "env", timeout_sec=60.0)
    assert adb_utils.issue_generic_request(["shell", "id"], "env") == "response"
    original_generic.assert_called_once_with(
        ["shell", "id"], "env", timeout_sec=60.0
    )


def test_exception_episode_length_nan_is_recorded_as_zero():
    assert safe_episode_length(float("nan")) == 0
    assert safe_episode_length(None) == 0
    assert safe_episode_length(7.0) == 7


def test_snapshot_preflight_fails_before_sampling_when_snapshot_is_missing():
    task_type = SimpleNamespace(app_names=("simple calendar pro",))
    registry = {"Calendar": task_type}
    adb_utils = SimpleNamespace(
        get_adb_activity=lambda _name: "com.example.calendar/.MainActivity",
        extract_package_name=lambda activity: activity.split("/")[0],
    )
    with mock.patch(
        "scripts.run_androidworld_latentskill_gate.subprocess.run",
        return_value=SimpleNamespace(returncode=1),
    ):
        try:
            verify_task_snapshots(
                ("Calendar",), registry, adb_utils, "/opt/android/adb"
            )
        except RuntimeError as exc:
            assert "com.example.calendar" in str(exc)
        else:
            raise AssertionError("missing task snapshot should fail closed")
