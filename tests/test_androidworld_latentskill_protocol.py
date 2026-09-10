from scripts.run_androidworld_latentskill_gate import (
    default_task_names,
    generation_route,
    instance_seed,
    phase_spec,
)


def test_phase_specs_are_fixed_before_androidworld_results():
    assert phase_spec("smoke").train_instances == 1
    assert phase_spec("smoke").test_instances == 1
    assert phase_spec("pilot").train_instances == 2
    assert phase_spec("pilot").rollouts_per_instance == 4
    assert phase_spec("pilot").test_instances == 4
    assert phase_spec("full").train_instances == 4
    assert phase_spec("full").rollouts_per_instance == 4
    assert phase_spec("full").test_instances == 8


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
