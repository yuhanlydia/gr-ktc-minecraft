import json

from gr_ktc.prompt_decommitment import (
    DecommitmentState,
    PromptMode,
    action_signature,
    build_control_instruction,
    parse_control_response,
)


def test_parse_control_response_extracts_revocations_and_belief_from_fenced_json():
    text = """```json
    {"revoke_event_ids":["E2","E5"],"current_belief":"wood is still missing",
     "thought":"try another tree","action":{"forward":1},"memory_update":"x"}
    ```"""
    control = parse_control_response(text)
    assert control.revoke_event_ids == ("E2", "E5")
    assert control.current_belief == "wood is still missing"


def test_state_applies_only_existing_past_event_revocations():
    state = DecommitmentState()
    state.observe_decision(step=1, thought="go north", action={"forward": 1})
    state.observe_decision(step=2, thought="mine", action={"attack": 1})

    applied = state.apply_revocations(["E1", "E9", "E2"], current_step=2)

    assert applied == ("E1",)
    assert state.revoked_event_ids == ("E1",)


def test_render_ledger_keeps_revoked_commitment_visible_but_labels_it_invalid():
    state = DecommitmentState()
    state.observe_decision(step=1, thought="coal is collected", action={"attack": 1})
    state.observe_decision(step=2, thought="go trade", action={"forward": 1})
    state.apply_revocations(["E1"], current_step=2)

    text = state.render_ledger(max_events=8)

    assert "E1 [REVOKED]" in text
    assert "coal is collected" in text
    assert "E2 [ACTIVE]" in text


def test_decommitment_instruction_requires_discrete_revocation_without_hidden_verifier_signal():
    text = build_control_instruction(PromptMode.DECOMMIT)
    assert "revoke_event_ids" in text
    assert "current_belief" in text
    assert "milestone score" not in text.lower()
    assert "TSR" not in text
    assert "MSR" not in text


def test_reflection_instruction_has_no_revocation_interface():
    text = build_control_instruction(PromptMode.REFLECTION)
    assert "reconsider" in text.lower()
    assert "revoke_event_ids" not in text


def test_ignore_revoke_mode_matches_decommitment_prompt_but_does_not_apply_control():
    text = build_control_instruction(PromptMode.DECOMMIT_IGNORE)
    assert "revoke_event_ids" in text
    state = DecommitmentState(ignore_revocations=True)
    state.observe_decision(step=1, thought="go north", action={"forward": 1})
    assert state.apply_revocations(["E1"], current_step=2) == ()
    assert state.revoked_event_ids == ()


def test_action_signature_is_stable_to_zero_fields_and_key_order():
    a = {"forward": 1, "sprint": 1, "attack": 0, "camera": [0, 0]}
    b = {"camera": [0.0, 0.0], "sprint": 1, "forward": 1}
    assert action_signature(a) == action_signature(b)
