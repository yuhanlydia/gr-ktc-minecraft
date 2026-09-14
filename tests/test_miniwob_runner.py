from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from scripts.run_miniwob_latentskill_gate import (
    DEFAULT_TASKS,
    GeneratedAction,
    MiniwobQwenPolicy,
    build_manifest,
    build_action_prompt,
    can_resume_acquisition,
    memory_for_mode,
    run_episode,
    validate_same_fingerprint,
    _FlattenedBrowserEnv,
)


def _kv(value: float) -> dict[int, torch.Tensor]:
    return {0: torch.full((1, 4), value), 1: torch.full((1, 4), value + 1)}


class FakeEnv:
    def __init__(self, raw_reward=0.8, wrapper_reward=0.0, action_error=""):
        self.raw_reward = raw_reward
        self.wrapper_reward = wrapper_reward
        self.action_error = action_error
        self.actions = []
        self.closed = False

    def reset(self):
        return {
            "goal": "Click the requested button.",
            "axtree_txt": "[12] button 'Submit'",
            "last_action_error": "",
        }, {}

    def step(self, action):
        self.actions.append(action)
        return (
            {
                "goal": "Click the requested button.",
                "axtree_txt": "[12] button 'Submit'",
                "last_action_error": self.action_error,
            },
            self.wrapper_reward,
            True,
            False,
            {
                "task_info": {
                    "RAW_REWARD_GLOBAL": self.raw_reward,
                    "REWARD_REASON": "task ended",
                },
            },
        )

    def close(self):
        self.closed = True


@dataclass
class FakePolicy:
    text: str = "Reason: submit\nAction: click('12')"

    def __post_init__(self):
        self.calls = []

    def generate(self, prompt, *, capture_kv, memory, seed):
        self.calls.append((prompt, capture_kv, memory, seed))
        return GeneratedAction(self.text, _kv(1.0) if capture_kv else None)


def test_episode_uses_native_raw_reward_and_captures_action_kv():
    env = FakeEnv(raw_reward=0.8, wrapper_reward=0.0)
    policy = FakePolicy()
    record, trajectory = run_episode(
        lambda _family, _seed: env,
        policy,
        family="miniwob.click-dialog",
        task_seed=123,
        model_seed=456,
        acquisition=True,
        mode="base",
        memory=None,
        max_steps=4,
    )
    assert env.closed is True
    assert env.actions == ["click('12')"]
    assert record["raw_reward"] == 0.8
    assert record["acquisition_reward"] == 0.8
    assert record["success"] is True
    assert record["browsergym_wrapper_reward"] == 0.0
    assert trajectory is not None and trajectory[0].shape == (1, 4)


def test_episode_ignores_positive_wrapper_reward_for_partial_failure():
    env = FakeEnv(raw_reward=0.1, wrapper_reward=1.0)
    record, _ = run_episode(
        lambda _family, _seed: env,
        FakePolicy(),
        family="miniwob.find-greatest",
        task_seed=1,
        model_seed=2,
        acquisition=True,
        mode="base",
        memory=None,
        max_steps=1,
    )
    assert record["success"] is False
    assert record["acquisition_reward"] == 0.1


def test_episode_fails_closed_when_raw_reward_is_missing():
    with pytest.raises(ValueError, match="finite raw reward"):
        run_episode(
            lambda _family, _seed: FakeEnv(raw_reward=None),
            FakePolicy(),
            family="miniwob.click-dialog",
            task_seed=1,
            model_seed=2,
            acquisition=True,
            mode="base",
            memory=None,
            max_steps=1,
        )


def test_evaluation_injects_memory_and_records_parser_failure():
    marker = object()
    policy = FakePolicy(text="not an action")
    env = FakeEnv(raw_reward=-1.0, action_error="bad bid")
    record, trajectory = run_episode(
        lambda _family, _seed: env,
        policy,
        family="miniwob.form-sequence",
        task_seed=11,
        model_seed=22,
        acquisition=False,
        mode="clsc",
        memory=marker,
        max_steps=1,
    )
    assert env.actions == ["noop()"]
    assert policy.calls[0][1:] == (False, marker, 22)
    assert record["parser_valid"] is False
    assert record["action_errors"] == ["bad bid"]
    assert trajectory is None


def test_action_prompt_contains_goal_tree_history_and_error():
    prompt = build_action_prompt(
        "Do it",
        "[1] textbox 'Name'",
        [("fill('1', 'Ada')", "[2] button 'Save'")],
        "Timeout",
    )
    assert "Do it" in prompt
    assert "[1] textbox 'Name'" in prompt
    assert "fill('1', 'Ada')" in prompt
    assert "Timeout" in prompt
    assert "one action" in prompt.lower()


def test_qwen_policy_captures_for_acquisition_and_injects_for_evaluation(monkeypatch):
    class Tokenizer:
        eos_token_id = 2

        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return "Action: click('12')"

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    model = type(
        "Model",
        (),
        {
            "device": torch.device("cpu"),
            "config": type(
                "Config",
                (),
                {"text_config": type("Text", (), {"num_hidden_layers": 2})()},
            )(),
            "generation_config": type("Generation", (), {"eos_token_id": 2})(),
        },
    )()
    capture_calls = []
    inject_calls = []

    def capture(model_arg, inputs, **kwargs):
        capture_calls.append((model_arg, inputs, kwargs))
        return type(
            "Captured",
            (),
            {
                "all_generated_token_ids": torch.tensor([[3]]),
                "kv_by_layer": _kv(2.0),
            },
        )()

    def inject(model_arg, inputs, memory, **kwargs):
        inject_calls.append((model_arg, inputs, memory, kwargs))
        return torch.tensor([[3]])

    monkeypatch.setattr(
        "scripts.run_miniwob_latentskill_gate.generate_with_final_kv", capture
    )
    monkeypatch.setattr(
        "scripts.run_miniwob_latentskill_gate.generate_with_kv_prefix", inject
    )
    policy = MiniwobQwenPolicy(model, Processor(), max_new_tokens=32)
    acquired = policy.generate("prompt", capture_kv=True, memory=None, seed=7)
    marker = object()
    evaluated = policy.generate("prompt", capture_kv=False, memory=marker, seed=7)
    assert acquired.kv_by_layer is not None
    assert evaluated.kv_by_layer is None
    assert capture_calls[0][2]["layer_ids"] == [0, 1]
    assert inject_calls[0][2] is marker
    assert inject_calls[0][3]["context_id"] == "family:miniwob"


def test_quick_manifest_preregisters_fixed_tasks_seeds_and_four_modes(tmp_path):
    manifest = build_manifest(
        phase="quick",
        tasks=DEFAULT_TASKS,
        model_path=tmp_path / "model",
        browsergym_head="browser-head",
        miniwob_head="miniwob-head",
        base_seed=42,
    )
    assert manifest["rollouts_per_instance"] == 10
    assert manifest["test_instances_per_family"] == 8
    assert manifest["target_qualified_families"] == 2
    assert manifest["modes"] == ["base", "context", "positive_all", "clsc"]
    assert manifest["tasks"] == list(DEFAULT_TASKS)
    family = DEFAULT_TASKS[0]
    assert len(manifest["acquisition_model_seeds"][family]) == 10
    assert len(manifest["evaluation_task_seeds"][family]) == 8
    assert set(manifest["acquisition_model_seeds"][family]).isdisjoint(
        manifest["evaluation_model_seeds"][family]
    )


def test_resume_requires_result_and_kv_for_acquisition(tmp_path):
    result = tmp_path / "result.json"
    kv = tmp_path / "episode_kv.safetensors"
    assert can_resume_acquisition(result, kv) is False
    result.write_text("{}")
    assert can_resume_acquisition(result, kv) is False
    kv.write_bytes(b"tensor")
    assert can_resume_acquisition(result, kv) is True


def test_group_and_paired_modes_require_identical_goal_fingerprint():
    validate_same_fingerprint(
        [{"goal_fingerprint": "same"}, {"goal_fingerprint": "same"}],
        "group",
    )
    with pytest.raises(RuntimeError, match="different randomized goals"):
        validate_same_fingerprint(
            [{"goal_fingerprint": "a"}, {"goal_fingerprint": "b"}],
            "group",
        )


def test_memory_for_mode_uses_only_named_bundle_memories():
    context, positive, clsc = object(), object(), object()
    bundle = type(
        "Bundle",
        (),
        {"memories": lambda self: {"context": context, "positive_all": positive, "clsc": clsc}},
    )()
    assert memory_for_mode(bundle, "base") is None
    assert memory_for_mode(bundle, "context") is context
    assert memory_for_mode(bundle, "positive_all") is positive
    assert memory_for_mode(bundle, "clsc") is clsc


def test_flattened_browser_env_passes_task_seed_only_at_reset():
    class SeedEnv:
        def __init__(self):
            self.seed = None

        def reset(self, *, seed):
            self.seed = seed
            return {
                "axtree_object": {"nodes": []},
                "extra_element_properties": {},
            }, {}

        def close(self):
            pass

    raw = SeedEnv()
    wrapped = _FlattenedBrowserEnv(raw, lambda *args, **kwargs: "tree", task_seed=731)
    observation, _ = wrapped.reset()
    assert raw.seed == 731
    assert observation["axtree_txt"] == "tree"
