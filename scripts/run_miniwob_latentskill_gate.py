#!/usr/bin/env python3
"""Run the fast LatentSkill/CLSC reliability gate on MiniWoB++."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.generation import generate_with_final_kv, generate_with_kv_prefix
from gr_ktc.latent_skill import concat_step_kv
from gr_ktc.miniwob_protocol import (
    extract_single_action,
    is_terminal_success,
    quality_reward,
)


@dataclass(frozen=True)
class GeneratedAction:
    text: str
    kv_by_layer: Mapping[int, torch.Tensor] | None = None


def _model_inputs(model: Any, processor: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _eos_ids(model: Any, processor: Any) -> tuple[int, ...]:
    values: list[int] = []
    for raw in (
        getattr(processor.tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if isinstance(raw, int):
            values.append(raw)
        elif isinstance(raw, (tuple, list)):
            values.extend(int(item) for item in raw if isinstance(item, int))
    return tuple(dict.fromkeys(values))


class MiniwobQwenPolicy:
    """Generate one BrowserGym action while capturing or injecting native K/V."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 96,
        acquisition_temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.acquisition_temperature = float(acquisition_temperature)
        self.top_p = float(top_p)
        self.layer_ids = list(range(model.config.text_config.num_hidden_layers))
        self.eos_ids = _eos_ids(model, processor)

    def _decode(self, ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        *,
        capture_kv: bool,
        memory: Any,
        seed: int,
    ) -> GeneratedAction:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        inputs = _model_inputs(self.model, self.processor, prompt)
        if capture_kv:
            generated = generate_with_final_kv(
                self.model,
                inputs,
                layer_ids=self.layer_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.acquisition_temperature,
                top_p=self.top_p,
            )
            return GeneratedAction(
                self._decode(generated.all_generated_token_ids),
                generated.kv_by_layer,
            )
        generator = torch.Generator(device=self.model.device).manual_seed(int(seed))
        ids = generate_with_kv_prefix(
            self.model,
            inputs,
            memory,
            context_id=getattr(memory, "context_id", "family:miniwob"),
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=self.top_p,
            eos_token_ids=self.eos_ids,
            generator=generator,
        )
        return GeneratedAction(self._decode(ids), None)


def build_action_prompt(
    goal: str,
    axtree: str,
    history: Sequence[tuple[str, str]],
    action_error: str,
) -> str:
    recent = list(history)[-4:]
    history_text = "\n".join(
        f"Action: {action}\nResulting accessibility tree:\n{tree}"
        for action, tree in recent
    ) or "(none)"
    error_text = action_error.strip() or "(none)"
    return f"""You are controlling a MiniWoB web task through BrowserGym.

Goal:
{goal}

Current accessibility tree (interactive elements have bid identifiers):
{axtree}

Recent history:
{history_text}

Last action error:
{error_text}

Choose exactly one action. Allowed forms are:
click('bid')
fill('bid', 'text')
select_option('bid', 'option')
press('bid', 'key')
clear('bid')
focus('bid')
scroll(0, pixels)
noop()

Respond with a short reason and one final line formatted as:
Action: <one action>
"""


def _goal_fingerprint(goal: str) -> str:
    return hashlib.sha256(str(goal).encode()).hexdigest()


def _tree_text(observation: Mapping[str, Any]) -> str:
    if "axtree_txt" not in observation:
        raise ValueError("preprocessed MiniWoB observation lacks axtree_txt")
    return str(observation["axtree_txt"])


def run_episode(
    env_factory: Callable[[str, int], Any],
    policy: Any,
    *,
    family: str,
    task_seed: int,
    model_seed: int,
    acquisition: bool,
    mode: str,
    memory: Any,
    max_steps: int,
) -> tuple[dict[str, Any], dict[int, torch.Tensor] | None]:
    """Run one seeded episode and trust only MiniWoB's native raw reward."""
    env = env_factory(str(family), int(task_seed))
    started = time.perf_counter()
    history: list[tuple[str, str]] = []
    captured: list[Mapping[int, torch.Tensor]] = []
    parser_valid_count = 0
    action_errors: list[str] = []
    wrapper_reward = 0.0
    final_info: Mapping[str, Any] = {}
    terminated = False
    truncated = False
    try:
        observation, _ = env.reset()
        goal = str(observation.get("goal", ""))
        if not goal:
            raise ValueError("MiniWoB observation has no goal")
        for step_index in range(int(max_steps)):
            tree = _tree_text(observation)
            prompt = build_action_prompt(
                goal,
                tree,
                history,
                str(observation.get("last_action_error", "")),
            )
            generated = policy.generate(
                prompt,
                capture_kv=bool(acquisition),
                memory=memory if not acquisition else None,
                seed=int(model_seed) + step_index,
            )
            try:
                action = extract_single_action(generated.text)
                parser_valid_count += 1
            except ValueError:
                action = "noop()"
            if acquisition and generated.kv_by_layer:
                captured.append(generated.kv_by_layer)
            observation, wrapper_reward, terminated, truncated, final_info = env.step(
                action
            )
            next_tree = _tree_text(observation)
            history.append((action, next_tree))
            error = str(observation.get("last_action_error", "")).strip()
            if error:
                action_errors.append(error)
            if terminated or truncated:
                break

        raw_reward = final_info.get("RAW_REWARD_GLOBAL")
        acquisition_reward = quality_reward(raw_reward)
        steps = len(history)
        record = {
            "family": str(family),
            "goal": goal,
            "goal_fingerprint": _goal_fingerprint(goal),
            "task_seed": int(task_seed),
            "model_seed": int(model_seed),
            "mode": str(mode),
            "raw_reward": float(raw_reward),
            "acquisition_reward": acquisition_reward,
            "success_threshold": 0.5,
            "success": is_terminal_success(raw_reward),
            "browsergym_wrapper_reward": float(wrapper_reward),
            "reward_reason": str(final_info.get("REWARD_REASON", "")),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps,
            "parser_valid": parser_valid_count == steps,
            "parser_valid_rate": parser_valid_count / steps if steps else 0.0,
            "action_errors": action_errors,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trajectory = concat_step_kv(captured) if acquisition and captured else None
        return record, trajectory
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "quick"), default="smoke")
    parser.parse_args()
    raise SystemExit("MiniWoB orchestration is not implemented yet")


if __name__ == "__main__":
    main()
