#!/usr/bin/env python3
"""Run the PEAM 11-task/3-seed protocol with the local Qwen fast tier.

This is protocol-compatible with PEAM's published task list, retry budget and
environment-side success criterion.  It is intentionally labelled a local
reproduction: PEAM's Azure slow tier and unpublished implementation are not
available.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.parser_validity import VoyagerActionParser
from evaluation.peam_suite import PEAM_SEEDS, PEAM_TASKS, verify_events
from evaluation.statistics import exact_mcnemar, wilson_interval
from gr_ktc.generation import generate_with_kv_prefix
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation, primitive_programs


SYSTEM = """You control a Minecraft 1.19 Mineflayer bot. Return exactly one fenced
JavaScript block containing one complete async function with exactly one argument
named bot. Use only exploreUntil, mineBlock, craftItem, placeItem, smeltItem, and
killMob. Never use chat commands. Complete the task from the current inventory;
perform every prerequisite in order. mineBlock searches for blocks. Keep the
answer under 400 tokens. Required form: ```javascript
async function solveTask(bot) {
  await mineBlock(bot, "oak_log", 1);
  await craftItem(bot, "oak_planks", 4);
}
```"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/peam_compatible.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-capped", action="store_true",
        help="With --resume, rerun complete trials containing a generation at the old cap.",
    )
    parser.add_argument("--task-ids", nargs="*", default=[])
    parser.add_argument(
        "--conditions", nargs="+", choices=("base", "control", "full"),
        default=("base", "control", "full"),
    )
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--precision", choices=("bf16", "nf4"), default="bf16",
        help="PEAM uses bf16 inference; nf4 is diagnostic-only on 24GB hardware.",
    )
    parser.add_argument(
        "--world-snapshot", type=Path,
        default=ROOT / "runtime/world_snapshots/peam_seed42",
        help="Pristine world copied before every paired condition.",
    )
    return parser.parse_args()


def restore_pristine_world(snapshot: Path) -> None:
    """Restore only the dedicated Gate-4 world, never an arbitrary path."""
    snapshot = snapshot.resolve()
    snapshot_root = (ROOT / "runtime/world_snapshots").resolve()
    world_root = (ROOT / "runtime/server-1.19").resolve()
    world = (world_root / "world_gate4").resolve()
    if snapshot.parent != snapshot_root or not (snapshot / "level.dat").is_file():
        raise ValueError(f"invalid Gate-4 snapshot: {snapshot}")
    if world.parent != world_root or world.name != "world_gate4":
        raise RuntimeError(f"refusing unsafe world target: {world}")
    subprocess.run([str(ROOT / "scripts/stop_local_stack.sh")], check=True)
    if world.exists():
        shutil.rmtree(world)
    shutil.copytree(snapshot, world)
    subprocess.run([str(ROOT / "scripts/start_local_stack.sh")], check=True)


def summarize(trials: list[dict], conditions: tuple[str, ...] | list[str]) -> dict:
    summary: dict[str, dict] = {}
    for condition in conditions:
        rows = [row for row in trials if row["condition"] == condition]
        successes = sum(bool(row["task_success"]) for row in rows)
        summary[condition] = {
            "successes": successes,
            "trials": len(rows),
            "success_rate": successes / len(rows) if rows else None,
            "wilson_95ci": list(wilson_interval(successes, len(rows))) if rows else None,
            "parser_valid_calls": sum(
                sum(bool(attempt["parser_valid"]) for attempt in row["attempts"])
                for row in rows
            ),
            "calls": sum(len(row["attempts"]) for row in rows),
            "tokens": sum(
                sum(int(attempt["generated_tokens"]) for attempt in row["attempts"])
                for row in rows
            ),
        }
    comparisons = {}
    keyed = {
        (row["task_id"], row["seed"], row["condition"]): bool(row["task_success"])
        for row in trials
    }
    for left, right in (("full", "control"), ("full", "base"), ("control", "base")):
        common = sorted(
            (task.task_id, seed) for task in PEAM_TASKS for seed in PEAM_SEEDS
            if (task.task_id, seed, left) in keyed and (task.task_id, seed, right) in keyed
        )
        if common:
            left_outcomes = [keyed[t, s, left] for t, s in common]
            right_outcomes = [keyed[t, s, right] for t, s in common]
            left_only, right_only, p_value = exact_mcnemar(left_outcomes, right_outcomes)
            comparisons[f"{left}_vs_{right}"] = {
                "pairs": len(common), "left_only": left_only, "right_only": right_only,
                "mcnemar_two_sided_p": p_value,
            }
    return {"conditions": summary, "comparisons": comparisons}


def main() -> None:
    args = parse_args()
    selected = [task for task in PEAM_TASKS if not args.task_ids or task.task_id in args.task_ids]
    unknown = set(args.task_ids) - {task.task_id for task in selected}
    if unknown:
        raise ValueError(f"unknown task ids: {sorted(unknown)}")

    base, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision=args.precision
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        base, ROOT / "results/qlora_full_100", adapter_name="full"
    )
    model.load_adapter(ROOT / "results/qlora_control_100", adapter_name="control")
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    eos_ids = tuple(token for token in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(token, int))

    client = VoyagerHTTPClient(timeout_seconds=180)
    parser = VoyagerActionParser(ROOT / "third_party/voyager/voyager/env/mineflayer")
    programs = primitive_programs()
    trials: list[dict] = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        previous_cap = int(previous.get("max_new_tokens", args.max_new_tokens))
        trials = previous.get("trials", [])
        for row in trials:
            row.setdefault("generation_cap", previous_cap)
        if args.refresh_capped:
            trials = [
                row for row in trials
                if not any(
                    int(attempt["generated_tokens"]) >= int(row["generation_cap"])
                    for attempt in row["attempts"]
                )
            ]
    completed = {(row["task_id"], row["seed"], row["condition"]) for row in trials}
    started = time.perf_counter()

    def checkpoint() -> None:
        payload = {
            "protocol": "peam-compatible-local-qwen-v1",
            "scope_warning": (
                "Published PEAM task/seed/retry/verifier protocol with local Qwen only; "
                "not an official PEAM reproduction and not Azure-GPT-4o parity."
            ),
            "minecraft": "1.19",
            "precision": args.precision,
            "seeds": list(PEAM_SEEDS),
            "retry_budget": args.attempts,
            "max_new_tokens": args.max_new_tokens,
            "generation_caps_in_trials": sorted({
                int(row.get("generation_cap", args.max_new_tokens)) for row in trials
            }),
            "world_snapshot": str(args.world_snapshot.resolve()),
            "paired_world_restore": True,
            "summary": summarize(trials, list(args.conditions)),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "trials": trials,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for task in selected:
        for seed in PEAM_SEEDS:
            for condition in args.conditions:
                key = (task.task_id, seed, condition)
                if key in completed:
                    continue
                # Bot reset does not undo mined blocks. Restore an identical
                # immutable world before every method in a paired trial.
                restore_pristine_world(args.world_snapshot)
                initial_events = client.reset(
                    hard=True, kill_on_hard_reset=False,
                    setup_commands=task.setup_commands,
                )
                attempts = []
                success = False
                events = initial_events
                for attempt_index in range(args.attempts):
                    abort_trial = False
                    observation = final_observation(events)
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": "text", "text": (
                            f"Observation: {compact_observation(observation)}\n"
                            f"Task: {task.instruction}\nAttempt: {attempt_index + 1}/{args.attempts}"
                        )}]},
                    ]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    )
                    inputs = {name: value.to(model.device) for name, value in inputs.items()}
                    generator = torch.Generator(device=model.device).manual_seed(
                        seed * 100 + attempt_index
                    )
                    if condition == "base":
                        context = model.disable_adapter()
                    else:
                        model.set_adapter(condition)
                        context = nullcontext()
                    call_started = time.perf_counter()
                    with context:
                        generated = generate_with_kv_prefix(
                            model, inputs, memory=None,
                            context_id=None,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.7, top_p=0.9,
                            eos_token_ids=eos_ids,
                            generator=generator,
                        )
                    text = processor.tokenizer.decode(
                        generated[0], skip_special_tokens=True
                    )
                    parse_error = None
                    execution_error = None
                    parsed = None
                    try:
                        parsed = parser.parse(text)
                    except ValueError as exc:
                        parse_error = str(exc)
                    if parsed is not None:
                        try:
                            events = client.step(
                                code=parsed.exec_code,
                                programs=programs + "\n\n" + parsed.program_code,
                            )
                        except Exception as exc:
                            execution_error = f"{type(exc).__name__}: {exc}"
                            # A timed-out /step may still be executing inside
                            # Mineflayer. Sending a recovery request to the same
                            # bridge can deadlock as well. Record the failure,
                            # end this trial, and let the next paired condition
                            # restore its pristine world and restart the stack.
                            abort_trial = True
                    success = parsed is not None and verify_events(task, events)
                    attempts.append({
                        "attempt": attempt_index + 1,
                        "parser_valid": parsed is not None,
                        "task_success": success,
                        "parse_error": parse_error,
                        "execution_error": execution_error,
                        "generated_tokens": int(generated.shape[-1]),
                        "latency_seconds": time.perf_counter() - call_started,
                        "response": text,
                        "final_inventory": final_observation(events).get("inventory", {}),
                    })
                    if success:
                        break
                    if abort_trial:
                        break
                trial = {
                    "task_id": task.task_id, "category": task.category,
                    "task": task.instruction, "seed": seed, "condition": condition,
                    "task_success": success, "attempts": attempts,
                    "generation_cap": args.max_new_tokens,
                }
                trials.append(trial)
                completed.add(key)
                checkpoint()
                print(json.dumps({
                    "task_id": task.task_id, "seed": seed, "condition": condition,
                    "success": success, "attempts": len(attempts),
                }), flush=True)
    checkpoint()


if __name__ == "__main__":
    main()
