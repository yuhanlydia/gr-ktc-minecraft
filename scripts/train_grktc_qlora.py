#!/usr/bin/env python3
"""100-update PEAM-matched QLoRA consolidation on frozen Gate-2 teachers."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from gr_ktc.generation import teacher_forced_hidden_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.lora_objectives import (
    SlowLossWeights,
    anchor_kl,
    combine_slow_losses,
    dpo_loss,
    grta_loss,
    negative_subspace_penalty,
    weighted_causal_lm_loss,
)
from gr_ktc.lora_setup import attach_grktc_lora
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.run_local_qwen_action import compact_observation
from scripts.run_local_smoke_suite import SYSTEM


def sequence_logp(logits: torch.Tensor, prompt_tokens: int, response_ids: torch.Tensor):
    start = prompt_tokens - 1
    prediction = logits[:, start:start + response_ids.shape[-1]]
    return torch.gather(
        torch.log_softmax(prediction.float(), dim=-1),
        -1, response_ids.unsqueeze(-1),
    ).squeeze(-1).sum(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("full", "control"), default="full")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trajectory-weight", type=float, default=0.5)
    parser.add_argument("--negative-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    source = json.loads((ROOT / "results/fast_kv_cross_context.json").read_text())
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    client = VoyagerHTTPClient(timeout_seconds=120)

    def build_prompt(processor, scenario):
        events = client.reset(
            hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands
        )
        observation = final_observation(events)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": (
                f"Observation: {compact_observation(observation)}\n"
                f"Task: {scenario.task_text}"
            )}]},
        ]
        return processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )

    teacher_effects = {}
    positive_basis = negative_basis = None
    prompt_inputs_cpu = {}
    chosen_text = {}
    rejected_text = {}
    for scene_id in scene_ids:
        records = source["acquisitions"][scene_id]["records"]
        chosen_text[scene_id] = max(
            records, key=lambda item: (item["score"], item["parser_valid"])
        )["text"]
        rejected_text[scene_id] = min(
            records, key=lambda item: (item["score"], item["parser_valid"])
        )["text"]

    if args.objective == "full":
        teacher_model, processor = load_qwen3_vl_24gb(
            ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
        )
        flat = load_file(ROOT / "results/fast_kv_cross_context_memories.safetensors")
        config = teacher_model.config.text_config
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        all_positive, all_negative = [], []
        for scene_id in scene_ids:
            prompt = build_prompt(processor, scenarios[scene_id])
            prompt_inputs_cpu[scene_id] = {key: value.cpu() for key, value in prompt.items()}
            prompt_gpu = {key: value.to(teacher_model.device) for key, value in prompt.items()}
            response = processor.tokenizer(
                chosen_text[scene_id], add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(teacher_model.device)
            advantages = torch.tensor(source["acquisitions"][scene_id]["advantages"])
            pos_count = int((advantages > 0).sum())
            neg_count = int((advantages < 0).sum())
            positive = {
                layer: flat[f"scene_{scene_id}_positive_layer_{layer}"]
                for layer in range(config.num_hidden_layers)
            }
            failed = {
                layer: flat[f"scene_{scene_id}_failed_layer_{layer}"]
                for layer in range(config.num_hidden_layers)
            }
            center = {
                layer: (pos_count * positive[layer] + neg_count * failed[layer])
                / (pos_count + neg_count)
                for layer in range(config.num_hidden_layers)
            }
            quality = dict(center)
            quality[24] = flat[f"scene_{scene_id}_contrastive_layer_24"]
            quality_memory = KVPrefixMemory.from_flattened(
                quality, kv_heads=config.num_key_value_heads, head_dim=head_dim,
                context_id=f"{scene_id}:matched", value_scale=0.25,
            )
            failed_memory = KVPrefixMemory.from_flattened(
                failed, kv_heads=config.num_key_value_heads, head_dim=head_dim,
                context_id=f"{scene_id}:matched", value_scale=0.25,
            )
            no_memory = teacher_forced_hidden_with_kv_prefix(
                teacher_model, prompt_gpu, response, None,
                context_id=None, layer_id=24,
            )
            quality_state = teacher_forced_hidden_with_kv_prefix(
                teacher_model, prompt_gpu, response, quality_memory,
                context_id=f"{scene_id}:matched", layer_id=24,
            )
            failed_state = teacher_forced_hidden_with_kv_prefix(
                teacher_model, prompt_gpu, response, failed_memory,
                context_id=f"{scene_id}:matched", layer_id=24,
            )
            positive_effect = quality_state - no_memory
            negative_effect = failed_state - no_memory
            teacher_effects[scene_id] = positive_effect
            all_positive.append(positive_effect)
            all_negative.append(negative_effect)
        _, _, positive_basis = torch.pca_lowrank(
            torch.cat(all_positive), q=16, center=False, niter=4
        )
        _, _, negative_basis = torch.pca_lowrank(
            torch.cat(all_negative), q=16, center=False, niter=4
        )
        save_file({
            "positive_basis": positive_basis,
            "negative_basis": negative_basis,
            **{f"teacher_effect_{scene}": effect for scene, effect in teacher_effects.items()},
        }, ROOT / "results/grktc_teacher_effects.safetensors")
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # Processor is returned with the NF4 model below; prompts are built then.
        processor = None

    torch.cuda.reset_peak_memory_stats()
    model, train_processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", training=True, precision="nf4"
    )
    model = attach_grktc_lora(model, rank=32, alpha=64, dropout=0.05)
    processor = train_processor
    samples = {}
    for scene_id in scene_ids:
        if scene_id not in prompt_inputs_cpu:
            prompt_inputs_cpu[scene_id] = {
                key: value.cpu() for key, value in build_prompt(
                    processor, scenarios[scene_id]
                ).items()
            }
        prompt_ids = prompt_inputs_cpu[scene_id]["input_ids"]
        chosen_ids = processor.tokenizer(
            chosen_text[scene_id], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        rejected_ids = processor.tokenizer(
            rejected_text[scene_id], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        samples[scene_id] = {
            "prompt_tokens": prompt_ids.shape[-1],
            "chosen_response": chosen_ids,
            "rejected_response": rejected_ids,
            "chosen_input": torch.cat((prompt_ids, chosen_ids), dim=-1),
            "rejected_input": torch.cat((prompt_ids, rejected_ids), dim=-1),
        }

    references = {}
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for scene_id, sample in samples.items():
            chosen_input = sample["chosen_input"].to(model.device)
            rejected_input = sample["rejected_input"].to(model.device)
            chosen_output = model(
                input_ids=chosen_input, attention_mask=torch.ones_like(chosen_input),
                output_hidden_states=True, use_cache=False,
            )
            rejected_output = model(
                input_ids=rejected_input, attention_mask=torch.ones_like(rejected_input),
                use_cache=False,
            )
            prompt_tokens = sample["prompt_tokens"]
            references[scene_id] = {
                "chosen_logp": sequence_logp(
                    chosen_output.logits, prompt_tokens,
                    sample["chosen_response"].to(model.device),
                ).detach(),
                "rejected_logp": sequence_logp(
                    rejected_output.logits, prompt_tokens,
                    sample["rejected_response"].to(model.device),
                ).detach(),
                "hidden": chosen_output.hidden_states[25][
                    :, prompt_tokens:prompt_tokens + sample["chosen_response"].shape[-1]
                ].detach(),
                "prompt_logits": chosen_output.logits[:, :prompt_tokens].detach(),
            }

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4,
    )
    warmup = max(1, math.ceil(args.steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: ((step + 1) / warmup if step < warmup else
                      0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                         max(args.steps - warmup, 1)))),
    )
    weights = SlowLossWeights(
        dpo=1.0, trajectory=args.trajectory_weight,
        negative=args.negative_weight, anchor=0.01,
    )
    history = []
    started = time.perf_counter()
    model.train()
    for step_index in range(args.steps):
        scene_id = scene_ids[step_index % len(scene_ids)]
        sample = samples[scene_id]
        prompt_tokens = sample["prompt_tokens"]
        chosen_input = sample["chosen_input"].to(model.device)
        chosen_response = sample["chosen_response"].to(model.device)
        rejected_input = sample["rejected_input"].to(model.device)
        rejected_response = sample["rejected_response"].to(model.device)
        labels = chosen_input.clone()
        labels[:, :prompt_tokens] = -100
        optimizer.zero_grad(set_to_none=True)
        chosen_output = model(
            input_ids=chosen_input, attention_mask=torch.ones_like(chosen_input),
            output_hidden_states=True, use_cache=False,
        )
        rejected_output = model(
            input_ids=rejected_input, attention_mask=torch.ones_like(rejected_input),
            use_cache=False,
        )
        bc = weighted_causal_lm_loss(
            chosen_output.logits, labels,
            torch.ones(1, device=model.device),
        )
        dpo = dpo_loss(
            sequence_logp(chosen_output.logits, prompt_tokens, chosen_response),
            sequence_logp(rejected_output.logits, prompt_tokens, rejected_response),
            references[scene_id]["chosen_logp"],
            references[scene_id]["rejected_logp"], beta=0.1,
        )
        student_hidden = chosen_output.hidden_states[25][
            :, prompt_tokens:prompt_tokens + chosen_response.shape[-1]
        ].float()
        student_effect = student_hidden - references[scene_id]["hidden"].float()
        if args.objective == "full":
            target = teacher_effects[scene_id].to(model.device).unsqueeze(0)
            trajectory = grta_loss(
                student_effect, target, positive_basis.to(model.device)
            )
            negative = negative_subspace_penalty(
                student_effect, negative_basis.to(model.device)
            )
        else:
            trajectory = student_effect.sum() * 0
            negative = student_effect.sum() * 0
        anchor = anchor_kl(
            references[scene_id]["prompt_logits"].float(),
            chosen_output.logits[:, :prompt_tokens].float(),
        )
        total, metrics = combine_slow_losses(
            bc=bc, dpo=dpo, trajectory=trajectory, negative=negative,
            anchor=anchor, weights=weights,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
        )
        optimizer.step()
        scheduler.step()
        if step_index % 5 == 0 or step_index == args.steps - 1:
            history.append({"step": step_index + 1, "scene_id": scene_id,
                            "lr": scheduler.get_last_lr()[0], **metrics})

    adapter_dir = ROOT / f"results/qlora_{args.objective}_100"
    model.save_pretrained(adapter_dir)
    report = {
        "protocol": "grktc-qlora-consolidation-v1",
        "objective": args.objective,
        "steps": args.steps,
        "scene_ids": scene_ids,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "adapter_directory": str(adapter_dir.relative_to(ROOT)),
        "lora": {"rank": 32, "alpha": 64, "dropout": 0.05},
        "loss_weights": weights.__dict__,
    }
    output = ROOT / f"results/qlora_{args.objective}_100.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
