from __future__ import annotations

from typing import Any

import torch


def attach_grktc_lora(
    model: Any,
    *,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
) -> Any:
    """Attach an unmerged shared QLoRA adapter to Qwen3-VL."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("PEFT is required for slow consolidation") from exc
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    adapter = get_peft_model(model, config)
    adapter.config.use_cache = False
    return adapter


@torch.no_grad()
def initialize_peft_layer(
    lora_layer: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    adapter_name: str = "default",
) -> None:
    """Copy closed-form factors into one PEFT LoRA layer without merging."""
    target_a = lora_layer.lora_A[adapter_name].weight
    target_b = lora_layer.lora_B[adapter_name].weight
    if target_a.shape != a.shape or target_b.shape != b.shape:
        raise ValueError(
            f"factor shapes do not match PEFT layer: A {a.shape}/{target_a.shape}, "
            f"B {b.shape}/{target_b.shape}"
        )
    target_a.copy_(a.to(device=target_a.device, dtype=target_a.dtype))
    target_b.copy_(b.to(device=target_b.device, dtype=target_b.dtype))

