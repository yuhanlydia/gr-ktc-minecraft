from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch


PrecisionMode = Literal["bf16", "int8", "nf4"]
Workload = Literal["latent_pilot", "acquisition", "long_context", "training"]


def choose_24gb_precision(
    workload: Workload,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
) -> PrecisionMode:
    """Choose the highest practical precision with a 2 GiB safety margin."""
    total_tokens = prompt_tokens + max_new_tokens
    if workload == "latent_pilot" and total_tokens <= 2304:
        return "bf16"
    if workload == "acquisition" and total_tokens <= 4608:
        return "int8"
    return "nf4"


def load_qwen3_vl_24gb(
    model_path: str | Path,
    *,
    training: bool = False,
    precision: PrecisionMode = "nf4",
) -> tuple[Any, Any]:
    """Load Qwen3-VL-8B in an explicit RTX-3090 precision mode."""
    try:
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "install the train extra: pip install -e '.[train]'"
        ) from exc

    if training and precision != "nf4":
        raise ValueError("24 GB training is supported only with NF4 QLoRA")
    if precision == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "bf16":
        quantization = None
    else:
        raise ValueError(f"unsupported precision mode: {precision}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    if training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        model.config.use_cache = True
    return model, processor
