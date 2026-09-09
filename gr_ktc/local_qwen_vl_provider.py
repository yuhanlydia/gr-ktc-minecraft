"""Local NF4 Qwen3-VL provider compatible with MineExplorer's OpenAI-style messages."""
from __future__ import annotations

import base64
import copy
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def _decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("only base64 data:image URLs are supported")
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("image data URL must be base64 encoded")
    raw = base64.b64decode(payload)
    return Image.open(BytesIO(raw)).convert("RGB")


def normalize_messages_for_hf(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MineExplorer/OpenAI image_url blocks into HF Qwen image blocks."""
    normalized = copy.deepcopy(messages)
    for message in normalized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                new_content.append(block)
                continue
            image_url = block.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str):
                raise ValueError("image_url block is missing string url")
            new_content.append({"type": "image", "image": _decode_data_url(url)})
        message["content"] = new_content
    return normalized


class LocalQwenVLProvider:
    """Inference-only local provider for Qwen3-VL-8B under a 16 GiB budget."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        gpu_memory_gib: int = 15,
        cpu_memory_gib: int = 48,
        precision: str = "nf4",
    ) -> None:
        if precision != "nf4":
            raise ValueError("prompt-decommitment 16GB profile currently supports nf4 only")
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.torch = torch
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: f"{gpu_memory_gib}GiB", "cpu": f"{cpu_memory_gib}GiB"},
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model.eval()
        self.model.config.use_cache = True
        self._base_seed = 0
        self._call_index = 0

    def set_seed(self, seed: int) -> None:
        self._base_seed = int(seed)
        self._call_index = 0

    @property
    def device(self):
        return self.model.device

    def peak_gpu_gib(self) -> float:
        if not self.torch.cuda.is_available():
            return 0.0
        return self.torch.cuda.max_memory_allocated() / 2**30

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 192,
        temperature: float = 0.7,
        response_format: dict | None = None,
        timeout: int = 120,
    ) -> str:
        del model, response_format, timeout
        normalized = normalize_messages_for_hf(messages)
        with self.torch.no_grad():
            inputs = self.processor.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[-1])
            self.torch.manual_seed(self._base_seed + self._call_index)
            self._call_index += 1
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                use_cache=True,
            )
            output_ids = generated[:, prompt_len:]
            return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
