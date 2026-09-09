from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .kv_prefix import KVPrefixMemory, append_memory_to_cache


@dataclass
class GeneratedKVTrajectory:
    sequences: torch.Tensor
    all_generated_token_ids: torch.Tensor
    trajectory_token_ids: torch.Tensor
    kv_by_layer: dict[int, torch.Tensor]
    prompt_tokens: int


def _sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    remove = cumulative - probabilities >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(
        sorted_probabilities, num_samples=1, generator=generator
    )
    return sorted_indices.gather(-1, sampled)


@torch.no_grad()
def generate_with_kv_prefix(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    memory: KVPrefixMemory | None,
    *,
    context_id: str | None,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_ids: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate after appending an all-layer matched-context KV suffix.

    Prompt prefill is unchanged, so the first generated token is sampled from
    the ordinary prompt logits. The appended latent plan affects every later
    token. No teacher token IDs are injected. Raw trajectory keys retain their
    matched-context RoPE coordinates, while subsequent query positions advance
    beyond the prompt plus memory support.
    """
    if max_new_tokens < 1 or "input_ids" not in inputs:
        raise ValueError("input_ids and a positive max_new_tokens are required")
    expected_layers = model.config.text_config.num_hidden_layers
    if memory is not None:
        memory.validate(expected_layers)
        if context_id is None:
            raise ValueError("context_id is required when memory is provided")
    prefill = model(**inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    if memory is not None:
        cache = append_memory_to_cache(
            cache,
            memory,
            expected_layers=expected_layers,
            expected_context_id=context_id,
        )
    next_token = _sample_top_p(
        prefill.logits[:, -1],
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    generated = [next_token]
    finished = torch.zeros(
        next_token.shape[0], dtype=torch.bool, device=next_token.device
    )
    if eos_token_ids:
        finished |= torch.isin(next_token.squeeze(-1), torch.tensor(
            eos_token_ids, device=next_token.device
        ))

    prompt_attention = inputs.get("attention_mask")
    prompt_length = int(inputs["input_ids"].shape[-1])
    if prompt_attention is None:
        prompt_attention = torch.ones(
            inputs["input_ids"].shape,
            device=inputs["input_ids"].device,
            dtype=torch.long,
        )
    attention_mask = torch.cat(
        (
            prompt_attention,
            torch.ones(
                (prompt_attention.shape[0], memory.token_count if memory else 0),
                device=prompt_attention.device,
                dtype=prompt_attention.dtype,
            ),
        ),
        dim=-1,
    )
    for _ in range(1, max_new_tokens):
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=-1
        )
        # Qwen3-VL's outer multimodal model otherwise derives 3-D positions
        # from the full extended attention mask and may broadcast a one-token
        # query to the whole cache. This continuation is text-only, so all four
        # RoPE axes share the next scalar cache position.
        next_position = cache.get_seq_length()
        position_ids = torch.full(
            (4, next_token.shape[0], 1),
            next_position,
            device=next_token.device,
            dtype=torch.long,
        )
        step = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = step.past_key_values
        next_token = _sample_top_p(
            step.logits[:, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(next_token)
        if eos_token_ids:
            finished |= torch.isin(next_token.squeeze(-1), torch.tensor(
                eos_token_ids, device=next_token.device
            ))
            if bool(finished.all()):
                break
    return torch.cat(generated, dim=-1)


@torch.no_grad()
def teacher_forced_hidden_with_kv_prefix(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    response_ids: torch.Tensor,
    memory: KVPrefixMemory | None,
    *,
    context_id: str | None,
    layer_id: int,
) -> torch.Tensor:
    """Collect one layer's response-token states under an optional KV memory."""
    expected_layers = model.config.text_config.num_hidden_layers
    if not 0 <= layer_id < expected_layers:
        raise ValueError("layer_id out of range")
    if response_ids.ndim != 2 or response_ids.shape[0] != 1:
        raise ValueError("response_ids must be [1, response_tokens]")
    prefill = model(**prompt_inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    if memory is not None:
        if context_id is None:
            raise ValueError("context_id is required with memory")
        cache = append_memory_to_cache(
            cache, memory, expected_layers=expected_layers,
            expected_context_id=context_id,
        )
    attention = prompt_inputs.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(prompt_inputs["input_ids"])
    if memory is not None:
        attention = torch.cat((attention, torch.ones(
            (1, memory.token_count), device=attention.device, dtype=attention.dtype
        )), dim=-1)
    states = []
    for token_index in range(response_ids.shape[-1]):
        token = response_ids[:, token_index:token_index + 1]
        attention = torch.cat((attention, torch.ones_like(attention[:, :1])), dim=-1)
        position = cache.get_seq_length()
        output = model(
            input_ids=token,
            attention_mask=attention,
            position_ids=torch.full(
                (4, 1, 1), position, device=token.device, dtype=torch.long
            ),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = output.past_key_values
        states.append(output.hidden_states[layer_id + 1][:, -1].float().cpu())
    return torch.cat(states, dim=0)


def _legacy_cache(cache: Any) -> Any:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "layers"):
        pairs = []
        for layer in cache.layers:
            key = getattr(layer, "keys", getattr(layer, "key_cache", None))
            value = getattr(layer, "values", getattr(layer, "value_cache", None))
            if key is None or value is None:
                raise TypeError("unsupported Transformers cache layer")
            pairs.append((key, value))
        return tuple(pairs)
    return cache


@torch.no_grad()
def extract_generated_kv(
    cache: Any,
    *,
    layer_ids: list[int],
    prompt_tokens: int,
    generated_tokens: int,
    storage_dtype: torch.dtype = torch.float16,
) -> dict[int, torch.Tensor]:
    """Extract flattened token K/V states from a completed generation cache."""
    legacy = _legacy_cache(cache)
    output: dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        key, value = legacy[layer_id][:2]
        if key.ndim != 4 or value.shape != key.shape or key.shape[0] != 1:
            raise ValueError("expected batch-one K/V cache [1, heads, seq, head_dim]")
        available = max(0, key.shape[-2] - prompt_tokens)
        take = min(generated_tokens, available)
        start = prompt_tokens
        end = start + take
        k = key[0, :, start:end, :].permute(1, 0, 2).reshape(take, -1)
        v = value[0, :, start:end, :].permute(1, 0, 2).reshape(take, -1)
        output[layer_id] = torch.cat((k, v), dim=-1).to(
            device="cpu", dtype=storage_dtype
        ).contiguous()
    return output


@torch.no_grad()
def generate_with_final_kv(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> GeneratedKVTrajectory:
    """Generate once and extract action-token K/V from the returned final cache."""
    if "input_ids" not in inputs:
        raise ValueError("inputs must contain input_ids")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    result = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        use_cache=True,
        return_dict_in_generate=True,
    )
    generated = result.sequences[:, prompt_tokens:]
    cache = getattr(result, "past_key_values", None)
    if cache is None:
        raise RuntimeError("generation output did not return past_key_values")
    kv = extract_generated_kv(
        cache,
        layer_ids=layer_ids,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated.shape[-1],
    )
    lengths = {tensor.shape[0] for tensor in kv.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"selected layers returned inconsistent KV lengths: {lengths}")
    trajectory_length = lengths.pop()
    trajectory_tokens = generated[:, :trajectory_length].contiguous()
    return GeneratedKVTrajectory(
        result.sequences,
        generated,
        trajectory_tokens,
        kv,
        prompt_tokens,
    )
