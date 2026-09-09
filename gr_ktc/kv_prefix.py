from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .merge_b_softdtw import linear_resample


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor  # [1, kv_heads, memory_tokens, head_dim]
    value: torch.Tensor

    def validate(self) -> None:
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError("memory K/V must have equal [1, heads, tokens, dim] shapes")
        if self.key.shape[0] != 1 or self.key.shape[-2] == 0:
            raise ValueError("memory K/V requires batch one and at least one token")


@dataclass(frozen=True)
class KVPrefixMemory:
    layers: Mapping[int, LayerKV]
    context_id: str

    def validate(self, expected_layers: int | None = None) -> None:
        if not self.layers:
            raise ValueError("KV memory has no layers")
        token_counts = set()
        for layer_id, layer in self.layers.items():
            if layer_id < 0:
                raise ValueError("layer ids must be non-negative")
            layer.validate()
            token_counts.add(layer.key.shape[-2])
        if len(token_counts) != 1:
            raise ValueError("all KV memory layers must have the same token count")
        if expected_layers is not None and set(self.layers) != set(range(expected_layers)):
            missing = sorted(set(range(expected_layers)) - set(self.layers))
            raise ValueError(f"fast injection requires every model layer; missing {missing}")

    @property
    def token_count(self) -> int:
        self.validate()
        return next(iter(self.layers.values())).key.shape[-2]

    @classmethod
    def from_flattened(
        cls,
        flattened: Mapping[int, torch.Tensor],
        *,
        kv_heads: int,
        head_dim: int,
        context_id: str,
        key_scale: float = 1.0,
        value_scale: float = 1.0,
    ) -> "KVPrefixMemory":
        if key_scale < 0 or value_scale < 0:
            raise ValueError("key_scale and value_scale must be nonnegative")
        width = kv_heads * head_dim
        layers: dict[int, LayerKV] = {}
        for layer_id, states in flattened.items():
            if states.ndim != 2 or states.shape[1] != 2 * width:
                raise ValueError(
                    f"layer {layer_id} expected flattened width {2 * width}, got {tuple(states.shape)}"
                )
            key = (states[:, :width] * key_scale).reshape(
                -1, kv_heads, head_dim
            ).permute(1, 0, 2).unsqueeze(0)
            value = (states[:, width:] * value_scale).reshape(
                -1, kv_heads, head_dim
            ).permute(1, 0, 2).unsqueeze(0)
            layers[int(layer_id)] = LayerKV(key.contiguous(), value.contiguous())
        memory = cls(layers, context_id)
        memory.validate()
        return memory


def merge_raw_kv_trajectories(
    trajectories_by_layer: Mapping[int, list[torch.Tensor]],
    advantages: torch.Tensor,
    *,
    memory_tokens: int = 16,
    temperature: float = 1.0,
    negative_scale: float = 0.0,
) -> dict[int, torch.Tensor]:
    """Create same-length raw K/V supports for cache injection.

    This is the state-space realization shared by merge operators. Alignment can
    be replaced by soft-DTW/OT, while this function provides the conservative
    phase-resampled weighted barycenter used by method A and smoke tests.
    """
    positive = advantages > 0
    if memory_tokens < 1:
        raise ValueError("memory_tokens must be positive")
    if not positive.any():
        raise ValueError("raw KV merge requires positive-advantage trajectories")
    positive_weights = torch.softmax(advantages[positive] / temperature, dim=0)
    negative = advantages < 0
    negative_weights = (
        torch.softmax(-advantages[negative] / temperature, dim=0)
        if negative.any()
        else None
    )
    merged: dict[int, torch.Tensor] = {}
    for layer_id, sequences in trajectories_by_layer.items():
        if len(sequences) != advantages.numel():
            raise ValueError(f"layer {layer_id} rollout count differs from advantages")
        aligned = [
            (sequence.float().mean(0, keepdim=True) if memory_tokens == 1
             else linear_resample(sequence.float(), memory_tokens))
            for sequence in sequences
        ]
        positive_aligned = [
            sequence for sequence, keep in zip(aligned, positive, strict=True) if bool(keep)
        ]
        positive_center = sum(
            weight * sequence
            for weight, sequence in zip(positive_weights, positive_aligned, strict=True)
        )
        if negative.any() and negative_scale:
            negative_aligned = [
                sequence for sequence, keep in zip(aligned, negative, strict=True) if bool(keep)
            ]
            negative_center = sum(
                weight * sequence
                for weight, sequence in zip(negative_weights, negative_aligned, strict=True)
            )
            # Contrastive extrapolation from the failure center through success.
            positive_center = positive_center + negative_scale * (
                positive_center - negative_center
            )
        merged[int(layer_id)] = positive_center.to(sequences[0].dtype).contiguous()
    return merged


def append_memory_to_cache(
    cache: Any,
    memory: KVPrefixMemory,
    *,
    expected_layers: int,
    expected_context_id: str,
) -> Any:
    """Append matched-context memory after prompt prefill and before generation."""
    memory.validate(expected_layers)
    if memory.context_id != expected_context_id:
        raise ValueError("refusing to inject KV memory into a different context")

    if hasattr(cache, "update") and hasattr(cache, "get_seq_length"):
        for layer_id in range(expected_layers):
            layer = memory.layers[layer_id]
            cache_layer = cache.layers[layer_id]
            reference_key = getattr(
                cache_layer, "keys", getattr(cache_layer, "key_cache", None)
            )
            reference_value = getattr(
                cache_layer, "values", getattr(cache_layer, "value_cache", None)
            )
            if reference_key is None or reference_value is None:
                raise TypeError("unsupported Transformers dynamic cache layer")
            cache.update(
                layer.key.to(device=reference_key.device, dtype=reference_key.dtype),
                layer.value.to(device=reference_value.device, dtype=reference_value.dtype),
                layer_id,
            )
        return cache

    if len(cache) != expected_layers:
        raise ValueError("cache layer count does not match model")
    output = []
    for layer_id, layer_cache in enumerate(cache):
        key, value, *rest = layer_cache
        memory_layer = memory.layers[layer_id]
        memory_key = memory_layer.key.to(device=key.device, dtype=key.dtype)
        memory_value = memory_layer.value.to(device=value.device, dtype=value.dtype)
        output.append(
            (
                torch.cat((key, memory_key), dim=-2),
                torch.cat((value, memory_value), dim=-2),
                *rest,
            )
        )
    return tuple(output)
