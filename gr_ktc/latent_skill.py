"""Training-free verifier-contrastive latent skill construction.

LatentSkill preserves the successful GR-KTC Gate-2 mechanism: a task-family
context state is distributed across all model layers while verifier-derived
quality is injected only at one pre-registered layer.  The module is benchmark
agnostic and contains no AndroidWorld imports.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch

from .group_advantage import group_relative_advantage
from .kv_prefix import KVPrefixMemory, LayerKV, merge_raw_kv_trajectories


_MEMORY_MODES = ("context", "positive_all", "failed", "quality_all", "clsc")


def concat_step_kv(
    step_kv: Sequence[Mapping[int, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Concatenate action-selection K/V trajectories in temporal order."""
    if not step_kv:
        raise ValueError("step_kv must contain at least one action-selection call")
    expected_layers = set(step_kv[0])
    if not expected_layers:
        raise ValueError("step K/V mapping has no layers")
    output: dict[int, list[torch.Tensor]] = {layer: [] for layer in expected_layers}
    widths: dict[int, int] = {}
    for step_index, mapping in enumerate(step_kv):
        if set(mapping) != expected_layers:
            raise ValueError(
                f"step {step_index} layer ids differ from the first action call"
            )
        for layer, tensor in mapping.items():
            if tensor.ndim != 2 or tensor.shape[0] < 1:
                raise ValueError(
                    f"layer {layer} K/V must be non-empty rank-2 [tokens,width]"
                )
            width = int(tensor.shape[1])
            previous = widths.setdefault(layer, width)
            if width != previous:
                raise ValueError(f"layer {layer} K/V width changed across action calls")
            output[layer].append(tensor)
    return {
        int(layer): torch.cat(parts, dim=0).contiguous()
        for layer, parts in output.items()
    }


def grouped_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
) -> torch.Tensor:
    """Compute group-relative verifier advantages independently per instance."""
    if len(rewards) != len(group_ids) or not rewards:
        raise ValueError("rewards and group_ids must have the same non-zero length")
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(reward_tensor).all():
        raise ValueError("rewards contain NaN or infinity")
    output = torch.zeros_like(reward_tensor)
    ordered_groups: list[str] = []
    seen: set[str] = set()
    for raw in group_ids:
        group = str(raw)
        if group not in seen:
            seen.add(group)
            ordered_groups.append(group)
    for group in ordered_groups:
        indices = [i for i, item in enumerate(group_ids) if str(item) == group]
        local = group_relative_advantage(reward_tensor[indices])
        output[torch.tensor(indices, dtype=torch.long)] = local
    return output


@dataclass(frozen=True)
class FamilyMemoryBundle:
    family_id: str
    context: KVPrefixMemory
    positive_all: KVPrefixMemory
    failed: KVPrefixMemory
    quality_all: KVPrefixMemory
    clsc: KVPrefixMemory
    quality_layer: int
    memory_tokens: int
    value_scale: float
    negative_scale: float
    mixed_group_count: int
    rollout_count: int

    def memories(self) -> dict[str, KVPrefixMemory]:
        return {mode: getattr(self, mode) for mode in _MEMORY_MODES}


def _mixed_group_count(advantages: torch.Tensor, group_ids: Sequence[str]) -> int:
    count = 0
    for group in dict.fromkeys(map(str, group_ids)):
        indices = [i for i, raw in enumerate(group_ids) if str(raw) == group]
        local = advantages[indices]
        if bool((local > 0).any()) and bool((local < 0).any()):
            count += 1
    return count


def _validate_trajectories(
    trajectories: Sequence[Mapping[int, torch.Tensor]],
) -> tuple[list[int], int]:
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    layers = sorted(int(layer) for layer in trajectories[0])
    if not layers:
        raise ValueError("trajectory has no layers")
    expected = set(layers)
    widths: dict[int, int] = {}
    for trajectory_index, trajectory in enumerate(trajectories):
        if set(trajectory) != expected:
            raise ValueError(
                f"trajectory {trajectory_index} layer ids do not match the family"
            )
        for layer, states in trajectory.items():
            if states.ndim != 2 or states.shape[0] < 1:
                raise ValueError(
                    f"trajectory {trajectory_index} layer {layer} must be non-empty rank 2"
                )
            width = int(states.shape[1])
            previous = widths.setdefault(int(layer), width)
            if width != previous:
                raise ValueError(f"layer {layer} flattened K/V width changed")
    return layers, widths[layers[0]]


def _to_memory(
    flattened: Mapping[int, torch.Tensor],
    *,
    kv_heads: int,
    head_dim: int,
    context_id: str,
    value_scale: float,
) -> KVPrefixMemory:
    return KVPrefixMemory.from_flattened(
        flattened,
        kv_heads=kv_heads,
        head_dim=head_dim,
        context_id=context_id,
        value_scale=value_scale,
    )


def build_family_memory_bundle(
    *,
    family_id: str,
    trajectories: Sequence[Mapping[int, torch.Tensor]],
    rewards: Sequence[float],
    group_ids: Sequence[str],
    kv_heads: int,
    head_dim: int,
    quality_layer: int = 24,
    memory_tokens: int = 4,
    value_scale: float = 0.25,
    negative_scale: float = 0.5,
) -> FamilyMemoryBundle:
    """Build context, controls, and quality-localized CLSC for one task family."""
    if len(trajectories) != len(rewards) or len(rewards) != len(group_ids):
        raise ValueError("trajectory/reward/group counts must match")
    layers, flattened_width = _validate_trajectories(trajectories)
    if quality_layer not in layers:
        raise ValueError(f"quality layer {quality_layer} is absent from trajectories")
    if kv_heads < 1 or head_dim < 1 or flattened_width != 2 * kv_heads * head_dim:
        raise ValueError(
            "flattened K/V width must equal 2 * kv_heads * head_dim"
        )
    if memory_tokens < 1:
        raise ValueError("memory_tokens must be positive")
    if not family_id.strip():
        raise ValueError("family_id must be non-empty")

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(reward_tensor).all():
        raise ValueError("rewards contain NaN or infinity")
    advantages = grouped_advantages(rewards, group_ids)
    mixed_count = _mixed_group_count(advantages, group_ids)
    if mixed_count == 0:
        raise ValueError(
            "LatentSkill requires at least one mixed-outcome acquisition group"
        )

    success_mask = reward_tensor > 0
    failed_mask = ~success_mask
    if not bool(success_mask.any()) or not bool(failed_mask.any()):
        raise ValueError("family requires at least one successful and one failed rollout")

    by_layer = {
        layer: [trajectory[layer] for trajectory in trajectories]
        for layer in layers
    }
    common = dict(memory_tokens=memory_tokens)
    context_flat = merge_raw_kv_trajectories(
        by_layer, torch.ones(len(trajectories), dtype=torch.float32), **common
    )
    positive_flat = merge_raw_kv_trajectories(
        by_layer, success_mask.float(), **common
    )
    failed_flat = merge_raw_kv_trajectories(
        by_layer, failed_mask.float(), **common
    )
    quality_flat = merge_raw_kv_trajectories(
        by_layer,
        advantages,
        memory_tokens=memory_tokens,
        negative_scale=negative_scale,
    )
    clsc_flat = dict(context_flat)
    clsc_flat[quality_layer] = quality_flat[quality_layer]

    context_id = f"family:{family_id.strip()}"
    memories = {
        "context": _to_memory(
            context_flat,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_id=context_id,
            value_scale=value_scale,
        ),
        "positive_all": _to_memory(
            positive_flat,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_id=context_id,
            value_scale=value_scale,
        ),
        "failed": _to_memory(
            failed_flat,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_id=context_id,
            value_scale=value_scale,
        ),
        "quality_all": _to_memory(
            quality_flat,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_id=context_id,
            value_scale=value_scale,
        ),
        "clsc": _to_memory(
            clsc_flat,
            kv_heads=kv_heads,
            head_dim=head_dim,
            context_id=context_id,
            value_scale=value_scale,
        ),
    }
    return FamilyMemoryBundle(
        family_id=family_id.strip(),
        quality_layer=int(quality_layer),
        memory_tokens=int(memory_tokens),
        value_scale=float(value_scale),
        negative_scale=float(negative_scale),
        mixed_group_count=int(mixed_count),
        rollout_count=len(trajectories),
        **memories,
    )


def save_family_memory_bundle(
    bundle: FamilyMemoryBundle,
    tensor_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist one family bundle as safetensors plus portable JSON metadata."""
    tensor_path = Path(tensor_path)
    metadata_path = Path(metadata_path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for mode, memory in bundle.memories().items():
        memory.validate()
        for layer_id, layer in memory.layers.items():
            tensors[f"{mode}.layer_{layer_id}.key"] = (
                layer.key.detach().cpu().contiguous().clone()
            )
            tensors[f"{mode}.layer_{layer_id}.value"] = (
                layer.value.detach().cpu().contiguous().clone()
            )
    save_file(tensors, str(tensor_path))
    metadata = {
        "format": "latentskill-family-memory-v1",
        "family_id": bundle.family_id,
        "context_id": bundle.context.context_id,
        "quality_layer": bundle.quality_layer,
        "memory_tokens": bundle.memory_tokens,
        "value_scale": bundle.value_scale,
        "negative_scale": bundle.negative_scale,
        "mixed_group_count": bundle.mixed_group_count,
        "rollout_count": bundle.rollout_count,
        "modes": list(_MEMORY_MODES),
        "layer_ids": sorted(bundle.context.layers),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def load_family_memory_bundle(
    tensor_path: str | Path,
    metadata_path: str | Path,
) -> FamilyMemoryBundle:
    """Load a bundle saved by :func:`save_family_memory_bundle`."""
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("format") != "latentskill-family-memory-v1":
        raise ValueError("unsupported LatentSkill memory metadata format")
    tensors = load_file(str(tensor_path))
    context_id = str(metadata["context_id"])
    layer_ids = [int(layer) for layer in metadata["layer_ids"]]
    memories: dict[str, KVPrefixMemory] = {}
    for mode in _MEMORY_MODES:
        layers: dict[int, LayerKV] = {}
        for layer_id in layer_ids:
            key_name = f"{mode}.layer_{layer_id}.key"
            value_name = f"{mode}.layer_{layer_id}.value"
            if key_name not in tensors or value_name not in tensors:
                raise ValueError(f"missing {mode} layer {layer_id} tensor")
            layers[layer_id] = LayerKV(
                tensors[key_name].contiguous(), tensors[value_name].contiguous()
            )
        memory = KVPrefixMemory(layers, context_id)
        memory.validate()
        memories[mode] = memory
    return FamilyMemoryBundle(
        family_id=str(metadata["family_id"]),
        quality_layer=int(metadata["quality_layer"]),
        memory_tokens=int(metadata["memory_tokens"]),
        value_scale=float(metadata["value_scale"]),
        negative_scale=float(metadata["negative_scale"]),
        mixed_group_count=int(metadata["mixed_group_count"]),
        rollout_count=int(metadata["rollout_count"]),
        **memories,
    )
