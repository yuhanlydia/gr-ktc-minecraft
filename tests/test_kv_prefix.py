import pytest
import torch

from gr_ktc.kv_prefix import (
    KVPrefixMemory,
    append_memory_to_cache,
    merge_raw_kv_trajectories,
)
from gr_ktc.generation import _sample_top_p


def test_flattened_memory_roundtrip_and_legacy_append():
    flattened = {layer: torch.randn(3, 16) for layer in range(2)}
    memory = KVPrefixMemory.from_flattened(
        flattened, kv_heads=2, head_dim=4, context_id="ctx_a"
    )
    cache = tuple(
        (torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 5, 4))
        for _ in range(2)
    )
    result = append_memory_to_cache(
        cache, memory, expected_layers=2, expected_context_id="ctx_a"
    )
    assert result[0][0].shape == (1, 2, 8, 4)
    restored = result[0][0][:, :, 5:, :].squeeze(0).permute(1, 0, 2).reshape(3, -1)
    assert torch.allclose(restored, flattened[0][:, :8])


def test_value_scale_only_scales_values():
    states = torch.ones(2, 8)
    memory = KVPrefixMemory.from_flattened(
        {0: states}, kv_heads=1, head_dim=4, context_id="ctx", value_scale=0.25
    )
    assert torch.all(memory.layers[0].key == 1)
    assert torch.all(memory.layers[0].value == 0.25)


def test_key_and_value_scales_are_independent():
    states = torch.ones(2, 8)
    memory = KVPrefixMemory.from_flattened(
        {0: states}, kv_heads=1, head_dim=4, context_id="ctx",
        key_scale=1.5, value_scale=0.25,
    )
    assert torch.all(memory.layers[0].key == 1.5)
    assert torch.all(memory.layers[0].value == 0.25)

    with pytest.raises(ValueError, match="nonnegative"):
        KVPrefixMemory.from_flattened(
            {0: states}, kv_heads=1, head_dim=4, context_id="ctx", key_scale=-1,
        )


def test_memory_rejects_cross_context_and_missing_layers():
    memory = KVPrefixMemory.from_flattened(
        {0: torch.randn(2, 8)}, kv_heads=1, head_dim=4, context_id="ctx_a"
    )
    with pytest.raises(ValueError, match="missing"):
        memory.validate(expected_layers=2)
    with pytest.raises(ValueError, match="different context"):
        append_memory_to_cache(
            ((torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4)),),
            memory,
            expected_layers=1,
            expected_context_id="ctx_b",
        )


def test_raw_merge_produces_equal_phase_supports():
    trajectories = {
        0: [torch.zeros(3, 8), torch.ones(5, 8)],
        1: [torch.zeros(3, 8), torch.ones(5, 8)],
    }
    merged = merge_raw_kv_trajectories(
        trajectories, torch.tensor([-1.0, 1.0]), memory_tokens=4
    )
    assert merged[0].shape == (4, 8)
    assert torch.allclose(merged[0], torch.ones(4, 8))


def test_top_p_sampler_greedy_and_seeded_sampling():
    logits = torch.tensor([[0.0, 1.0, 3.0]])
    assert _sample_top_p(logits, temperature=0.0, top_p=0.9).item() == 2
    first = _sample_top_p(
        logits, temperature=1.0, top_p=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    second = _sample_top_p(
        logits, temperature=1.0, top_p=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)
