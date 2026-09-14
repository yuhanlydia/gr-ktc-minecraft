from pathlib import Path

import torch

from gr_ktc.latent_skill import (
    build_family_memory_bundle,
    concat_step_kv,
    grouped_advantages,
    load_family_memory_bundle,
    save_family_memory_bundle,
)


def _trajectory(offset: float) -> dict[int, torch.Tensor]:
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )
    return {
        0: base + offset,
        1: (base * 2.0) + offset,
    }


def test_concat_step_kv_preserves_step_order():
    step1 = _trajectory(0.0)
    step2 = _trajectory(10.0)
    episode = concat_step_kv([step1, step2])
    assert episode[0].shape == (6, 4)
    assert torch.equal(episode[0][:3], step1[0])
    assert torch.equal(episode[0][3:], step2[0])


def test_grouped_advantages_are_local_to_instance():
    advantages = grouped_advantages(
        rewards=[1.0, 0.0, 1.0, 1.0],
        group_ids=["instance-a", "instance-a", "instance-b", "instance-b"],
    )
    assert torch.allclose(advantages[:2].mean(), torch.tensor(0.0), atol=1e-5)
    assert advantages[0] > 0 and advantages[1] < 0
    assert torch.equal(advantages[2:], torch.zeros(2))


def test_clsc_replaces_only_quality_layer_and_round_trips(tmp_path: Path):
    trajectories = [
        _trajectory(0.0),
        _trajectory(4.0),
        _trajectory(1.0),
        _trajectory(5.0),
    ]
    bundle = build_family_memory_bundle(
        family_id="DemoTask",
        trajectories=trajectories,
        rewards=[1.0, 0.0, 1.0, 0.0],
        group_ids=["a", "a", "b", "b"],
        kv_heads=1,
        head_dim=2,
        quality_layer=1,
        memory_tokens=2,
        value_scale=0.25,
        negative_scale=0.5,
    )

    assert bundle.clsc.context_id == "family:DemoTask"
    assert bundle.context.context_id == "family:DemoTask"
    assert torch.equal(
        bundle.clsc.layers[0].key,
        bundle.context.layers[0].key,
    )
    assert torch.equal(
        bundle.clsc.layers[0].value,
        bundle.context.layers[0].value,
    )
    assert not torch.equal(
        bundle.clsc.layers[1].value,
        bundle.context.layers[1].value,
    )
    assert bundle.mixed_group_count == 2

    tensor_path = tmp_path / "demo.safetensors"
    metadata_path = tmp_path / "demo.json"
    save_family_memory_bundle(bundle, tensor_path, metadata_path)
    restored = load_family_memory_bundle(tensor_path, metadata_path)

    assert restored.family_id == bundle.family_id
    assert restored.quality_layer == 1
    assert restored.memory_tokens == 2
    assert restored.mixed_group_count == 2
    for mode in ("context", "positive_all", "failed", "quality_all", "clsc"):
        original = getattr(bundle, mode)
        loaded = getattr(restored, mode)
        assert original.context_id == loaded.context_id
        for layer in original.layers:
            assert torch.equal(original.layers[layer].key, loaded.layers[layer].key)
            assert torch.equal(original.layers[layer].value, loaded.layers[layer].value)


def test_family_memory_requires_mixed_quality_signal():
    trajectories = [_trajectory(0.0), _trajectory(1.0), _trajectory(2.0), _trajectory(3.0)]
    try:
        build_family_memory_bundle(
            family_id="NoSignal",
            trajectories=trajectories,
            rewards=[1.0, 1.0, 1.0, 1.0],
            group_ids=["a", "a", "b", "b"],
            kv_heads=1,
            head_dim=2,
            quality_layer=1,
            memory_tokens=2,
        )
    except ValueError as exc:
        assert "mixed-outcome" in str(exc)
    else:
        raise AssertionError("expected no-quality-signal family to be rejected")
