import json

import pytest
import torch

from acquisition.mineexplorer import load_mineexplorer
from acquisition.pool import ImmutableTrajectoryPool
from acquisition.schema import RolloutMetadata, stable_id


def test_pool_roundtrip_finalize_and_tamper_detection(tmp_path):
    data = "data/MineExplorer-Benchmark/benchmark.jsonl"
    group = next(load_mineexplorer(data)).acquisition_group(seed=42)
    pool = ImmutableTrajectoryPool(tmp_path / "pool")
    pool.add_group(group)
    metadata = RolloutMetadata(
        trajectory_id=stable_id("traj", {"group": group.group_id, "index": 0}),
        group_id=group.group_id,
        rollout_index=0,
        seed=42,
        verifier_score=1.0,
        parser_valid=True,
        task_success=True,
        generated_tokens=3,
        latency_seconds=0.5,
        model_id="test-model",
        selected_layers=(1, 2),
    )
    directory = pool.add_rollout(
        metadata,
        tokens=torch.tensor([1, 2, 3]),
        kv_by_layer={1: torch.randn(3, 4), 2: torch.randn(3, 4)},
        execution_events=[{"type": "verified", "success": True}],
    )
    manifest = pool.finalize()
    assert manifest["format"] == "gr-ktc-pool-v1"
    pool.verify()
    with pytest.raises(RuntimeError, match="immutable"):
        pool.add_group(group)

    metadata_path = directory / "metadata.json"
    record = json.loads(metadata_path.read_text())
    record["verifier_score"] = 0.0
    metadata_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        pool.verify()

