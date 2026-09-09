from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file

from .schema import AcquisitionGroup, RolloutMetadata, canonical_json


class ImmutableTrajectoryPool:
    """Append-only trajectory pool that becomes read-only after finalization."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    @property
    def finalized(self) -> bool:
        return self.manifest_path.exists()

    def add_group(self, group: AcquisitionGroup) -> Path:
        self._ensure_mutable()
        group_dir = self.root / "groups" / group.group_id
        group_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_text(group_dir / "group.json", canonical_json(asdict(group)) + "\n")
        return group_dir

    def add_rollout(
        self,
        metadata: RolloutMetadata,
        *,
        tokens: torch.Tensor,
        kv_by_layer: Mapping[int, torch.Tensor],
        execution_events: list[dict],
    ) -> Path:
        self._ensure_mutable()
        metadata.validate()
        group_dir = self.root / "groups" / metadata.group_id
        if not (group_dir / "group.json").exists():
            raise FileNotFoundError(f"unknown group {metadata.group_id}")
        trajectory_dir = group_dir / "trajectories" / metadata.trajectory_id
        trajectory_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_text(
            trajectory_dir / "metadata.json", canonical_json(asdict(metadata)) + "\n"
        )
        save_file({"token_ids": tokens.detach().cpu().long().contiguous()}, trajectory_dir / "tokens.safetensors")
        tensors = {
            f"layer_{int(layer)}_kv": tensor.detach().cpu().contiguous()
            for layer, tensor in kv_by_layer.items()
        }
        save_file(tensors, trajectory_dir / "kv.safetensors")
        self._atomic_text(
            trajectory_dir / "execution.jsonl",
            "".join(canonical_json(event) + "\n" for event in execution_events),
        )
        return trajectory_dir

    def finalize(self) -> dict:
        self._ensure_mutable()
        files = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path != self.manifest_path
        )
        entries = []
        for path in files:
            entries.append({
                "path": str(path.relative_to(self.root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        manifest = {"format": "gr-ktc-pool-v1", "files": entries}
        self._atomic_text(self.manifest_path, canonical_json(manifest) + "\n")
        return manifest

    def verify(self) -> None:
        if not self.finalized:
            raise RuntimeError("pool is not finalized")
        manifest = json.loads(self.manifest_path.read_text())
        for entry in manifest["files"]:
            path = self.root / entry["path"]
            if not path.is_file():
                raise RuntimeError(f"missing pool file: {entry['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise RuntimeError(f"pool checksum mismatch: {entry['path']}")

    def _ensure_mutable(self) -> None:
        if self.finalized:
            raise RuntimeError("trajectory pool is finalized and immutable")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

