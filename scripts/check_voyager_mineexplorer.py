#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import requests
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHECKS: list[dict[str, object]] = []


def require(condition: bool, message: str) -> None:
    CHECKS.append({"ok": bool(condition), "check": message})
    if not condition:
        raise RuntimeError(message)
    print(f"[ok] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    import voyager

    require(torch.cuda.is_available(), "CUDA is available")
    require(
        torch.cuda.get_device_properties(0).total_memory >= 23 * 2**30,
        "GPU has at least 23 GiB VRAM",
    )
    require(Path(voyager.__file__).exists(), "Voyager Python package imports")

    mineflayer = ROOT / "third_party/voyager/voyager/env/mineflayer"
    require((mineflayer / "node_modules/mineflayer").is_dir(), "Mineflayer is installed")
    require((mineflayer / "package-lock.json").is_file(), "Mineflayer lockfile exists")
    package = json.loads((mineflayer / "package.json").read_text())
    require(package["dependencies"].get("mineflayer") == "4.14.0", "Mineflayer 4.14.0 crafting fix is pinned")
    node = subprocess.run(
        [str(mineflayer / "node_modules/.bin/node"), "--version"],
        capture_output=True, text=True, check=True,
    )
    require(node.stdout.strip() == "v18.20.8", "local Node 18.20.8 runtime is pinned")

    minecraft = ROOT / "runtime/minecraft"
    require((minecraft / "versions/1.19/1.19.json").is_file(), "Minecraft 1.19 is installed")
    require(
        (minecraft / "versions/fabric-loader-0.14.18-1.19").is_dir(),
        "Fabric loader 0.14.18 for Minecraft 1.19 is installed",
    )
    mods = list((minecraft / "mods").glob("*.jar"))
    require(len(mods) == 5, "five Voyager Fabric mods are installed")

    server = ROOT / "runtime/server-1.19"
    server_jar = server / "server.jar"
    require(server_jar.is_file(), "local Minecraft 1.19 dedicated server exists")
    digest = hashlib.sha1(server_jar.read_bytes()).hexdigest()
    require(digest == "e00c4052dac1d59a1188b2aa9d5a87113aaf1122", "server jar matches Mojang SHA-1")
    properties = (server / "server.properties").read_text()
    require("server-ip=127.0.0.1" in properties, "offline server is loopback-only")
    require("online-mode=false" in properties, "local protocol requires no Microsoft login")

    model_path = ROOT / "models/Qwen3-VL-8B-Instruct"
    shards = list(model_path.glob("model-*-of-*.safetensors"))
    require(len(shards) == 4, "all four Qwen3-VL model shards exist")
    config = AutoConfig.from_pretrained(model_path)
    require(config.model_type == "qwen3_vl", "Qwen3-VL config loads")
    require(config.text_config.num_hidden_layers == 36, "Qwen text stack has 36 layers")

    dataset = ROOT / "data/MineExplorer-Benchmark"
    main_rows = (dataset / "benchmark.jsonl").read_text().splitlines()
    hard_rows = (dataset / "benchmark_hard.jsonl").read_text().splitlines()
    require(len(main_rows) == 813, "MineExplorer main split has 813 scenarios")
    require(len(hard_rows) == 100, "MineExplorer hard split has 100 scenarios")
    required_fields = {
        "scene_id", "mode", "task_text", "scene_name", "commands", "milestones"
    }
    sample = json.loads(main_rows[0])
    require(required_fields <= sample.keys(), "MineExplorer schema has required fields")

    java = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, check=True
    )
    require("17." in java.stderr, "Java 17 is active")
    try:
        health = requests.get("http://127.0.0.1:3000/health", timeout=2).json()
    except requests.RequestException:
        health = None
    require(health is not None and health.get("status") == "ok", "Voyager bridge health endpoint responds")

    smoke = ROOT / "results/local_qwen_action_retry2.json"
    smoke_record = json.loads(smoke.read_text()) if smoke.is_file() else {}
    require(smoke_record.get("parser_valid") is True, "local Qwen smoke action is parser-valid")
    require(smoke_record.get("post_inventory", {}).get("oak_log", 0) >= 4, "local Qwen action passed oak-log environment verifier")
    suite = ROOT / "results/local_smoke_10.json"
    suite_record = json.loads(suite.read_text()) if suite.is_file() else {}
    require(suite_record.get("parser_valid") == 10, "10-scene BF16 parser smoke passed")
    require(suite_record.get("successes") == 10, "10-scene Minecraft verifier smoke passed")
    qlora = ROOT / "results/qlora_smoke.json"
    qlora_record = json.loads(qlora.read_text()) if qlora.is_file() else {}
    require(qlora_record.get("steps") == 1 and bool(qlora_record.get("losses")), "real-rollout QLoRA backward/update smoke passed")
    from acquisition.pool import ImmutableTrajectoryPool
    acquisition_pool = ImmutableTrajectoryPool(ROOT / "data/acquisition/local_t1_k4")
    try:
        acquisition_pool.verify()
        pool_valid = True
    except RuntimeError:
        pool_valid = False
    require(pool_valid, "real K=4 acquisition pool is finalized and checksum-valid")

    report = {
        "gate": 0,
        "protocol": "local-qwen-voyager-v1",
        "passed": all(check["ok"] for check in CHECKS),
        "checks": CHECKS,
        "peam_difference": "No Azure GPT-4o slow tier; local Qwen is the rollout sampler.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
