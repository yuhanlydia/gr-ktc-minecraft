from importlib.metadata import metadata
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_train_extra_installs_qwen3_vl_processor_runtime():
    requirements = metadata("gr-ktc-minecraft").get_all("Requires-Dist") or []

    assert any(
        requirement.startswith("torchvision")
        and 'extra == "train"' in requirement
        for requirement in requirements
    ), "the train extra must install torchvision for AutoProcessor"


def test_miniwob_extra_and_bootstrap_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = project["project"]["optional-dependencies"]["miniwob"]
    assert any(item.startswith("gymnasium") for item in requirements)
    assert any(item.startswith("playwright") for item in requirements)
    bootstrap = ROOT / "scripts/bootstrap_miniwob.sh"
    assert bootstrap.stat().st_mode & 0o111
    text = bootstrap.read_text()
    assert "9e779f087de9a65668b6974d11f9ce9816026e96" in text
    assert "7fd85d71a4b60325c6585396ec4f48377d049838" in text
    assert "uv pip install --python" in text
    assert 'playwright install-deps chromium' in text


def test_miniwob_docs_state_raw_reward_and_k10_protocol():
    docs = (ROOT / "docs/LATENTSKILL_MINIWOB_EXPERIMENT.md").read_text()
    assert "RAW_REWARD_GLOBAL" in docs
    assert ">= 0.5" in docs
    assert "K=10" in docs
    readme = (ROOT / "README.md").read_text()
    assert "run_miniwob_latentskill_gate.py" in readme
