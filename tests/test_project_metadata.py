from importlib.metadata import metadata


def test_train_extra_installs_qwen3_vl_processor_runtime():
    requirements = metadata("gr-ktc-minecraft").get_all("Requires-Dist") or []

    assert any(
        requirement.startswith("torchvision")
        and 'extra == "train"' in requirement
        for requirement in requirements
    ), "the train extra must install torchvision for AutoProcessor"
