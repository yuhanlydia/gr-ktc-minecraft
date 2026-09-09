import torch

from gr_ktc.generation import GeneratedKVTrajectory, extract_generated_kv
from gr_ktc.model_loader import choose_24gb_precision


def test_extract_generated_kv_slices_prompt_and_flattens_heads():
    key = torch.arange(1 * 2 * 6 * 3).reshape(1, 2, 6, 3).float()
    value = key + 100
    result = extract_generated_kv(
        ((key, value),),
        layer_ids=[0],
        prompt_tokens=4,
        generated_tokens=2,
    )
    assert result[0].shape == (2, 12)
    expected_k = key[0, :, 4, :].reshape(-1)
    assert torch.equal(result[0][0, :6], expected_k)


def test_precision_policy_uses_highest_practical_mode():
    assert choose_24gb_precision("latent_pilot", prompt_tokens=1024, max_new_tokens=256) == "bf16"
    assert choose_24gb_precision("acquisition", prompt_tokens=3000, max_new_tokens=512) == "int8"
    assert choose_24gb_precision("long_context", prompt_tokens=8192, max_new_tokens=2048) == "nf4"
    assert choose_24gb_precision("training", prompt_tokens=512, max_new_tokens=512) == "nf4"


def test_trajectory_dataclass_distinguishes_generated_and_cached_tokens():
    result = GeneratedKVTrajectory(
        sequences=torch.tensor([[1, 2, 3, 4]]),
        all_generated_token_ids=torch.tensor([[3, 4]]),
        trajectory_token_ids=torch.tensor([[3]]),
        kv_by_layer={0: torch.zeros(1, 8)},
        prompt_tokens=2,
    )
    assert result.all_generated_token_ids.shape[-1] == 2
    assert result.trajectory_token_ids.shape[-1] == 1
