import torch

from gr_ktc.group_advantage import group_relative_advantage


def test_mixed_binary_group_is_centered_and_scaled():
    result = group_relative_advantage(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_constant_group_has_no_relative_signal():
    assert torch.equal(
        group_relative_advantage(torch.ones(4)), torch.zeros(4)
    )

