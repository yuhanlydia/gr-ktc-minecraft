import torch

from gr_ktc.merge_b_softdtw import merge_b, soft_dtw


def test_soft_dtw_is_differentiable():
    x = torch.randn(4, 3, requires_grad=True)
    y = torch.randn(5, 3)
    loss = soft_dtw(x, y)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_merge_b_returns_phase_residuals():
    sequences = [torch.randn(5, 3), torch.randn(7, 3)]
    result = merge_b(
        sequences, torch.tensor([1.0, -1.0]), phases=4, steps=1
    )
    assert result.shape == (3, 3)

