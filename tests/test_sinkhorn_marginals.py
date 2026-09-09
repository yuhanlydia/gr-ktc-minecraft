import torch

from gr_ktc.merge_c_sinkhorn import sinkhorn_cost, sinkhorn_phase_barycenter


def test_sinkhorn_cost_is_finite_and_nonnegative():
    cost = sinkhorn_cost(torch.randn(4, 3), torch.randn(5, 3), iterations=10)
    assert torch.isfinite(cost)
    assert cost >= 0


def test_phase_barycenter_shape():
    result = sinkhorn_phase_barycenter(
        [torch.randn(8, 3), torch.randn(9, 3)],
        torch.tensor([0.5, 0.5]),
        phases=2,
        support_points=2,
        steps=1,
    )
    assert result.shape == (2, 2, 3)

