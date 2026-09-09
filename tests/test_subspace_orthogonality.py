import torch

from gr_ktc.correctness_subspace import fit_correctness_subspace


def test_positive_and_negative_subspaces_are_orthogonal():
    covariance = torch.diag(torch.tensor([3.0, 2.0, -1.0, -4.0]))
    result = fit_correctness_subspace(covariance, rank=2)
    assert result.positive.shape == (4, 2)
    assert result.negative.shape == (4, 2)
    assert torch.allclose(
        result.positive.T @ result.negative, torch.zeros(2, 2), atol=1e-6
    )

