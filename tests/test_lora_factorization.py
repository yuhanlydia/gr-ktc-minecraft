import torch

from gr_ktc.lora_closed_form import fit_low_rank_delta


def test_full_rank_factorization_fits_linear_teacher_effect():
    torch.manual_seed(0)
    x = torch.randn(20, 4)
    target = torch.randn(4, 3)
    delta_y = x @ target
    a, b = fit_low_rank_delta(x, delta_y, rank=3, ridge=1e-8)
    predicted = x @ (b @ a).T
    relative_error = (predicted - delta_y).norm() / delta_y.norm()
    assert relative_error < 1e-4


def test_compact_dual_svd_handles_wide_hidden_features():
    torch.manual_seed(1)
    x = torch.randn(6, 64)
    target = torch.randn(64, 48)
    delta_y = x @ target
    a, b = fit_low_rank_delta(x, delta_y, rank=6, ridge=1e-8)
    predicted = x @ (b @ a).T
    relative_error = (predicted - delta_y).norm() / delta_y.norm()
    assert relative_error < 1e-4
