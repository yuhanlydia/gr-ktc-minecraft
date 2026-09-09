from __future__ import annotations

import torch


def fit_low_rank_delta(
    x: torch.Tensor,
    delta_y: torch.Tensor,
    rank: int,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual ridge solution followed by truncated SVD in PEFT A/B convention."""
    if x.ndim != 2 or delta_y.ndim != 2 or x.shape[0] != delta_y.shape[0]:
        raise ValueError("x and delta_y need matching [samples, features] shapes")
    if rank <= 0:
        raise ValueError("rank must be positive")
    compute_dtype = torch.float64 if x.device.type == "cpu" else torch.float32
    x_work = x.to(compute_dtype)
    y_work = delta_y.to(compute_dtype)
    gram = x_work @ x_work.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    dual = torch.linalg.solve(gram + ridge * identity, y_work)
    # ``x.T @ dual`` is at most rank ``num_samples``.  Real Qwen hidden
    # features are 4096-dimensional but a context only contains tens of action
    # tokens; materializing and decomposing a 4096x4096 matrix is therefore
    # wasteful.  QR-factor both thin matrices and SVD only their small core.
    if x_work.shape[0] < min(x_work.shape[1], y_work.shape[1]):
        q_left, r_left = torch.linalg.qr(x_work.T, mode="reduced")
        q_right, r_right = torch.linalg.qr(dual.T, mode="reduced")
        core = r_left @ r_right.T
        u_core, singular_values, vh_core = torch.linalg.svd(
            core, full_matrices=False
        )
        u = q_left @ u_core
        vh = vh_core @ q_right.T
    else:
        delta_w_in_out = x_work.T @ dual
        u, singular_values, vh = torch.linalg.svd(
            delta_w_in_out, full_matrices=False
        )
    actual_rank = min(rank, singular_values.numel())
    sqrt_s = singular_values[:actual_rank].clamp_min(0).sqrt()
    a = sqrt_s[:, None] * u[:, :actual_rank].T
    b = vh[:actual_rank].T * sqrt_s[None, :]
    return a.to(x.dtype), b.to(x.dtype)
