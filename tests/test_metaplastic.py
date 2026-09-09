import torch

from gr_ktc.metaplastic import (
    PlasticFactors,
    group_relative_policy_loss,
    project_global_rank_budget,
)


def test_invalid_retention_is_rejected():
    from gr_ktc.metaplastic import decay_peft_plastic_weights
    import pytest
    with pytest.raises(ValueError):
        decay_peft_plastic_weights(torch.nn.Linear(2, 2), 1.1)


def test_global_budget_moves_rank_to_strongest_layers():
    modules = {
        "early": PlasticFactors(torch.eye(2), torch.diag(torch.tensor([5.0, 1.0]))),
        "late": PlasticFactors(torch.eye(2), torch.diag(torch.tensor([4.0, 3.0]))),
    }
    projected, report = project_global_rank_budget(modules, budget=2)
    assert report.active_rank == 2
    assert report.ranks == {"early": 1, "late": 1}
    assert torch.linalg.matrix_rank(projected["early"].b @ projected["early"].a) == 1
    assert torch.linalg.matrix_rank(projected["late"].b @ projected["late"].a) == 1
    # Inactive probes remain capable of receiving a B gradient and regrowing.
    assert projected["early"].a[1].norm() > 0
    assert projected["early"].b[:, 1].norm() == 0


def test_projection_preserves_all_directions_when_budget_is_large():
    torch.manual_seed(3)
    original = PlasticFactors(torch.randn(2, 5), torch.randn(4, 2), 0.5)
    projected, report = project_global_rank_budget({"m": original}, budget=2)
    before = original.scale * original.b @ original.a
    after = projected["m"].b @ projected["m"].a
    torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)
    assert report.retained_energy > 0.999


def test_group_relative_loss_uses_negative_rollouts_as_repulsion():
    labels = torch.tensor([[0, 1], [0, 1]])
    advantages = torch.tensor([1.0, -1.0])
    logits = torch.zeros(2, 2, 2, requires_grad=True)
    loss = group_relative_policy_loss(logits, labels, advantages)
    loss.backward()
    # The preferred rollout increases token-1 logit; rejected does the opposite.
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[1, 0, 1] > 0


def test_zero_variance_advantage_produces_zero_update():
    logits = torch.randn(4, 3, 5, requires_grad=True)
    labels = torch.ones(4, 3, dtype=torch.long)
    loss = group_relative_policy_loss(logits, labels, torch.zeros(4))
    assert float(loss.detach()) == 0.0
