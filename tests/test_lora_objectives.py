import torch

from gr_ktc.lora_objectives import (
    anchor_kl,
    dpo_loss,
    grta_loss,
    negative_subspace_penalty,
    weighted_causal_lm_loss,
)


def test_weighted_bc_ignores_zero_weight_example():
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[0, 1, 2], [0, 3, 3]])
    first_only = weighted_causal_lm_loss(logits, labels, torch.tensor([1.0, 0.0]))
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(first_only, expected)


def test_dpo_improves_when_student_margin_is_larger():
    weak = dpo_loss(
        torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0,
    )
    strong = dpo_loss(
        torch.tensor([2.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0,
    )
    assert strong < weak


def test_trajectory_and_negative_subspaces_are_separated():
    positive = torch.tensor([[1.0], [0.0]])
    negative = torch.tensor([[0.0], [1.0]])
    teacher = torch.tensor([[[1.0, 0.0]]])
    aligned = torch.tensor([[[2.0, 0.0]]])
    assert grta_loss(aligned, teacher, positive) < 1e-6
    assert negative_subspace_penalty(aligned, negative) == 0


def test_negative_penalty_is_scale_invariant_energy_fraction():
    negative = torch.tensor([[0.0], [1.0]])
    effect = torch.tensor([[[1.0, 1.0]]])
    first = negative_subspace_penalty(effect, negative)
    second = negative_subspace_penalty(effect * 100, negative)
    assert torch.allclose(first, torch.tensor(0.5))
    assert torch.allclose(first, second)


def test_anchor_kl_is_zero_for_identical_logits():
    logits = torch.randn(2, 3, 5)
    assert torch.allclose(anchor_kl(logits, logits), torch.tensor(0.0), atol=1e-6)
