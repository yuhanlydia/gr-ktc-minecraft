import torch

from gr_ktc.conditional_lora import (
    conditional_reachability_score,
    fit_conditional_lora_basis,
    fit_contextual_lora_basis,
)
from gr_ktc.future_distillation import backward_future_targets, future_kl_loss, future_privileged_target
from gr_ktc.latent_population import LatentPopulation, PopulationItem
from gr_ktc.reachability import fit_individual_and_shared, rank_scaling, split_state_correction


def test_reachability_reports_individual_above_shared_for_incompatible_targets():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(12, 6, generator=g)
    w1 = torch.randn(4, 6, generator=g)
    w2 = torch.randn(4, 6, generator=g)
    result = fit_individual_and_shared([(x, x @ w1.T), (x, x @ w2.T)], rank=4, ridge=1e-7)
    assert result["rho_individual_mean"] > result["rho_shared"]


def test_rank_curve_is_monotonic_up_to_numerical_tolerance():
    x = torch.eye(4)
    y = x.clone()
    curve = rank_scaling([(x, y)], ranks=(1, 2, 4), ridge=1e-8)
    assert curve[0]["rho_shared"] <= curve[1]["rho_shared"] + 1e-5
    assert curve[1]["rho_shared"] <= curve[2]["rho_shared"] + 1e-5


def test_state_correction_split_reconstructs_target():
    x = torch.eye(3)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    reachable, unreachable, _ = split_state_correction(x, target, rank=2, ridge=1e-8)
    assert torch.allclose(reachable + unreachable, target, atol=1e-5)


def test_conditional_basis_uses_continuous_coordinates():
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
    features = torch.eye(4, 3)
    targets = torch.randn(4, 2)
    basis = fit_conditional_lora_basis(coords, features, targets, num_bases=2, rank=1)
    coefficients = basis.coefficients(torch.tensor([0.05, 0.0]))
    assert coefficients.shape == (1, 2)
    assert torch.isclose(coefficients.sum(), torch.tensor(1.0))


def test_future_target_and_kl_are_finite():
    current = torch.zeros(2, 3)
    future = torch.ones(2, 4, 3)
    target = future_privileged_target(current, future)
    assert target.shape == current.shape
    assert future_kl_loss(torch.randn(2, 4, 5), torch.randn(2, 4, 5)).isfinite()
    backward = backward_future_targets(torch.arange(12.0).reshape(4, 3), horizon=2)
    assert backward.shape == (4, 3)
    assert torch.equal(backward[-1], torch.tensor([9.0, 10.0, 11.0]))


def test_contextual_basis_supports_variable_token_counts():
    coordinates = torch.tensor([[0.0], [10.0]])
    contexts = [
        (torch.randn(3, 4), torch.randn(3, 2)),
        (torch.randn(5, 4), torch.randn(5, 2)),
    ]
    basis = fit_contextual_lora_basis(
        coordinates, contexts, num_bases=2, rank=2, temperature=0.01
    )
    score = conditional_reachability_score(basis, coordinates, contexts)
    assert 0 <= score <= 1


def test_population_retrieval_and_evolution_are_bounded():
    population = LatentPopulation(max_size=2)
    for i in range(4):
        population.add(PopulationItem(torch.ones(3, 2) * i, float(i), str(i)))
    assert len(population) == 2
    retrieved = population.retrieve(torch.ones(2), top_k=1)
    assert retrieved.shape == (3, 2)
    population.evolve([], elite_fraction=0.5)
    assert len(population) <= 2
