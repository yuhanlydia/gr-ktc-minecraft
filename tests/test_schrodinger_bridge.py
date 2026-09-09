import torch

from gr_ktc.schrodinger_bridge import (
    empirical_bridge_trust,
    fit_discrete_schrodinger_bridge,
    merge_schrodinger_bridge,
)


def test_empirical_bridge_trust_shrinks_small_noisy_groups():
    one = torch.tensor([1.0])
    small = empirical_bridge_trust(one, one, advantage_margin=1.0, prior_strength=4.0)
    broad = empirical_bridge_trust(
        torch.ones(8), torch.ones(8), advantage_margin=1.0, prior_strength=4.0
    )
    assert 0 < small < broad < 1


def test_soft_bridge_control_is_shrunk_and_clipped():
    source = [torch.zeros(4, 2)]
    target = [torch.full((4, 2), 10.0)]
    bridge = fit_discrete_schrodinger_bridge(
        source, target, torch.ones(1), torch.ones(1), phases=4,
        trust=0.5, max_control_norm=1.0,
    )
    assert torch.all(bridge.control.norm(dim=-1) <= 1.00001)
    assert bridge.trust == 0.5


def test_bridge_coupling_matches_empirical_marginals():
    source = [torch.tensor([[0.0], [0.5], [1.0]])]
    target = [torch.tensor([[2.0], [2.5], [3.0]])]
    bridge = fit_discrete_schrodinger_bridge(
        source, target, torch.ones(1), torch.ones(1), phases=3,
        epsilon=0.2, temporal_cost=5.0, iterations=200,
    )
    assert torch.allclose(bridge.coupling.sum(1), bridge.source_marginal, atol=1e-4)
    assert torch.allclose(bridge.coupling.sum(0), bridge.target_marginal, atol=1e-4)
    assert torch.equal(bridge.conditional_mean(0), bridge.source)
    assert torch.equal(bridge.conditional_mean(1), bridge.barycentric_target)


def test_bridge_merge_moves_negative_trajectory_toward_positive():
    sequences = [
        torch.zeros(5, 2),
        torch.ones(5, 2) * 0.1,
        torch.arange(5).float()[:, None].repeat(1, 2),
        torch.arange(5).float()[:, None].repeat(1, 2) + 0.1,
    ]
    residual, bridge = merge_schrodinger_bridge(
        sequences, torch.tensor([-1.0, -1.0, 1.0, 1.0]), phases=5,
        epsilon=0.1, temporal_cost=10.0, adaptive_trust=False,
    )
    assert residual.shape == (4, 2)
    assert residual.mean() > 0.5
    assert bridge.control.mean() > 0


def test_stochastic_interpolant_has_exact_endpoints():
    bridge = fit_discrete_schrodinger_bridge(
        [torch.zeros(3, 2)], [torch.ones(3, 2)],
        torch.ones(1), torch.ones(1), phases=3,
    )
    assert torch.equal(bridge.stochastic_interpolant(0), bridge.source)
    assert torch.equal(bridge.stochastic_interpolant(1), bridge.barycentric_target)
