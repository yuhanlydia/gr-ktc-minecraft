from __future__ import annotations

from dataclasses import dataclass

import torch

from .merge_b_softdtw import linear_resample


@dataclass(frozen=True)
class DiscreteBridge:
    """Empirical Schrödinger bridge under a Brownian reference kernel."""

    source: torch.Tensor
    target: torch.Tensor
    source_phase: torch.Tensor
    target_phase: torch.Tensor
    coupling: torch.Tensor
    source_marginal: torch.Tensor
    target_marginal: torch.Tensor
    barycentric_target: torch.Tensor
    trust: float = 1.0

    @property
    def control(self) -> torch.Tensor:
        return self.barycentric_target - self.source

    def conditional_mean(self, time: float) -> torch.Tensor:
        if not 0.0 <= time <= 1.0:
            raise ValueError("time must be in [0, 1]")
        return (1.0 - time) * self.source + time * self.barycentric_target

    def stochastic_interpolant(
        self, time: float, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample Brownian-bridge interpolants around the conditional mean."""
        mean = self.conditional_mean(time)
        if time in (0.0, 1.0):
            return mean
        variance = time * (1.0 - time)
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
        )
        return mean + variance**0.5 * noise


def _empirical_support(
    sequences: list[torch.Tensor], weights: torch.Tensor, phases: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences or len(sequences) != weights.numel():
        raise ValueError("sequence and weight counts must match and be non-empty")
    weights = weights.to(device=sequences[0].device, dtype=sequences[0].dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    support = torch.cat([linear_resample(sequence, phases) for sequence in sequences])
    phase = torch.arange(phases, device=support.device).repeat(len(sequences))
    marginal = weights.repeat_interleave(phases) / phases
    return support, phase, marginal


def _log_sinkhorn_coupling(
    cost: torch.Tensor,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    log_kernel = -cost / epsilon
    log_a = source_marginal.clamp_min(1e-30).log()
    log_b = target_marginal.clamp_min(1e-30).log()
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    transport = torch.exp(log_u[:, None] + log_kernel + log_v[None])
    # A short probability-space projection removes the residual marginal error
    # left by finite log-domain iterations. The kernel is already stabilized,
    # so these normalizations do not suffer the original underflow problem.
    # Nearly deterministic phase kernels converge slowly; cap at 2k cheap
    # matrix scalings and stop once both marginals meet experiment tolerance.
    for step in range(2000):
        transport = transport * (
            source_marginal / transport.sum(dim=1).clamp_min(1e-30)
        )[:, None]
        transport = transport * (
            target_marginal / transport.sum(dim=0).clamp_min(1e-30)
        )[None, :]
        if step % 20 == 19:
            row_error = (transport.sum(dim=1) - source_marginal).abs().max()
            col_error = (transport.sum(dim=0) - target_marginal).abs().max()
            if max(float(row_error), float(col_error)) < 1e-7:
                break
    return transport


def fit_discrete_schrodinger_bridge(
    source_sequences: list[torch.Tensor],
    target_sequences: list[torch.Tensor],
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    phases: int = 16,
    epsilon: float = 0.1,
    temporal_cost: float = 1.0,
    iterations: int = 100,
    trust: float = 1.0,
    max_control_norm: float | None = None,
) -> DiscreteBridge:
    """Fit a low-dimensional empirical bridge without training another network.

    Entropic OT is the static Schrödinger problem for a Gibbs/Brownian reference
    kernel.  The phase penalty prevents semantically implausible cross-time
    couplings; the barycentric projection is used as deterministic KV memory.
    """
    source, source_phase, source_marginal = _empirical_support(
        source_sequences, source_weights, phases
    )
    target, target_phase, target_marginal = _empirical_support(
        target_sequences, target_weights, phases
    )
    feature_cost = torch.cdist(source, target).square()
    phase_scale = max(phases - 1, 1)
    phase_cost = (
        (source_phase[:, None] - target_phase[None, :]).to(source.dtype)
        / phase_scale
    ).square()
    coupling = _log_sinkhorn_coupling(
        feature_cost + temporal_cost * phase_cost,
        source_marginal,
        target_marginal,
        epsilon,
        iterations,
    )
    if not 0.0 <= trust <= 1.0:
        raise ValueError("trust must be in [0, 1]")
    conditional = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-30)
    hard_target = conditional @ target
    control = trust * (hard_target - source)
    if max_control_norm is not None:
        if max_control_norm <= 0:
            raise ValueError("max_control_norm must be positive")
        norm = control.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        control = control * (max_control_norm / norm).clamp_max(1.0)
    barycentric_target = source + control
    return DiscreteBridge(
        source,
        target,
        source_phase,
        target_phase,
        coupling,
        source_marginal,
        target_marginal,
        barycentric_target,
        trust,
    )


def empirical_bridge_trust(
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    advantage_margin: float = 1.0,
    prior_strength: float = 4.0,
) -> float:
    """Shrink tiny empirical bridges toward identity using endpoint evidence."""
    if prior_strength < 0 or advantage_margin < 0:
        raise ValueError("prior_strength and advantage_margin must be nonnegative")
    source = source_weights / source_weights.sum().clamp_min(1e-12)
    target = target_weights / target_weights.sum().clamp_min(1e-12)
    source_ess = 1.0 / source.square().sum().clamp_min(1e-12)
    target_ess = 1.0 / target.square().sum().clamp_min(1e-12)
    joint_ess = 2.0 / (source_ess.reciprocal() + target_ess.reciprocal())
    evidence = float(joint_ess) * advantage_margin
    return evidence / (evidence + prior_strength) if evidence > 0 else 0.0


def merge_schrodinger_bridge(
    sequences: list[torch.Tensor],
    advantages: torch.Tensor,
    *,
    phases: int = 16,
    epsilon: float = 0.1,
    temporal_cost: float = 1.0,
    temperature: float = 1.0,
    adaptive_trust: bool = True,
    prior_strength: float = 4.0,
    max_control_norm: float | None = None,
) -> tuple[torch.Tensor, DiscreteBridge]:
    """Return a phase-residual memory and its auditable transport bridge."""
    positive = advantages > 0
    negative = advantages < 0
    if not positive.any() or not negative.any():
        raise ValueError("Schrodinger bridge requires positive and negative rollouts")
    sources = [s for s, keep in zip(sequences, negative, strict=True) if bool(keep)]
    targets = [s for s, keep in zip(sequences, positive, strict=True) if bool(keep)]
    source_weights = torch.softmax(-advantages[negative] / temperature, dim=0)
    target_weights = torch.softmax(advantages[positive] / temperature, dim=0)
    margin = float(advantages[positive].mean() - advantages[negative].mean())
    trust = (
        empirical_bridge_trust(
            source_weights,
            target_weights,
            advantage_margin=max(margin, 0.0),
            prior_strength=prior_strength,
        )
        if adaptive_trust
        else 1.0
    )
    bridge = fit_discrete_schrodinger_bridge(
        sources,
        targets,
        source_weights,
        target_weights,
        phases=phases,
        epsilon=epsilon,
        temporal_cost=temporal_cost,
        trust=trust,
        max_control_norm=max_control_norm,
    )
    phase_target = torch.stack(
        [
            (bridge.barycentric_target[bridge.source_phase == phase]
             * bridge.source_marginal[bridge.source_phase == phase, None]).sum(dim=0)
            / bridge.source_marginal[bridge.source_phase == phase].sum().clamp_min(1e-30)
            for phase in range(phases)
        ]
    )
    return phase_target[1:] - phase_target[:-1], bridge
