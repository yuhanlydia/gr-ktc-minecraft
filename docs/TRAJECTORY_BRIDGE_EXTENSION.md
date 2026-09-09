# Trajectory bridge extension (experimental Method D)

## Decision

Schrödinger bridges, stochastic interpolants, optimal transport, and stochastic
control are useful here, but they are different views of one extension rather
than four independent modules.

The implemented `gr_ktc.schrodinger_bridge` fits an empirical entropic bridge in
the learned low-dimensional correctness subspace. Negative-advantage rollouts
form the source marginal and positive-advantage rollouts form the target
marginal. Squared latent distance plus a phase mismatch cost defines the
Brownian/Gibbs reference kernel. Sinkhorn scaling solves the discrete static
Schrödinger problem; its barycentric projection becomes deterministic fast KV
memory.

This adds no neural network and is therefore practical on a 24 GB RTX 3090. It
also exposes the coupling, marginal error, and expected control energy for
mechanism analysis.

## Relationship among the four concepts

- Optimal transport gives the zero-entropy endpoint-coupling idealization.
- The Schrödinger bridge adds entropy/KL regularization relative to stochastic
  reference dynamics, which is preferable with only four noisy rollouts.
- A stochastic interpolant samples intermediate latent states around the
  bridge's conditional mean. The implementation has exact endpoints and a
  Brownian variance schedule `t(1-t)`.
- Stochastic control interprets the bridge displacement as a minimum-KL control
  action. Its weighted squared norm is logged as control energy; this can later
  regularize fast steering and LoRA alignment.

For the actual K=4 regime the implementation now uses a soft-constrained
bridge. The transport displacement is shrunk toward the identity according to
the harmonic-mean effective sample size and the positive/negative advantage
margin. An optional per-token control-norm cap prevents one poorly matched
latent from dominating the KV prefix. `adaptive_trust=False` recovers the
original hard empirical bridge.

## What it can improve

Compared with Method C's independent token-cloud barycenter at every phase,
Method D explicitly transports low-quality trajectories toward superior ones
and penalizes cross-phase matches. This could help long-horizon tasks when
failure and success share partial structure. It is not expected to help groups
without both positive and negative rollouts, and with K=4 its coupling can still
have high variance.

## Preregistered evaluation

Method D remains experimental until it passes the same gates as A/B/C:

1. leave-one-group-out fit only;
2. next-retry rescue above no-memory and failed-prefix controls;
3. parser validity degradation no more than 2 percentage points;
4. memory-free LoRA above shared BC+DPO;
5. report success and control energy against epsilon and temporal-cost sweeps.

Pilot values: phases `{8,16}`, epsilon `{0.1,0.2,0.5}`, temporal cost
`{0.5,1,2}`. No value is selected using the held-out PEAM task-seed trials.

## Primary references

- Albergo & Vanden-Eijnden, *Building Normalizing Flows with Stochastic
  Interpolants*, arXiv:2209.15571.
- Chen, Georgiou & Pavon, *On the Relation Between Optimal Transport and
  Schrödinger Bridges: A Stochastic Control Viewpoint*, arXiv:1412.4430.
- Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*,
  NeurIPS 2013.
- De Bortoli et al., *Diffusion Schrödinger Bridge with Applications to
  Score-Based Generative Modeling*, NeurIPS 2021 / arXiv:2106.01357.

## Claim boundary

The implementation is a discrete empirical/static Schrödinger bridge followed
by Brownian conditional interpolation. It is not a learned continuous-time
Schrödinger bridge, diffusion model, or proof of globally optimal control in the
full Qwen KV space.
