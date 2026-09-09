# MetaPlastic Gate Report

Date: 2026-08-28

## Scope

This is the first inexpensive gate requested before online 16-loop or
5h/24h/72h runs. It replays real Qwen3-VL-8B MineExplorer rollouts with their
environment-derived MSR, but it is **not** a fresh online TSR evaluation.

Implemented:

- signed group-relative policy loss from K=4 environment scores;
- all text-transformer LoRA modules are candidates, with no layer/skill router;
- compact per-module SVD and global top-B projection;
- fixed total plasticity budget B=32;
- inactive A probes retained so pruned modules can receive future gradients;
- optional retention term gamma;
- disjoint two-scene replay probe at loops 0..8;
- fixed rank-32 LoRA comparison on exactly the same groups.

## Results

| Method | Loop-0 probe objective | Loop-8 | Structural replacements | Peak GPU |
|---|---:|---:|---:|---:|
| MetaPlastic hard top-B, gamma=1.0 | 0.00317 | 0.00089 | 0 after loop 1 | 13.38 GiB |
| MetaPlastic hard top-B, gamma=0.9 | 0.00317 | 0.00224 | 0 after loop 1 | 13.38 GiB |
| Fixed rank-32 LoRA | 0.00317 | -0.10432 | n/a | 13.69 GiB |

Lower group-relative NLL is better. The fixed rank-32 baseline has much more
total rank than global-B MetaPlastic, so this table is a diagnostic rather than
a capacity-matched final comparison.

## Gate decision

The current hard-projection algorithm **fails the structural-evolution gate**.
It selects 32 MLP gate/up directions on loop 1, then performs no layer-level
grow/prune replacements through loop 8. Gamma=0.9 does not fix the lock-in.
The controller/ES, online 16-loop run, and 5h/24h/72h study are therefore not
started: they would optimize a structural mechanism that is not yet evolving.

The implementation does establish that the real 8B/NF4 gradient and global
budget projection fit comfortably on a 24 GB GPU. It does **not** establish a
TSR/MSR improvement claim.

## Next falsifiable gate

Replace post-hoc hard projection with an exploration-capable budget mechanism
(for example, gradient-space candidate atoms plus age/usage-neutral global
competition), then require both:

1. non-zero grow/prune events across heterogeneous contexts; and
2. a capacity-matched fixed-allocation B=32 baseline on fresh held-out
   MineExplorer rollouts.

Only after both conditions pass should the learned eta/gamma controller and ES
outer loop be enabled.
