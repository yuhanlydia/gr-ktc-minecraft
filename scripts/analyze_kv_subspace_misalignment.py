#!/usr/bin/env python3
"""Test whether native-KV subspace misalignment predicts LoRA interference."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.grassmann import (
    bootstrap_spearman_interval,
    consensus_spectrum,
    context_bootstrap_spearman_interval,
    grassmann_distance,
    latent_subspace,
    permutation_pvalue,
    principal_angles,
    spearman_correlation,
)
from gr_ktc.merge_b_softdtw import linear_resample
from gr_ktc.reachability import fit_reachability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memories", type=Path,
        default=ROOT / "results/fast_kv_four_context_memories.safetensors",
    )
    parser.add_argument(
        "--reachability", type=Path,
        default=ROOT / "results/real_reachability_four_contexts.safetensors",
    )
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--raw-trajectories", type=Path)
    parser.add_argument("--phase-tokens", type=int, default=32)
    parser.add_argument(
        "--replication-primary",
        choices=("key_grassmann", "value_grassmann", "joint_grassmann", "kv_mean_grassmann"),
        default="kv_mean_grassmann",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/kv_subspace_misalignment_four_contexts.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if (args.source is None) != (args.raw_trajectories is None):
        raise ValueError("source and raw-trajectories must be supplied together")

    raw = load_file(args.memories)
    raw_rollouts = load_file(args.raw_trajectories) if args.raw_trajectories else None
    source = json.loads(args.source.read_text()) if args.source else None
    reachability = load_file(args.reachability)
    scenes = sorted(
        key.removeprefix("features_")
        for key in reachability if key.startswith("features_")
    )
    if len(scenes) < 3:
        raise ValueError("at least three contexts are required")
    width = raw[f"scene_{scenes[0]}_positive_layer_{args.layer}"].shape[1] // 2
    bases = {"key": {}, "value": {}, "joint": {}}
    corrections = {}
    contexts = {}
    for scene in scenes:
        positive = raw[f"scene_{scene}_positive_layer_{args.layer}"].float()
        failed = raw[f"scene_{scene}_failed_layer_{args.layer}"].float()
        if raw_rollouts is None:
            correction = positive - failed
        else:
            advantages = torch.tensor(source["acquisitions"][scene]["advantages"])
            trajectories = [
                linear_resample(
                    raw_rollouts[f"scene_{scene}_rollout_{index}_layer_{args.layer}"].float(),
                    args.phase_tokens,
                )
                for index in range(len(advantages))
            ]
            failed_center = torch.stack([
                trajectory for trajectory, advantage in zip(
                    trajectories, advantages, strict=True,
                ) if float(advantage) < 0
            ]).mean(0)
            # Each successful rollout contributes its full phase-aligned
            # correction relative to the same failed barycenter.
            correction = torch.cat([
                trajectory - failed_center for trajectory, advantage in zip(
                    trajectories, advantages, strict=True,
                ) if float(advantage) > 0
            ])
        corrections[scene] = correction
        matrices = {
            "key": correction[:, :width],
            "value": correction[:, width:],
            "joint": correction,
        }
        for channel, matrix in matrices.items():
            bases[channel][scene] = latent_subspace(matrix, args.subspace_rank)
        contexts[scene] = (
            reachability[f"features_{scene}"],
            reachability[f"kv_effect_{scene}"],
        )

    individual_fits = {
        scene: fit_reachability(*contexts[scene], rank=args.lora_rank)
        for scene in scenes
    }
    pairs = []
    for first, second in itertools.combinations(scenes, 2):
        pair_contexts = [contexts[first], contexts[second]]
        shared_fit = fit_reachability(
            torch.cat([pair[0] for pair in pair_contexts]),
            torch.cat([pair[1] for pair in pair_contexts]),
            rank=args.lora_rank,
        )
        individual_mean = (
            individual_fits[first].rho + individual_fits[second].rho
        ) / 2
        shared = float(shared_fit.rho)
        record = {
            "first": first,
            "second": second,
            "rho_individual_mean": individual_mean,
            "rho_shared": shared,
            "interference": individual_mean - shared,
        }
        channel_distances = []
        for channel in ("key", "value", "joint"):
            angles = principal_angles(
                bases[channel][first], bases[channel][second],
            )
            distance = grassmann_distance(
                bases[channel][first], bases[channel][second],
            )
            record[f"{channel}_angles_rad"] = angles.tolist()
            record[f"{channel}_grassmann"] = distance
            if channel in ("key", "value"):
                channel_distances.append(distance)
        record["kv_mean_grassmann"] = sum(channel_distances) / 2
        pairs.append(record)

    interference = torch.tensor([x["interference"] for x in pairs])
    correlations = {}
    for metric in (
        "key_grassmann", "value_grassmann", "joint_grassmann",
        "kv_mean_grassmann",
    ):
        distances = torch.tensor([x[metric] for x in pairs])
        rho = spearman_correlation(distances, interference)
        p_value, permutations, exact = permutation_pvalue(
            distances, interference, samples=args.permutation_samples,
        )
        low, high = bootstrap_spearman_interval(
            distances, interference, samples=args.bootstrap_samples,
        )
        pair_values = {
            tuple(sorted((record["first"], record["second"]))): (
                record[metric], record["interference"],
            )
            for record in pairs
        }
        context_low, context_high, valid_bootstraps = context_bootstrap_spearman_interval(
            pair_values, scenes, samples=args.bootstrap_samples,
        )
        correlations[metric] = {
            "spearman": rho,
            "two_sided_permutation_p": p_value,
            "permutations": permutations,
            "permutation_exact": exact,
            "bootstrap_90ci_pair_resampling": [low, high],
            "bootstrap_95ci_context_resampling": [context_low, context_high],
            "valid_context_bootstraps": valid_bootstraps,
        }

    spectra = {}
    for channel in ("key", "value", "joint"):
        spectrum = consensus_spectrum([bases[channel][scene] for scene in scenes])
        spectra[channel] = {
            "nonzero_eigenvalues": spectrum.tolist(),
            "top_r_mean": float(spectrum[:args.subspace_rank].mean()),
            "trace": float(spectrum.sum()),
        }

    primary = correlations[args.replication_primary]
    replication = raw_rollouts is not None
    gate = (
        primary["spearman"] > (0.0 if replication else 0.5)
        and primary["two_sided_permutation_p"] <= (0.05 if replication else 0.10)
        and primary[
            "bootstrap_95ci_context_resampling" if replication
            else "bootstrap_90ci_pair_resampling"
        ][0] > 0
    )
    report = {
        "protocol": "cross-context-consolidation-misalignment-raw-replication-v2" if replication else "cross-context-consolidation-misalignment-pilot-v2",
        "scenes": scenes,
        "layer": args.layer,
        "subspace_rank": args.subspace_rank,
        "lora_rank": args.lora_rank,
        "native_correction": (
            "each phase-aligned successful raw KV trajectory - failed raw KV barycenter"
            if replication else "positive_KV_barycenter - failed_KV_barycenter"
        ),
        "basis_axis": "right singular vectors in latent feature space",
        "pairs": pairs,
        "correlations": correlations,
        "consensus_spectrum": spectra,
        "preregistered_primary": args.replication_primary,
        "gate_rule": (
            "positive Spearman, permutation p <= 0.05, context-bootstrap 95% lower bound > 0"
            if replication else
            "Spearman > 0.5, permutation p <= 0.10, pair-bootstrap 90% lower bound > 0"
        ),
        "gate_pass": gate,
        "next_action": (
            "implement transport/SMC after independent raw-trajectory replication"
            if gate else
            "stop Grassmann method branch; preregistered misalignment hypothesis did not replicate"
        ),
        "scope_warning": (
            "Eight-context raw-trajectory replication; pair observations remain dependent, so context-cluster bootstrap is required. Reachability is a residual-stream low-rank proxy."
            if replication else
            "Only four contexts/six dependent pairs and four-token merged supports. Pair bootstrap is descriptive; permutation is the primary finite-sample test."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
