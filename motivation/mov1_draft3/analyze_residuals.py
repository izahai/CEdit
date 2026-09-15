#!/usr/bin/env python3
"""Measure residual and edit-statistic spectra for the MOV1 methods."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from diffusers import StableDiffusionPipeline

from src.residual_subspace import (
    build_norm_matched_truncated_svd_residuals,
    build_target_global_pairwise_residual_subspace_residuals,
)
from train_erase_null import encode_last_subject_embeddings


METHOD_LABELS = {
    "legacy_full": "Legacy full-rank",
    "legacy_svd_rank30": "Legacy-SVD rank 30",
    "tgprs_rank30": "TGPRS rank 30",
}


def load_target_concepts(path, expected_count=100):
    concepts = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            concept = (row.get("concept") or "").strip()
            if row.get("type") == "erase" and concept and concept not in concepts:
                concepts.append(concept)
    if len(concepts) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique erase concepts in {path}, "
            f"found {len(concepts)}"
        )
    return concepts


def spectral_summary(matrix):
    matrix = matrix.float()
    singular_values = torch.linalg.svdvals(matrix)
    if not singular_values.numel():
        raise ValueError("Cannot summarize an empty matrix")
    maximum = singular_values.max()
    tolerance = (
        max(matrix.shape)
        * torch.finfo(singular_values.dtype).eps
        * maximum
    )
    numerical_rank = int((singular_values > tolerance).sum().item())
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy > 0:
        probabilities = energy / total_energy
        nonzero = probabilities > 0
        effective_rank = math.exp(
            -(probabilities[nonzero] * probabilities[nonzero].log()).sum().item()
        )
        stable_rank = (total_energy / maximum.square()).item()
        normalized_energy = probabilities.tolist()
    else:
        effective_rank = 0.0
        stable_rank = 0.0
        normalized_energy = [0.0] * len(singular_values)
    cumulative = []
    running = 0.0
    for value in normalized_energy:
        running += value
        cumulative.append(running)
    return {
        "shape": list(matrix.shape),
        "numerical_rank": numerical_rank,
        "stable_rank": stable_rank,
        "spectral_effective_rank": effective_rank,
        "rank_tolerance": tolerance.item(),
        "singular_values": singular_values.tolist(),
        "normalized_energy": normalized_energy,
        "cumulative_energy": cumulative,
    }


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


@torch.no_grad()
def run(config, output_dir):
    experiment = config["experiment"]
    benchmark_path = REPO_ROOT / "data" / f"{experiment['benchmark_name']}.csv"
    concepts = load_target_concepts(benchmark_path)
    device = torch.device("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(experiment["sd_ckpt"]).to(
        device
    )
    targets = encode_last_subject_embeddings(
        pipeline,
        concepts,
        device=device,
        chunk_size=int(experiment["embedding_batch_size"]),
    )
    anchor = encode_last_subject_embeddings(
        pipeline,
        [experiment["anchor_concept"]],
        device=device,
        chunk_size=1,
    )
    anchors = anchor.expand_as(targets)
    extra_anchors = encode_last_subject_embeddings(
        pipeline,
        ["", experiment["anchor_concept"]],
        device=device,
        chunk_size=2,
    )
    rank = int(experiment["residual_rank"])
    legacy = anchors - targets
    legacy_svd, legacy_svd_diagnostics = (
        build_norm_matched_truncated_svd_residuals(targets, anchors, rank=rank)
    )
    tgprs, tgprs_diagnostics = (
        build_target_global_pairwise_residual_subspace_residuals(
            targets,
            anchors,
            extra_anchor_embeddings=extra_anchors,
            rank=rank,
        )
    )
    residuals_by_method = {
        "legacy_full": legacy,
        "legacy_svd_rank30": legacy_svd,
        "tgprs_rank30": tgprs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_rows = []
    spectrum_rows = []
    method_details = {}
    flattened_targets = targets.reshape(len(targets), -1).float()
    for method in experiment["methods"]:
        residual_matrix = residuals_by_method[method].reshape(len(targets), -1)
        edit_statistic = residual_matrix.float().T @ flattened_targets / len(targets)
        residual_summary = spectral_summary(residual_matrix)
        edit_summary = spectral_summary(edit_statistic)
        rank_rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "residual_numerical_rank": residual_summary["numerical_rank"],
            "residual_stable_rank": residual_summary["stable_rank"],
            "residual_spectral_effective_rank": residual_summary[
                "spectral_effective_rank"
            ],
            "edit_statistic_numerical_rank": edit_summary["numerical_rank"],
            "edit_statistic_stable_rank": edit_summary["stable_rank"],
            "edit_statistic_spectral_effective_rank": edit_summary[
                "spectral_effective_rank"
            ],
        })
        for index, (singular_value, energy, cumulative) in enumerate(
            zip(
                residual_summary["singular_values"],
                residual_summary["normalized_energy"],
                residual_summary["cumulative_energy"],
            ),
            start=1,
        ):
            spectrum_rows.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "component": index,
                "singular_value": singular_value,
                "normalized_energy": energy,
                "cumulative_energy": cumulative,
            })
        method_details[method] = {
            "residual": residual_summary,
            "edit_statistic": edit_summary,
        }

    with (output_dir / "rank_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)
    with (output_dir / "residual_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(spectrum_rows[0]))
        writer.writeheader()
        writer.writerows(spectrum_rows)
    analysis = {
        "hypothesis": (
            "rank(delta_W) <= rank(edit_statistic) <= rank(residual_matrix)"
        ),
        "checkpoint": experiment["sd_ckpt"],
        "benchmark": experiment["benchmark_name"],
        "target_count": len(concepts),
        "anchor": experiment["anchor_concept"],
        "requested_rank": rank,
        "methods": method_details,
        "legacy_svd_diagnostics": legacy_svd_diagnostics,
        "tgprs_diagnostics": tgprs_diagnostics,
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(analysis), file, indent=2)
        file.write("\n")
    print("Residual rank summary")
    for row in rank_rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
