#!/usr/bin/env python3
"""Combine MOV1 GCD outputs with residual-rank diagnostics."""

import argparse
import csv
import math
from pathlib import Path

import yaml


METHOD_LABELS = {
    "legacy_full": "Legacy full-rank",
    "legacy_svd_rank30": "Legacy-SVD rank 30",
    "tgprs_rank30": "TGPRS rank 30",
}


def scale_slug(value):
    return str(value).replace(".", "p")


def wilson_interval(successes, total, confidence_z=1.959963984540054):
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    probability = successes / total
    z2 = confidence_z * confidence_z
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = (
        confidence_z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_gcd(path, expected_count):
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows in {path}, found {len(rows)}")
    if not rows or "p_celebrity_correct" not in rows[0]:
        raise ValueError(f"Missing p_celebrity_correct column: {path}")
    detected = 0
    hits = 0
    probability_sum = 0.0
    for row in rows:
        raw = (row.get("p_celebrity_correct") or "").strip()
        if raw == "N" or not raw:
            continue
        try:
            probability = float(raw)
        except ValueError as error:
            raise ValueError(f"Invalid p_celebrity_correct value {raw!r}: {path}") from error
        detected += 1
        probability_sum += probability
        if probability > 0:
            hits += 1
    low, high = wilson_interval(hits, len(rows))
    return {
        "n_images": len(rows),
        "n_faces_detected": detected,
        "n_identity_hits": hits,
        "face_detection_rate": detected / len(rows),
        "identity_hit_rate": hits / len(rows),
        "identity_hit_ci_low": low,
        "identity_hit_ci_high": high,
        "mean_matched_top1_probability": probability_sum / len(rows),
    }


def load_rank_summary(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return {row["method"]: row for row in rows}


def build_rows(config, rank_summary, gcd_root, checkpoint_root):
    experiment = config["experiment"]
    expected_count = int(experiment["expected_images_per_split"])
    rows = []
    for method in experiment["methods"]:
        if method not in rank_summary:
            raise ValueError(f"Rank summary is missing method {method}")
        ranks = rank_summary[method]
        for scale in experiment["residual_scales"]:
            slug = scale_slug(scale)
            run_dir = gcd_root / method / f"scale_{slug}"
            checkpoint_path = (
                checkpoint_root / method / f"scale_{slug}" / "weight.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            erase = summarize_gcd(run_dir / "erase.csv", expected_count)
            retain = summarize_gcd(run_dir / "retain.csv", expected_count)
            erasure_success = 1.0 - erase["identity_hit_rate"]
            rows.append({
                "benchmark": experiment.get("benchmark_name", "100_celebrity"),
                "target_concept_count": int(
                    str(experiment.get("benchmark_name", "100_celebrity")).split("_")[0]
                ),
                "anchor_concept": experiment.get("anchor_concept", "person"),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "residual_scale": float(scale),
                "requested_residual_rank": (
                    "" if method == "legacy_full" else experiment["residual_rank"]
                ),
                "residual_numerical_rank": ranks["residual_numerical_rank"],
                "residual_stable_rank": ranks["residual_stable_rank"],
                "residual_spectral_effective_rank": ranks[
                    "residual_spectral_effective_rank"
                ],
                "edit_statistic_numerical_rank": ranks[
                    "edit_statistic_numerical_rank"
                ],
                "edit_statistic_stable_rank": ranks["edit_statistic_stable_rank"],
                "inference_timesteps": experiment.get("inference_timesteps", 50),
                "erase_n_images": erase["n_images"],
                "erase_n_faces_detected": erase["n_faces_detected"],
                "erase_n_identity_hits": erase["n_identity_hits"],
                "erase_face_detection_rate": erase["face_detection_rate"],
                "erase_identity_hit_rate": erase["identity_hit_rate"],
                "erase_identity_hit_ci_low": erase["identity_hit_ci_low"],
                "erase_identity_hit_ci_high": erase["identity_hit_ci_high"],
                "erasure_success": erasure_success,
                "erasure_success_ci_low": 1.0 - erase["identity_hit_ci_high"],
                "erasure_success_ci_high": 1.0 - erase["identity_hit_ci_low"],
                "erase_mean_matched_top1_probability": erase[
                    "mean_matched_top1_probability"
                ],
                "retain_n_images": retain["n_images"],
                "retain_n_faces_detected": retain["n_faces_detected"],
                "retain_n_identity_hits": retain["n_identity_hits"],
                "retain_face_detection_rate": retain["face_detection_rate"],
                "retain_identity_hit_rate": retain["identity_hit_rate"],
                "retain_identity_hit_ci_low": retain["identity_hit_ci_low"],
                "retain_identity_hit_ci_high": retain["identity_hit_ci_high"],
                "retain_mean_matched_top1_probability": retain[
                    "mean_matched_top1_probability"
                ],
                "checkpoint_path": str(checkpoint_path),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank-summary", type=Path, required=True)
    parser.add_argument("--gcd-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    ranks = load_rank_summary(args.rank_summary)
    rows = build_rows(config, ranks, args.gcd_root, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} MOV1 rows: {args.output}")


if __name__ == "__main__":
    main()
