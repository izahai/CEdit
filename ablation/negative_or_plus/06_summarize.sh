#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
NEGATIVE_CONFIG="$(train_config_for_method negative_projection)"
POSITIVE_CONFIG="$(train_config_for_method positive_projection)"
mkdir -p "${GCD_OUTPUT_DIR}"
require_file "${NEGATIVE_CONFIG}"
require_file "${POSITIVE_CONFIG}"

"${PYTHON_BIN}" - \
    "${GCD_OUTPUT_DIR}" "${SUMMARY_CSV}" "${METHODS_RAW}" \
    "${BENCHMARK_NAMES_RAW}" "${REPO_ROOT}" \
    "${NEGATIVE_CONFIG}" "${POSITIVE_CONFIG}" <<'PY'
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml


def target_count(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return len(dict.fromkeys(
            row["concept"].strip()
            for row in csv.DictReader(csv_file)
            if row.get("type") == "erase" and row.get("concept", "").strip()
        ))


def training_metadata(config, count):
    return {
        "anchor_mode": config["anchor_mode"],
        "target_projection_direction": config["target_projection_direction"],
        "anchor_concepts": ", ".join(config.get("anchor_concepts", [])),
        "params": config["params"],
        "requested_residual_rank": config["residual_rank"],
        "target_global_residual_count": count * (count + 1),
        "aug_num": config["aug_num"],
        "threshold": config["threshold"],
        "retain_scale": config["retain_scale"],
    }


def summarize(path, method, split, benchmark_name, config, repo_root):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    if "p_celebrity_correct" not in frame:
        raise SystemExit(f"Missing p_celebrity_correct column: {path}")
    raw = frame["p_celebrity_correct"].astype(str)
    detected = raw.ne("N")
    scores = pd.to_numeric(raw.where(detected), errors="coerce").fillna(0.0)
    correct = detected & scores.gt(0)
    count = target_count(repo_root / "data" / f"{benchmark_name}.csv")
    row = {
        "benchmark": benchmark_name,
        "model": method,
        "split": split,
        "target_concept_count": count,
    }
    row.update(training_metadata(config, count))
    row.update({
        "n_images": len(frame),
        "n_faces_detected": int(detected.sum()),
        "n_identity_correct": int(correct.sum()),
        "face_detection_rate": float(detected.mean()),
        "conditional_accuracy_CE_Eval": (
            float(correct.sum() / detected.sum()) if detected.any() else 0.0
        ),
        "identity_hit_rate": float(correct.mean()),
        "mean_matched_top1_probability": float(scores.mean()),
    })
    return row


(
    output_dir,
    output_csv,
    methods_raw,
    benchmarks_raw,
    repo_root,
    negative_config_path,
    positive_config_path,
) = sys.argv[1:]
repo_root = Path(repo_root)
configs = {}
for method, config_path in (
    ("negative_projection", negative_config_path),
    ("positive_projection", positive_config_path),
):
    with open(config_path, encoding="utf-8") as config_file:
        configs[method] = yaml.safe_load(config_file)

methods = methods_raw.split()
benchmarks = benchmarks_raw.split()
rows = []
for method in methods:
    for benchmark_name in benchmarks:
        run_dir = Path(output_dir) / method / benchmark_name
        for split in ("erase", "retain"):
            path = run_dir / f"{method}_{split}.csv"
            if not path.is_file():
                raise SystemExit(f"Missing evaluation CSV: {path}")
            rows.append(
                summarize(
                    path,
                    method,
                    split,
                    benchmark_name,
                    configs[method],
                    repo_root,
                )
            )

expected_rows = len(methods) * len(benchmarks) * 2
if len(rows) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} summary rows, found {len(rows)}")
summary = pd.DataFrame(rows)
summary.to_csv(output_csv, index=False)
print(summary.to_string(index=False))
print(f"Saved: {output_csv}")
PY
