#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
TRAIN_CONFIG="${WORKFLOW_DIR}/train_config.yaml"
mkdir -p "${GCD_OUTPUT_DIR}"
require_file "${TRAIN_CONFIG}"

"${PYTHON_BIN}" - "${GCD_OUTPUT_DIR}" "${SUMMARY_CSV}" "${ANCHOR_MODE}" \
    "${TRAIN_CONFIG}" "${BENCHMARK_NAMES_RAW}" "${REPO_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml


def concept_count(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return len(dict.fromkeys(row["concept"] for row in csv.DictReader(csv_file)))


def summarize(path, method, split, benchmark_name, train_config, repo_root):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    if "p_celebrity_correct" not in frame:
        raise SystemExit(f"Missing p_celebrity_correct column: {path}")
    raw = frame["p_celebrity_correct"].astype(str)
    detected = raw.ne("N")
    scores = pd.to_numeric(raw.where(detected), errors="coerce").fillna(0.0)
    correct = detected & scores.gt(0)
    global_concept_count = concept_count(
        repo_root / "data" / f"{benchmark_name}.csv"
    )
    requested_rank = train_config["residual_rank"]
    return {
        "benchmark": benchmark_name,
        "model": method,
        "requested_residual_rank": requested_rank,
        "basis_rank": requested_rank,
        "global_concept_count": global_concept_count,
        "global_residual_count": global_concept_count * (global_concept_count + 1),
        "aug_num": train_config["aug_num"],
        "threshold": train_config["threshold"],
        "retain_scale": train_config["retain_scale"],
        "split": split,
        "n_images": len(frame),
        "n_faces_detected": int(detected.sum()),
        "n_identity_correct": int(correct.sum()),
        "face_detection_rate": float(detected.mean()),
        "conditional_accuracy_CE_Eval": (
            float(correct.sum() / detected.sum()) if detected.any() else 0.0
        ),
        "identity_hit_rate": float(correct.mean()),
        "mean_matched_top1_probability": float(scores.mean()),
    }


output_dir, output_csv, method, train_config_path, benchmarks, repo_root = sys.argv[1:]
repo_root = Path(repo_root)
with open(train_config_path, encoding="utf-8") as config_file:
    train_config = yaml.safe_load(config_file)
rows = []
for benchmark_name in benchmarks.split():
    run_dir = Path(output_dir) / benchmark_name
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
                train_config,
                repo_root,
            )
        )

summary = pd.DataFrame(rows)
summary.to_csv(output_csv, index=False)
print(summary.to_string(index=False))
print(f"Saved: {output_csv}")
PY
