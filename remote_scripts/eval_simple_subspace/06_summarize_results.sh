#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
mkdir -p "${GCD_OUTPUT_DIR}"

"${PYTHON_BIN}" - "${GCD_OUTPUT_DIR}" "${SUMMARY_CSV}" "${ANCHOR_MODE}" \
    "${WORKFLOW_DIR}" "${CONFIG_NAMES_RAW}" "${BENCHMARK_NAMES_RAW}" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import yaml


def summarize(path, method, split, config_name, benchmark_name, train_config):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    if "p_celebrity_correct" not in frame:
        raise SystemExit(f"Missing p_celebrity_correct column: {path}")
    raw = frame["p_celebrity_correct"].astype(str)
    detected = raw.ne("N")
    scores = pd.to_numeric(raw.where(detected), errors="coerce").fillna(0.0)
    correct = detected & scores.gt(0)
    target_count = int(benchmark_name.removesuffix("_celebrity"))
    requested_rank = train_config["residual_rank"]
    return {
        "config": config_name,
        "benchmark": benchmark_name,
        "model": method,
        "requested_residual_rank": requested_rank,
        "effective_residual_rank": min(requested_rank, target_count),
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


output_dir, output_csv, method, workflow_dir, config_names, benchmarks = sys.argv[1:]
rows = []
for config_name in config_names.split():
    config_path = Path(workflow_dir) / f"train_{config_name}.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        train_config = yaml.safe_load(config_file)
    for benchmark_name in benchmarks.split():
        run_dir = Path(output_dir) / config_name / benchmark_name
        for split in ("erase", "retain"):
            path = run_dir / f"{method}_{split}.csv"
            if not path.is_file():
                raise SystemExit(f"Missing evaluation CSV: {path}")
            rows.append(
                summarize(
                    path,
                    method,
                    split,
                    config_name,
                    benchmark_name,
                    train_config,
                )
            )

summary = pd.DataFrame(rows)
summary.to_csv(output_csv, index=False)
print(summary.to_string(index=False))
print(f"Saved: {output_csv}")
PY
