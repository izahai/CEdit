#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
CLIP_FID_SUMMARY_CSV="${CLIP_FID_OUTPUT_DIR}/summary.csv"
LEGACY_CONFIG="$(train_config_for_method legacy)"
TARGET_GLOBAL_CONFIG="$(train_config_for_method target_global_pairwise_residual_subspace)"
mkdir -p "${GCD_OUTPUT_DIR}"
require_file "${LEGACY_CONFIG}"
require_file "${TARGET_GLOBAL_CONFIG}"
require_file "${CLIP_FID_SUMMARY_CSV}"

"${PYTHON_BIN}" - \
    "${GCD_OUTPUT_DIR}" "${SUMMARY_CSV}" "${METHODS_RAW}" \
    "${BENCHMARK_NAMES_RAW}" "${REPO_ROOT}" \
    "${LEGACY_CONFIG}" "${TARGET_GLOBAL_CONFIG}" \
    "${CLIP_FID_SUMMARY_CSV}" <<'PY'
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


def training_metadata(method, configs, count):
    if method == "original":
        return {
            "anchor_mode": "",
            "anchor_concepts": "",
            "params": "",
            "requested_residual_rank": "",
            "target_global_residual_count": "",
            "aug_num": "",
            "threshold": "",
            "retain_scale": "",
        }

    config = configs[method]
    is_target_global = method == "target_global_pairwise_residual_subspace"
    anchors = config.get("anchor_concepts", [])
    return {
        "anchor_mode": config["anchor_mode"],
        "anchor_concepts": ", ".join(anchors),
        "params": config["params"],
        "requested_residual_rank": (
            config.get("residual_rank", "") if is_target_global else ""
        ),
        "target_global_residual_count": count * (count + 1) if is_target_global else "",
        "aug_num": config["aug_num"],
        "threshold": config["threshold"],
        "retain_scale": config["retain_scale"],
    }


def summarize(
    path,
    method,
    split,
    benchmark_name,
    configs,
    repo_root,
    clip_fid_metrics,
):
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
    row.update(training_metadata(method, configs, count))
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
    row.update(clip_fid_metrics[(benchmark_name, method)])
    return row


(
    output_dir,
    output_csv,
    methods_raw,
    benchmarks_raw,
    repo_root,
    legacy_config_path,
    target_global_config_path,
    clip_fid_summary_path,
) = sys.argv[1:]
repo_root = Path(repo_root)
with open(legacy_config_path, encoding="utf-8") as config_file:
    legacy_config = yaml.safe_load(config_file)
with open(target_global_config_path, encoding="utf-8") as config_file:
    target_global_config = yaml.safe_load(config_file)
configs = {
    "legacy": legacy_config,
    "target_global_pairwise_residual_subspace": target_global_config,
}

clip_fid_frame = pd.read_csv(clip_fid_summary_path)
required_metric_columns = {
    "benchmark",
    "model",
    "mscoco_n_images",
    "mscoco_clip_score",
    "mscoco_fid_vs_original",
}
missing_metric_columns = required_metric_columns - set(clip_fid_frame.columns)
if missing_metric_columns:
    raise SystemExit(
        "Missing MS-COCO metric columns: "
        + ", ".join(sorted(missing_metric_columns))
    )
if clip_fid_frame.duplicated(["benchmark", "model"]).any():
    raise SystemExit("Duplicate benchmark/model rows in MS-COCO metric summary")
clip_fid_metrics = {
    (row.benchmark, row.model): {
        "mscoco_n_images": int(row.mscoco_n_images),
        "mscoco_clip_score": float(row.mscoco_clip_score),
        "mscoco_fid_vs_original": float(row.mscoco_fid_vs_original),
    }
    for row in clip_fid_frame.itertuples(index=False)
}

methods = methods_raw.split()
benchmarks = benchmarks_raw.split()
expected_metric_keys = {
    (benchmark, method) for method in methods for benchmark in benchmarks
}
if set(clip_fid_metrics) != expected_metric_keys:
    missing = sorted(expected_metric_keys - set(clip_fid_metrics))
    unexpected = sorted(set(clip_fid_metrics) - expected_metric_keys)
    raise SystemExit(
        f"MS-COCO metric rows do not match workflow; "
        f"missing={missing}, unexpected={unexpected}"
    )
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
                    configs,
                    repo_root,
                    clip_fid_metrics,
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
