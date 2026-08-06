#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

ERASE_CSV="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_erase.csv"
RETAIN_CSV="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_retain.csv"
SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
require_file "${ERASE_CSV}"
require_file "${RETAIN_CSV}"

"${PYTHON_BIN}" - "${ERASE_CSV}" "${RETAIN_CSV}" "${SUMMARY_CSV}" "${ANCHOR_MODE}" <<'PY'
import sys
import pandas as pd


def summarize(path, method, split):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    raw = frame["p_celebrity_correct"].astype(str)
    detected = raw.ne("N")
    scores = pd.to_numeric(raw.where(detected), errors="coerce").fillna(0.0)
    correct = detected & scores.gt(0)
    return {
        "model": method,
        "split": split,
        "n_images": len(frame),
        "n_faces_detected": int(detected.sum()),
        "n_identity_correct": int(correct.sum()),
        "face_detection_rate": float(detected.mean()),
        "conditional_accuracy_CE_Eval": float(correct.sum() / detected.sum()) if detected.any() else 0.0,
        "identity_hit_rate": float(correct.mean()),
        "mean_matched_top1_probability": float(scores.mean()),
    }


erase_csv, retain_csv, output_csv, method = sys.argv[1:]
summary = pd.DataFrame([
    summarize(erase_csv, method, "erase"),
    summarize(retain_csv, method, "retain"),
])
summary.to_csv(output_csv, index=False)
print(summary.to_string(index=False))
print(f"Saved: {output_csv}")
PY
