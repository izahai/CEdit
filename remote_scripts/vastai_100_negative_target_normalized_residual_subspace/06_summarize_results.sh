#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${GCD_OUTPUT_DIR}/summary.csv"
mkdir -p "${GCD_OUTPUT_DIR}"

"${PYTHON_BIN}" - "${GCD_OUTPUT_DIR}" "${SUMMARY_CSV}" "${ANCHOR_MODE}" <<'PY'
import sys
from pathlib import Path

import pandas as pd


def summarize(path, method, split, rank):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    raw = frame["p_celebrity_correct"].astype(str)
    detected = raw.ne("N")
    scores = pd.to_numeric(raw.where(detected), errors="coerce").fillna(0.0)
    correct = detected & scores.gt(0)
    return {
        "model": method,
        "subspace_rank": rank,
        "split": split,
        "n_images": len(frame),
        "n_faces_detected": int(detected.sum()),
        "n_identity_correct": int(correct.sum()),
        "face_detection_rate": float(detected.mean()),
        "conditional_accuracy_CE_Eval": float(correct.sum() / detected.sum()) if detected.any() else 0.0,
        "identity_hit_rate": float(correct.mean()),
        "mean_matched_top1_probability": float(scores.mean()),
    }


output_dir, output_csv, method = sys.argv[1:]
rows = []
for rank in range(10, 101, 10):
    rank_dir = Path(output_dir) / f"k_{rank}"
    for split in ("erase", "retain"):
        path = rank_dir / f"{method}_{split}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing evaluation CSV: {path}")
        rows.append(summarize(path, method, split, rank))

summary = pd.DataFrame(rows)
summary.to_csv(output_csv, index=False)
print(summary.to_string(index=False))
print(f"Saved: {output_csv}")
PY
