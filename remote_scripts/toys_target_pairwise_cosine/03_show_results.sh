#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for artifact in \
    analysis.json \
    labels.csv \
    pairwise_metrics.csv \
    space_summary.csv \
    embedding_pairwise_cosine_heatmap.png \
    layer_pairwise_cosine_heatmaps.png \
    layer_pair_type_summary.png; do
    require_file "${OUTPUT_ROOT}/${artifact}"
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/space_summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as summary_file:
    rows = list(csv.DictReader(summary_file))
embedding = rows[0]
layers = rows[1:]
print("Embedding target-target mean cosine:", embedding["target_target_cosine_mean"])
for key, label in (
    ("target_target_cosine_mean", "Layer target-target mean range"),
    ("target_person_cosine_mean", "Layer target-person mean range"),
    ("target_empty_cosine_mean", "Layer target-empty mean range"),
    ("target_correlation_to_embedding", "Correlation-to-embedding range"),
):
    values = [float(row[key]) for row in layers]
    print(f"{label}: {min(values):.6f}-{max(values):.6f}")
print("Artifacts:", sys.argv[1].rsplit("/", 1)[0])
PY
