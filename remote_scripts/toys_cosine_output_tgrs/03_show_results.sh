#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_file "${OUTPUT_ROOT}/analysis.json"
require_file "${OUTPUT_ROOT}/target_layer_metrics.csv"
require_file "${OUTPUT_ROOT}/layer_summary.csv"
require_file "${OUTPUT_ROOT}/target_summary.csv"
require_file "${OUTPUT_ROOT}/output_cosine_heatmap.png"
require_file "${OUTPUT_ROOT}/output_angle_heatmap.png"
require_file "${OUTPUT_ROOT}/residual_norm_vs_output_angle.png"
require_file "${OUTPUT_ROOT}/embedding_angle_vs_output_angle.png"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/analysis.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as analysis_file:
    analysis = json.load(analysis_file)
summary = analysis["overall_summary"]
print("Targets:", ", ".join(analysis["target_concepts"]))
print("to_v layers:", analysis["layer_count"])
print(f"Mean output cosine: {summary['output_cosine_mean']:.6f}")
print(f"Mean output angle: {summary['output_angle_degrees_mean']:.3f} degrees")
print(
    "Output angle range: "
    f"{summary['output_angle_degrees_min']:.3f}-"
    f"{summary['output_angle_degrees_max']:.3f} degrees"
)
print("Artifacts:", sys.argv[1].rsplit("/", 1)[0])
PY
