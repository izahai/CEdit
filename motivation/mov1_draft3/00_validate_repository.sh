#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/train_erase_null.py"
require_file "${REPO_ROOT}/sample2.py"
require_file "${REPO_ROOT}/data/${BENCHMARK_NAME}.csv"
require_file "${REPO_ROOT}/src/residual_subspace.py"
require_file "${WORKFLOW_CONFIG}"
require_file "${TRAIN_CONFIG}"
require_file "${WORKFLOW_DIR}/analyze_residuals.py"
require_file "${WORKFLOW_DIR}/summarize_results.py"
require_file "${WORKFLOW_DIR}/plot_figure.py"

"${PYTHON_BIN}" - "${WORKFLOW_CONFIG}" "${TRAIN_CONFIG}" <<'PY'
import sys
import yaml

workflow_path, train_path = sys.argv[1:]
with open(workflow_path, encoding="utf-8") as file:
    workflow = yaml.safe_load(file)
with open(train_path, encoding="utf-8") as file:
    train = yaml.safe_load(file)

expected_methods = ["legacy_full", "legacy_svd_rank30", "tgprs_rank30"]
expected_scales = [0.2, 0.4, 0.6, 0.8, 1.0]
if workflow["experiment"]["methods"] != expected_methods:
    raise SystemExit("MOV1 method list has changed")
if workflow["experiment"]["residual_scales"] != expected_scales:
    raise SystemExit("MOV1 residual-scale sweep has changed")
if workflow["experiment"]["benchmark_name"] != "100_celebrity":
    raise SystemExit("MOV1 must use the 100_celebrity benchmark")
expected_train = {
    "anchor_concepts": ["person"],
    "aug_num": 0,
    "threshold": 1e-4,
    "retain_scale": 0.05,
    "lamb": 0.0,
    "disable_filter": True,
    "params": "V",
}
for key, value in expected_train.items():
    if train.get(key) != value:
        raise SystemExit(f"Expected {key}={value!r}, found {train.get(key)!r}")
PY

target_concepts >/dev/null
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
printf 'MOV1 repository and configuration validation complete.\n'

