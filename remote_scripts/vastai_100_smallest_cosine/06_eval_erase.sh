#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

IMAGE_DIR="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/erase/edit"
CSV_PATH="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_erase.csv"
XLSX_PATH="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_erase.xlsx"
LOG_PATH="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_erase.log"
require_directory "${IMAGE_DIR}"

if [[ -f "${CSV_PATH}" && "${FORCE_EVAL:-0}" != "1" ]]; then
    printf 'Erase evaluation exists; skipping: %s\n' "${CSV_PATH}"
    exit 0
fi

mkdir -p "${GCD_OUTPUT_DIR}"
CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss" \
EVALUATE_SCRIPT="${CE_EVAL_ROOT}/eval/evaluate_by_GCD.py" \
GCD_USE_CUDA="${GCD_USE_CUDA}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
bash "${CE_EVAL_ROOT}/run/run_gcd_evaluation_colab.sh" \
    "${IMAGE_DIR}" "${XLSX_PATH}" "${CSV_PATH}" >"${LOG_PATH}" 2>&1

require_file "${CSV_PATH}"
printf 'Erase evaluation complete: %s\n' "${CSV_PATH}"
