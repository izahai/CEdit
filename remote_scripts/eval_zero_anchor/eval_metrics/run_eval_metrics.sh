#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd -- "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/concat_images}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/eval_metrics}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=========================================================="
echo " Running Quick CLIP Metrics: Art Anchor vs Zero Anchor"
echo " Working directory: ${REPO_ROOT}"
echo " GPU:               ${GPU_ID}"
echo " Python binary:     ${PYTHON_BIN}"
echo " Input images:      ${IMAGE_DIR}"
echo " Output directory:  ${OUTPUT_DIR}"
echo "=========================================================="

DEVICE="${DEVICE:-cuda}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/compute_quick_metrics.py" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --style_csv "${REPO_ROOT}/data/style.csv" \
    --device "${DEVICE}"

echo ""
echo "=========================================================="
echo " Evaluation Complete!"
echo " Summary table:  ${OUTPUT_DIR}/summary_table.txt"
echo " JSON summary:   ${OUTPUT_DIR}/summary_metrics.json"
echo " Per-image CSV:  ${OUTPUT_DIR}/per_image_metrics.csv"
echo "=========================================================="
