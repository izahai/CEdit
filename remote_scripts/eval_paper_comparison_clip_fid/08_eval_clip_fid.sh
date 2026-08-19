#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

EVALUATOR="${WORKFLOW_DIR}/evaluate_mscoco_clip_fid.py"
OUTPUT_CSV="${CLIP_FID_OUTPUT_DIR}/summary.csv"
require_file "${EVALUATOR}"
require_file "${REPO_ROOT}/data/mscoco.csv"
mkdir -p "${CLIP_FID_OUTPUT_DIR}"

if [[ -f "${OUTPUT_CSV}" && "${FORCE_EVAL:-0}" != "1" ]]; then
    printf 'MS-COCO CLIP/FID evaluation exists; skipping: %s\n' "${OUTPUT_CSV}"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${EVALUATOR}" \
    --mscoco-csv "${REPO_ROOT}/data/mscoco.csv" \
    --image-root "${MSCOCO_IMAGE_ROOT}" \
    --output-csv "${OUTPUT_CSV}" \
    --methods "${METHODS[@]}" \
    --benchmarks "${BENCHMARK_NAMES[@]}" \
    --num-prompts "${MSCOCO_NUM_PROMPTS}" \
    --batch-size "${CLIP_BATCH_SIZE}" \
    --clip-model "${CLIP_MODEL}" \
    --device cuda

require_file "${OUTPUT_CSV}"
printf 'MS-COCO CLIP/FID evaluation complete: %s\n' "${OUTPUT_CSV}"
