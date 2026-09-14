#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

EVALUATOR="${WORKFLOW_DIR}/evaluate_clip_fid.py"
require_file "${EVALUATOR}"
mkdir -p "${METRICS_DIR}" "${FID_CACHE_ROOT}"

args=(
    --config "${WORKFLOW_CONFIG}"
    --repo-root "${REPO_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --metrics-dir "${METRICS_DIR}"
    --fid-cache-root "${FID_CACHE_ROOT}"
    --num-samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}"
    --mscoco-num-prompts "${MSCOCO_NUM_PROMPTS}"
    --seed "${SEED}"
    --inference-timesteps "${INFERENCE_TIMESTEPS}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --clip-batch-size "${CLIP_BATCH_SIZE}"
    --fid-batch-size "${FID_BATCH_SIZE}"
    --fid-feature-layer "${FID_FEATURE_LAYER}"
    --clip-model "${CLIP_MODEL}"
    --device cuda
)
if [[ "${FORCE_EVAL:-0}" == "1" ]]; then
    args+=(--force)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${EVALUATOR}" "${args[@]}"

require_file "${METRICS_DIR}/detailed_metrics.csv"
require_file "${METRICS_DIR}/summary.csv"
require_file "${METRICS_DIR}/comparison.csv"
printf 'Evaluation complete: %s\n' "${METRICS_DIR}"

