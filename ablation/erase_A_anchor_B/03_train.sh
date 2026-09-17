#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

if [[ -f "${CHECKPOINT_PATH}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    printf 'Checkpoint exists; skipping: %s\n' "${CHECKPOINT_PATH}"
    exit 0
fi

mkdir -p "${CHECKPOINT_DIR}"
USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" -u train_erase_null.py \
    --config "${WORKFLOW_DIR}/train_config.yaml" \
    --target_concepts "${TARGET_A}" \
    --anchor_concepts "${ANCHOR_B}" \
    --retain_path "${RETAIN_CSV}" \
    --save_path "${CHECKPOINT_DIR}"
require_file "${CHECKPOINT_PATH}"

