#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}/anchor"

manifest="${LEARNED_ANCHOR_ROOT}/manifest.json"
tensors="${LEARNED_ANCHOR_ROOT}/embeddings.safetensors"
if [[ -f "${manifest}" && -f "${tensors}" && "${FORCE_RELEARN:-0}" != "1" ]]; then
    printf 'Complete learned-anchor artifact exists; skipping: %s\n' \
        "${LEARNED_ANCHOR_ROOT}"
    exit 0
fi
if [[ -e "${LEARNED_ANCHOR_ROOT}" ]]; then
    if [[ "${FORCE_RELEARN:-0}" != "1" ]]; then
        printf 'Anchor artifact is incomplete. Inspect or set FORCE_RELEARN=1: %s\n' \
            "${LEARNED_ANCHOR_ROOT}" >&2
        exit 1
    fi
    backup="${LEARNED_ANCHOR_ROOT}.backup.$(date +%Y%m%d-%H%M%S)"
    mv "${LEARNED_ANCHOR_ROOT}" "${backup}"
    printf 'Moved previous anchor artifact to %s\n' "${backup}"
fi

USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" -u learn_anchor.py \
        --config "${WORKFLOW_DIR}/learn_anchor_config.yaml" \
        --save_root "${LEARNED_ANCHOR_ROOT}" \
        2>&1 | tee "${LOG_ROOT}/anchor/van_gogh.log"
require_file "${manifest}"
require_file "${tensors}"
printf 'Learned-anchor search complete: %s\n' "${LEARNED_ANCHOR_ROOT}"
