#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}/train"
require_file "${LEARNED_ANCHOR_ROOT}/manifest.json"
require_file "${LEARNED_ANCHOR_ROOT}/embeddings.safetensors"

for method in "${EDITED_METHODS[@]}"; do
    train_config="$(method_train_config "${method}")"
    checkpoint="$(checkpoint_path "${method}")"
    checkpoint_dir="$(dirname -- "${checkpoint}")"
    dependency="${train_config}"
    if [[ "${method}" == "clip_guided_learned_anchor" ]]; then
        dependency="${LEARNED_ANCHOR_ROOT}/manifest.json"
    fi
    if [[ -f "${checkpoint}" \
        && "${checkpoint}" -nt "${dependency}" \
        && "${FORCE_RETRAIN:-0}" != "1" ]]; then
        printf '%s checkpoint exists; skipping: %s\n' "${method}" "${checkpoint}"
        continue
    fi
    mkdir -p "${checkpoint_dir}"
    args=(
        --config "${train_config}"
        --target_concepts "${TARGET_CONCEPT}"
        --retain_path "data/${ERASE_TYPE}.csv"
        --save_path "${checkpoint_dir}"
    )
    if [[ "${method}" == "clip_guided_learned_anchor" ]]; then
        args+=(--learned_anchor_path "${LEARNED_ANCHOR_ROOT}")
    else
        args+=(--anchor_concepts "${ANCHOR_CONCEPT}")
    fi
    printf 'Training %s for %s\n' "${method}" "${TARGET_CONCEPT}"
    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u train_erase_null.py "${args[@]}" \
        2>&1 | tee "${LOG_ROOT}/train/${method}_${TASK_ID}.log"
    require_file "${checkpoint}"
done
printf 'All edited checkpoints are complete.\n'
