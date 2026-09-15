#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

TARGET_CONCEPTS="$(target_concepts)"
for method in "${METHODS[@]}"; do
    anchor_mode="$(anchor_mode_for_method "${method}")"
    for scale in "${RESIDUAL_SCALES[@]}"; do
        checkpoint_dir="$(checkpoint_dir_for_run "${method}" "${scale}")"
        checkpoint_path="${checkpoint_dir}/weight.pt"
        log_path="${LOG_DIR}/train_${method}_scale_$(scale_slug "${scale}").log"
        checkpoint_stale=0
        for input_path in \
            "${WORKFLOW_CONFIG}" \
            "${TRAIN_CONFIG}" \
            "${REPO_ROOT}/train_erase_null.py" \
            "${REPO_ROOT}/src/residual_subspace.py"; do
            if [[ ! -f "${checkpoint_path}" || "${input_path}" -nt "${checkpoint_path}" ]]; then
                checkpoint_stale=1
            fi
        done
        if [[ -f "${checkpoint_path}" && "${checkpoint_stale}" == "0" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
            printf 'Checkpoint exists; skipping: %s\n' "${checkpoint_path}"
            continue
        fi
        mkdir -p "${checkpoint_dir}" "${LOG_DIR}"
        printf 'Training method=%s scale=%s\n' "${method}" "${scale}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_TF=0 TRANSFORMERS_NO_TF=1 \
        "${PYTHON_BIN}" -u train_erase_null.py \
            --config "${TRAIN_CONFIG}" \
            --anchor_mode "${anchor_mode}" \
            --residual_rank "${RESIDUAL_RANK}" \
            --residual_scale "${scale}" \
            --target_concepts "${TARGET_CONCEPTS}" \
            --retain_path "data/${BENCHMARK_NAME}.csv" \
            --save_path "${checkpoint_dir}" \
            --file_name weight \
            2>&1 | tee "${log_path}"
        require_file "${checkpoint_path}"
    done
done
