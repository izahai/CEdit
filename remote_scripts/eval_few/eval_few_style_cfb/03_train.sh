#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}/train"

for method in "${EDITED_METHODS[@]}"; do
    train_config="$(train_config_for_method "${method}")"
    require_file "${train_config}"
    while IFS="${FIELD_SEPARATOR}" read -r \
        task_id erase_type target_concepts anchor_concept \
        contents prompt_templates prompt_count expected_count; do
        checkpoint_path="$(checkpoint_path_for_run "${method}" "${task_id}")"
        checkpoint_dir="$(dirname -- "${checkpoint_path}")"
        if [[ -f "${checkpoint_path}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
            printf '%s/%s checkpoint exists; skipping: %s\n' \
                "${method}" "${task_id}" "${checkpoint_path}"
            continue
        fi

        mkdir -p "${checkpoint_dir}"
        train_args=(
            --config "${train_config}"
            --sd_ckpt "${SD_CKPT}"
            --target_concepts "${target_concepts}"
            --anchor_concepts "${anchor_concept}"
            --retain_path "data/${erase_type}.csv"
            --save_path "${checkpoint_dir}"
        )
        if [[ "${method}" == "legacy" ]]; then
            entrypoint="train_erase_null.py"
        else
            entrypoint="train_closed_form_backprop.py"
        fi

        printf 'Training %s/%s: targets=[%s], anchor=[%s]\n' \
            "${method}" "${task_id}" "${target_concepts}" "${anchor_concept}"
        USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u "${entrypoint}" "${train_args[@]}" \
            2>&1 | tee "${LOG_ROOT}/train/${method}_${task_id}.log"
        require_file "${checkpoint_path}"
    done < <(task_rows)
done

printf 'Few-concept training complete.\n'
