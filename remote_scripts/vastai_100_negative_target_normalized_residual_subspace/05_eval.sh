#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${CE_EVAL_ROOT}/celeb-detection-oss"
mkdir -p "${GCD_OUTPUT_DIR}"

for subspace_k in "${K_VALUES[@]}"; do
    image_root="$(image_root_for_k "${subspace_k}")"
    gcd_output_dir="$(gcd_output_dir_for_k "${subspace_k}")"
    mkdir -p "${gcd_output_dir}"
    for split in erase retain; do
        image_dir="${image_root}/${BENCHMARK_NAME}/${split}/edit"
        csv_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.csv"
        xlsx_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.xlsx"
        log_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.log"
        require_directory "${image_dir}"

        if [[ -f "${csv_path}" && "${FORCE_EVAL:-0}" != "1" ]]; then
            printf 'k=%s %s evaluation exists; skipping: %s\n' "${subspace_k}" "${split}" "${csv_path}"
            continue
        fi

        CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss" \
        EVALUATE_SCRIPT="${CE_EVAL_ROOT}/eval/evaluate_by_GCD.py" \
        GCD_USE_CUDA="${GCD_USE_CUDA}" \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        bash "${CE_EVAL_ROOT}/run/run_gcd_evaluation_colab.sh" \
            "${image_dir}" "${xlsx_path}" "${csv_path}" >"${log_path}" 2>&1

        require_file "${csv_path}"
        printf 'k=%s %s evaluation complete: %s\n' "${subspace_k}" "${split}" "${csv_path}"
    done
done
