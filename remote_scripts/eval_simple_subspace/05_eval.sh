#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${CE_EVAL_ROOT}/celeb-detection-oss"
mkdir -p "${GCD_OUTPUT_DIR}"
TF_CUDA_LIBRARY_PATH="$(tensorflow_cuda_library_path)"

for config_name in "${CONFIG_NAMES[@]}"; do
    for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
        image_root="$(image_root_for_run "${config_name}" "${benchmark_name}")"
        gcd_output_dir="$(gcd_output_dir_for_run "${config_name}" "${benchmark_name}")"
        mkdir -p "${gcd_output_dir}"
        for split in erase retain; do
            image_dir="${image_root}/${benchmark_name}/${split}/edit"
            csv_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.csv"
            xlsx_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.xlsx"
            log_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.log"
            require_directory "${image_dir}"

            if [[ -f "${csv_path}" && "${FORCE_EVAL:-0}" != "1" ]]; then
                printf '%s/%s %s evaluation exists; skipping: %s\n' \
                    "${config_name}" "${benchmark_name}" "${split}" "${csv_path}"
                continue
            fi

            CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss" \
            EVALUATE_SCRIPT="${CE_EVAL_ROOT}/eval/evaluate_by_GCD.py" \
            GCD_USE_CUDA="${GCD_USE_CUDA}" \
            CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            TF_FORCE_GPU_ALLOW_GROWTH=true \
            LD_LIBRARY_PATH="${TF_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}" \
            bash "${CE_EVAL_ROOT}/run/run_gcd_evaluation_colab.sh" \
                "${image_dir}" "${xlsx_path}" "${csv_path}" >"${log_path}" 2>&1

            require_file "${csv_path}"
            printf '%s/%s %s evaluation complete: %s\n' \
                "${config_name}" "${benchmark_name}" "${split}" "${csv_path}"
        done
    done
done
