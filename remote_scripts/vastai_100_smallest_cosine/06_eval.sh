#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

mkdir -p "${GCD_OUTPUT_DIR}"

for split in erase retain; do
    image_dir="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/${split}/edit"
    csv_path="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_${split}.csv"
    xlsx_path="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_${split}.xlsx"
    log_path="${GCD_OUTPUT_DIR}/${ANCHOR_MODE}_${split}.log"
    require_directory "${image_dir}"

    if [[ -f "${csv_path}" && "${FORCE_EVAL:-0}" != "1" ]]; then
        printf '%s evaluation exists; skipping: %s\n' "${split}" "${csv_path}"
        continue
    fi

    printf 'Evaluating %s images...\n' "${split}"
    CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss" \
    EVALUATE_SCRIPT="${CE_EVAL_ROOT}/eval/evaluate_by_GCD.py" \
    GCD_USE_CUDA="${GCD_USE_CUDA}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    bash "${CE_EVAL_ROOT}/run/run_gcd_evaluation_colab.sh" \
        "${image_dir}" "${xlsx_path}" "${csv_path}" >"${log_path}" 2>&1

    require_file "${csv_path}"
    printf '%s evaluation complete: %s\n' "${split}" "${csv_path}"
done
