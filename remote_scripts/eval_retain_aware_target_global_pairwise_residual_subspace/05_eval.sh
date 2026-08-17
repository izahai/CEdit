#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss"
RESOURCES_DIR="${CELEB_DIR}/examples/resources"
EVALUATOR="${WORKFLOW_DIR}/evaluate_by_GCD_workers.py"
require_directory "${CELEB_DIR}"
require_file "${EVALUATOR}"
require_file "${RESOURCES_DIR}/face_recognition/labels.csv"
require_file "${RESOURCES_DIR}/face_recognition/best_model_states.pkl"
mkdir -p "${GCD_OUTPUT_DIR}"
TF_CUDA_LIBRARY_PATH="$(tensorflow_cuda_library_path)"

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
    image_root="$(image_root_for_run "${benchmark_name}")"
    gcd_output_dir="$(gcd_output_dir_for_run "${benchmark_name}")"
    mkdir -p "${gcd_output_dir}"
    for split in erase retain; do
        image_dir="${image_root}/${benchmark_name}/${split}/edit"
        csv_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.csv"
        xlsx_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.xlsx"
        log_path="${gcd_output_dir}/${ANCHOR_MODE}_${split}.log"
        require_directory "${image_dir}"

        if [[ -f "${csv_path}" && "${FORCE_EVAL:-0}" != "1" ]]; then
            printf '%s %s evaluation exists; skipping: %s\n' \
                "${benchmark_name}" "${split}" "${csv_path}"
            continue
        fi

        printf 'Evaluating %s %s with %s workers and batch size %s\n' \
            "${benchmark_name}" "${split}" \
            "${GCD_NUM_WORKERS}" "${GCD_BATCH_SIZE}"
        PYTHONPATH="${REPO_ROOT}:${CELEB_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        APP_DATA_DIR="${RESOURCES_DIR}" \
        APP_RECOGNITION_WEIGHTS_FILE="face_recognition/best_model_states.pkl" \
        APP_FACE_MARGIN="${APP_FACE_MARGIN:-0.2}" \
        APP_FACE_SIZE="${APP_FACE_SIZE:-224}" \
        APP_USE_CUDA="${GCD_USE_CUDA}" \
        USE_CUDA="${GCD_USE_CUDA}" \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        LD_LIBRARY_PATH="${TF_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}" \
        "${PYTHON_BIN}" "${EVALUATOR}" \
            --image_folder "${image_dir}" \
            --save_excel_path "${xlsx_path}" \
            --save_csv_path "${csv_path}" \
            --num_workers "${GCD_NUM_WORKERS}" \
            --batch_size "${GCD_BATCH_SIZE}" \
            --prefetch_factor "${GCD_PREFETCH_FACTOR}" \
            >"${log_path}" 2>&1

        require_file "${csv_path}"
        printf '%s %s evaluation complete: %s\n' \
            "${benchmark_name}" "${split}" "${csv_path}"
    done
done
