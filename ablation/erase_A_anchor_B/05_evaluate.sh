#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss"
RESOURCES_DIR="${CELEB_DIR}/examples/resources"
EVALUATOR="${WORKFLOW_DIR}/evaluate_pair_probabilities.py"
require_file "${EVALUATOR}"
require_file "${RESOURCES_DIR}/face_recognition/best_model_states.pkl"
mkdir -p "${GCD_ROOT}"
TF_CUDA_LIBRARY_PATH="$(tensorflow_cuda_library_path)"

for state in original edit; do
    for identity in "${TARGET_A}" "${ANCHOR_B}"; do
        directory="$(image_dir "${identity}" "${state}")"
        output="$(gcd_csv "${state}" "${identity}")"
        require_directory "${directory}"
        if [[ -f "${output}" && "${FORCE_EVAL:-0}" != "1" ]]; then
            printf 'Evaluation exists; skipping: %s\n' "${output}"
            continue
        fi
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
            --image-dir "${directory}" \
            --prompt-identity "${identity}" \
            --target-a "${TARGET_A}" \
            --anchor-b "${ANCHOR_B}" \
            --model-state "${state}" \
            --output-csv "${output}" \
            --num-workers "${GCD_NUM_WORKERS}" \
            --batch-size "${GCD_BATCH_SIZE}" \
            --prefetch-factor "${GCD_PREFETCH_FACTOR}"
    done
done

