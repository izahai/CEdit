#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/closed_form_backprop.yaml}"

cd -- "${REPO_ROOT}"

extra_args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    extra_args+=(
        --anchor_steps 1
        --validation_interval 1
        --validation_samples 1
        --num_inference_steps 2
        --resolution 256
        --save_path "logs/closed_form_backprop/snoopy_smoke"
    )
fi

USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" -u train_closed_form_backprop.py \
    --config "${CONFIG_PATH}" \
    "${extra_args[@]}" \
    "$@"
