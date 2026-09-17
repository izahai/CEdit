#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${WORKFLOW_DIR}/../.." && pwd)"
CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/eval_100_ins}"
START_STAGE="${START_STAGE:-setup}"
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/workflow.log") 2>&1

stages=(setup prepare train sample evaluate)
start=0
found=0
for i in "${!stages[@]}"; do
    if [[ "${stages[i]}" == "${START_STAGE}" ]]; then
        start="${i}"
        found=1
    fi
done
if [[ "${found}" != 1 ]]; then
    printf 'Unknown START_STAGE: %s\n' "${START_STAGE}" >&2
    exit 2
fi

for ((i=start; i<${#stages[@]}; i++)); do
    stage="${stages[i]}"
    printf 'Running %s\n' "${stage}"
    if [[ "${stage}" == setup ]]; then
        "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "${PYTHON_BIN}" -r "${REPO_ROOT}/requirements.txt"
        else
            "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
        fi
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -c 'import torch, diffusers, transformers; assert torch.cuda.is_available()'
    else
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u "${WORKFLOW_DIR}/workflow.py" "${stage}" \
            --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" \
            --python-bin "${PYTHON_BIN}" --gpu-id "${GPU_ID}"
    fi
done
