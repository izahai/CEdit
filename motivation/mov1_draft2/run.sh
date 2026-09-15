#!/usr/bin/env bash
set -Eeuo pipefail

MOV2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${MOV2_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_ID="${GPU_ID:-0}"

cd -- "${REPO_ROOT}"
mkdir -p "${MOV2_DIR}/outputs"

CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_TF=0 TRANSFORMERS_NO_TF=1 \
"${PYTHON_BIN}" "${MOV2_DIR}/generate_figure.py" \
    --config "${MOV2_DIR}/config.yaml" \
    --output-dir "${MOV2_DIR}/outputs"

