#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${OUTPUT_ROOT}"
cd -- "${REPO_ROOT}"

printf 'Analyzing natural target and anchor geometry on GPU %s\n' "${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
TOKENIZERS_PARALLELISM=false \
USE_TF=0 \
TRANSFORMERS_NO_TF=1 \
"${PYTHON_BIN}" -u "${WORKFLOW_DIR}/analyze_target_pairwise_cosine.py" \
    --config "${WORKFLOW_CONFIG}" \
    --output-dir "${OUTPUT_ROOT}"

printf 'Analysis complete: %s\n' "${OUTPUT_ROOT}"
