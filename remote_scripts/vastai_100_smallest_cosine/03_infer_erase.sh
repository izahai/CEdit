#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${CHECKPOINT_PATH}"
cd -- "${REPO_ROOT}"
IMAGE_DIR="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/erase/edit"
if [[ -d "${IMAGE_DIR}" && "$(count_pngs "${IMAGE_DIR}")" == "${EXPECTED_IMAGE_COUNT}" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
    printf 'Erase images already complete: %s\n' "${IMAGE_DIR}"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
    --sd_ckpt "${SD_CKPT}" \
    --erase_type "${BENCHMARK_NAME}" \
    --target_concept "${BENCHMARK_NAME}" \
    --contents erase \
    --mode edit \
    --batch_size "${BATCH_SIZE}" \
    --save_root "${IMAGE_ROOT}/${ANCHOR_MODE}" \
    --edit_ckpt "${CHECKPOINT_PATH}"

[[ "$(count_pngs "${IMAGE_DIR}")" == "${EXPECTED_IMAGE_COUNT}" ]] || {
    printf 'Expected %s erase images in %s\n' "${EXPECTED_IMAGE_COUNT}" "${IMAGE_DIR}" >&2
    exit 1
}
