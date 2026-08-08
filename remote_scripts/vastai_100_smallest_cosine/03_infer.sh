#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${CHECKPOINT_PATH}"
cd -- "${REPO_ROOT}"

ERASE_IMAGE_DIR="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/erase/edit"
RETAIN_IMAGE_DIR="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/retain/edit"

erase_complete=0
retain_complete=0
if [[ -d "${ERASE_IMAGE_DIR}" && "$(count_pngs "${ERASE_IMAGE_DIR}")" == "${EXPECTED_IMAGE_COUNT}" ]]; then
    erase_complete=1
fi
if [[ -d "${RETAIN_IMAGE_DIR}" && "$(count_pngs "${RETAIN_IMAGE_DIR}")" == "${EXPECTED_IMAGE_COUNT}" ]]; then
    retain_complete=1
fi

if [[ "${erase_complete}" == "1" && "${retain_complete}" == "1" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
    printf 'Erase and retain images are already complete.\n'
    exit 0
fi

if [[ "${FORCE_RESAMPLE:-0}" == "1" || "${erase_complete}" == "${retain_complete}" ]]; then
    contents="erase, retain"
elif [[ "${erase_complete}" == "0" ]]; then
    contents="erase"
else
    contents="retain"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
    --sd_ckpt "${SD_CKPT}" \
    --erase_type "${BENCHMARK_NAME}" \
    --target_concept "${BENCHMARK_NAME}" \
    --contents "${contents}" \
    --mode edit \
    --batch_size "${BATCH_SIZE}" \
    --save_root "${IMAGE_ROOT}/${ANCHOR_MODE}" \
    --edit_ckpt "${CHECKPOINT_PATH}"

for split in erase retain; do
    image_dir="${IMAGE_ROOT}/${ANCHOR_MODE}/${BENCHMARK_NAME}/${split}/edit"
    image_count="$(count_pngs "${image_dir}")"
    [[ "${image_count}" == "${EXPECTED_IMAGE_COUNT}" ]] || {
        printf 'Expected %s %s images in %s, found %s\n' \
            "${EXPECTED_IMAGE_COUNT}" "${split}" "${image_dir}" "${image_count}" >&2
        exit 1
    }
    printf '%s inference complete: %s images in %s\n' \
        "${split}" "${image_count}" "${image_dir}"
done
