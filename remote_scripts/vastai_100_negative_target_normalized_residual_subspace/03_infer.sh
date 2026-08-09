#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

for subspace_k in "${K_VALUES[@]}"; do
    checkpoint_path="$(checkpoint_dir_for_k "${subspace_k}")/weight.pt"
    image_root="$(image_root_for_k "${subspace_k}")"
    require_file "${checkpoint_path}"
    erase_image_dir="${image_root}/${BENCHMARK_NAME}/erase/edit"
    retain_image_dir="${image_root}/${BENCHMARK_NAME}/retain/edit"

    erase_complete=0
    retain_complete=0
    [[ -d "${erase_image_dir}" && "$(count_pngs "${erase_image_dir}")" == "${EXPECTED_IMAGE_COUNT}" ]] && erase_complete=1
    [[ -d "${retain_image_dir}" && "$(count_pngs "${retain_image_dir}")" == "${EXPECTED_IMAGE_COUNT}" ]] && retain_complete=1
    if [[ "${erase_complete}" == "1" && "${retain_complete}" == "1" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
        printf 'k=%s erase and retain images are already complete.\n' "${subspace_k}"
        continue
    fi

    if [[ "${FORCE_RESAMPLE:-0}" == "1" || "${erase_complete}" == "${retain_complete}" ]]; then
        contents="erase, retain"
    elif [[ "${erase_complete}" == "0" ]]; then
        contents="erase"
    else
        contents="retain"
    fi

    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
        --sd_ckpt "${SD_CKPT}" \
        --erase_type "${BENCHMARK_NAME}" \
        --target_concept "${BENCHMARK_NAME}" \
        --contents "${contents}" \
        --mode edit \
        --batch_size "${BATCH_SIZE}" \
        --save_root "${image_root}" \
        --edit_ckpt "${checkpoint_path}"

    for split in erase retain; do
        image_dir="${image_root}/${BENCHMARK_NAME}/${split}/edit"
        image_count="$(count_pngs "${image_dir}")"
        [[ "${image_count}" == "${EXPECTED_IMAGE_COUNT}" ]] || {
            printf 'k=%s expected %s %s images in %s, found %s\n' "${subspace_k}" "${EXPECTED_IMAGE_COUNT}" "${split}" "${image_dir}" "${image_count}" >&2
            exit 1
        }
        printf 'k=%s %s inference complete: %s images\n' "${subspace_k}" "${split}" "${image_count}"
    done
done
