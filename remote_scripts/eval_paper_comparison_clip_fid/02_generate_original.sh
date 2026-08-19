#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

method="original"
mode="$(image_mode_for_method "${method}")"
for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
    benchmark_csv="$(benchmark_csv_for_name "${benchmark_name}")"
    image_root="$(image_root_for_run "${method}" "${benchmark_name}")"
    require_file "${benchmark_csv}"
    target_concepts_for_benchmark \
        "${benchmark_csv}" "${benchmark_name}" >/dev/null

    erase_image_dir="${image_root}/${benchmark_name}/erase/${mode}"
    retain_image_dir="${image_root}/${benchmark_name}/retain/${mode}"
    erase_complete=0
    retain_complete=0
    [[ -d "${erase_image_dir}" && "$(count_pngs "${erase_image_dir}")" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] && erase_complete=1
    [[ -d "${retain_image_dir}" && "$(count_pngs "${retain_image_dir}")" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] && retain_complete=1
    if [[ "${erase_complete}" == "1" && "${retain_complete}" == "1" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
        printf '%s original erase and retain images are already complete.\n' \
            "${benchmark_name}"
        continue
    fi

    if [[ "${FORCE_RESAMPLE:-0}" == "1" || "${erase_complete}" == "${retain_complete}" ]]; then
        contents="erase, retain"
    elif [[ "${erase_complete}" == "0" ]]; then
        contents="erase"
    else
        contents="retain"
    fi

    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" sample2.py \
        --sd_ckpt "${SD_CKPT}" \
        --erase_type "${benchmark_name}" \
        --target_concept "${benchmark_name}" \
        --contents "${contents}" \
        --mode original \
        --num_samples 1 \
        --batch_size "${BATCH_SIZE}" \
        --save_root "${image_root}"

    for split in erase retain; do
        image_dir="${image_root}/${benchmark_name}/${split}/${mode}"
        image_count="$(count_pngs "${image_dir}")"
        [[ "${image_count}" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] || {
            printf '%s expected %s original %s images in %s, found %s\n' \
                "${benchmark_name}" "${EXPECTED_IMAGES_PER_SPLIT}" \
                "${split}" "${image_dir}" "${image_count}" >&2
            exit 1
        }
        printf '%s original %s inference complete: %s images\n' \
            "${benchmark_name}" "${split}" "${image_count}"
    done
done
