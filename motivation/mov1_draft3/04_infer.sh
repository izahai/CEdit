#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

for method in "${METHODS[@]}"; do
    for scale in "${RESIDUAL_SCALES[@]}"; do
        checkpoint_path="$(checkpoint_dir_for_run "${method}" "${scale}")/weight.pt"
        image_root="$(image_root_for_run "${method}" "${scale}")"
        checkpoint_manifest="${image_root}/checkpoint.sha256"
        erase_dir="${image_root}/${BENCHMARK_NAME}/erase/edit"
        retain_dir="${image_root}/${BENCHMARK_NAME}/retain/edit"
        require_file "${checkpoint_path}"
        checkpoint_hash="$(file_sha256 "${checkpoint_path}")"

        erase_complete=0
        retain_complete=0
        recorded_hash=""
        [[ -f "${checkpoint_manifest}" ]] && recorded_hash="$(<"${checkpoint_manifest}")"
        if [[ "${recorded_hash}" == "${checkpoint_hash}" ]]; then
            [[ -d "${erase_dir}" && "$(count_pngs "${erase_dir}")" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] && erase_complete=1
            [[ -d "${retain_dir}" && "$(count_pngs "${retain_dir}")" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] && retain_complete=1
        fi
        if [[ "${erase_complete}" == "1" && "${retain_complete}" == "1" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
            printf 'Images complete; skipping method=%s scale=%s\n' "${method}" "${scale}"
            continue
        fi
        if [[ "${FORCE_RESAMPLE:-0}" == "1" || "${erase_complete}" == "${retain_complete}" ]]; then
            contents="erase, retain"
        elif [[ "${erase_complete}" == "0" ]]; then
            contents="erase"
        else
            contents="retain"
        fi

        log_path="${LOG_DIR}/infer_${method}_scale_$(scale_slug "${scale}").log"
        mkdir -p "${image_root}" "${LOG_DIR}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_TF=0 TRANSFORMERS_NO_TF=1 \
        "${PYTHON_BIN}" sample2.py \
            --sd_ckpt "${SD_CKPT}" \
            --erase_type "${BENCHMARK_NAME}" \
            --target_concept "${BENCHMARK_NAME}" \
            --contents "${contents}" \
            --mode edit \
            --num_samples 1 \
            --batch_size "${BATCH_SIZE}" \
            --total_timesteps "${INFERENCE_TIMESTEPS}" \
            --save_root "${image_root}" \
            --edit_ckpt "${checkpoint_path}" \
            2>&1 | tee "${log_path}"

        for split in erase retain; do
            image_dir="${image_root}/${BENCHMARK_NAME}/${split}/edit"
            count="$(count_pngs "${image_dir}")"
            [[ "${count}" == "${EXPECTED_IMAGES_PER_SPLIT}" ]] || {
                printf 'Expected %s images in %s, found %s\n' \
                    "${EXPECTED_IMAGES_PER_SPLIT}" "${image_dir}" "${count}" >&2
                exit 1
            }
        done
        printf '%s\n' "${checkpoint_hash}" >"${checkpoint_manifest}"
    done
done
