#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

require_file "${REPO_ROOT}/data/mscoco.csv"
if [[ "${MSCOCO_NUM_PROMPTS}" != "1000" ]]; then
    printf 'sample2.py fixes MS-COCO sampling at 1000 prompts; configured: %s\n' \
        "${MSCOCO_NUM_PROMPTS}" >&2
    exit 1
fi
if (( MSCOCO_NUM_PROMPTS % MSCOCO_BATCH_SIZE != 0 )); then
    printf 'mscoco_num_prompts must be divisible by mscoco_batch_size.\n' >&2
    exit 1
fi

generate_original_mscoco() {
    local image_dir image_count
    image_dir="$(mscoco_image_dir_for_run original shared)"
    image_count=0
    [[ -d "${image_dir}" ]] && image_count="$(count_pngs "${image_dir}")"
    if [[ "${image_count}" == "${MSCOCO_NUM_PROMPTS}" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
        printf 'Shared original MS-COCO images are complete: %s\n' "${image_dir}"
        return
    fi

    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" sample2.py \
        --sd_ckpt "${SD_CKPT}" \
        --target_concept original \
        --contents coco \
        --mode original \
        --num_samples 1 \
        --batch_size "${MSCOCO_BATCH_SIZE}" \
        --save_root "${MSCOCO_IMAGE_ROOT}"

    image_count="$(count_pngs "${image_dir}")"
    [[ "${image_count}" == "${MSCOCO_NUM_PROMPTS}" ]] || {
        printf 'Expected %s original MS-COCO images in %s, found %s\n' \
            "${MSCOCO_NUM_PROMPTS}" "${image_dir}" "${image_count}" >&2
        exit 1
    }
}

generate_edited_mscoco() {
    local method benchmark_name checkpoint_path image_dir image_count
    method="$1"
    benchmark_name="$2"
    checkpoint_path="$(checkpoint_dir_for_run "${method}" "${benchmark_name}")/weight.pt"
    image_dir="$(mscoco_image_dir_for_run "${method}" "${benchmark_name}")"
    require_file "${checkpoint_path}"
    image_count=0
    [[ -d "${image_dir}" ]] && image_count="$(count_pngs "${image_dir}")"
    if [[ "${image_count}" == "${MSCOCO_NUM_PROMPTS}" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
        printf '%s %s MS-COCO images are complete: %s\n' \
            "${method}" "${benchmark_name}" "${image_dir}"
        return
    fi

    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" sample2.py \
        --sd_ckpt "${SD_CKPT}" \
        --target_concept "${benchmark_name}" \
        --contents coco \
        --mode edit \
        --num_samples 1 \
        --batch_size "${MSCOCO_BATCH_SIZE}" \
        --save_root "${MSCOCO_IMAGE_ROOT}/${method}" \
        --edit_ckpt "${checkpoint_path}"

    image_count="$(count_pngs "${image_dir}")"
    [[ "${image_count}" == "${MSCOCO_NUM_PROMPTS}" ]] || {
        printf 'Expected %s edited MS-COCO images in %s, found %s\n' \
            "${MSCOCO_NUM_PROMPTS}" "${image_dir}" "${image_count}" >&2
        exit 1
    }
}

generate_original_mscoco
for method in legacy target_global_pairwise_residual_subspace; do
    for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
        generate_edited_mscoco "${method}" "${benchmark_name}"
    done
done

printf 'MS-COCO generation complete.\n'
