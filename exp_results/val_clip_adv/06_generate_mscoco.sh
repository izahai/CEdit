#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

original_dir="$(mscoco_image_dir original)"
clear_pngs_if_forced "${original_dir}"
if [[ "$(count_pngs "${original_dir}")" != "${MSCOCO_NUM_PROMPTS}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
        --sd_ckpt "${SD_CKPT}" \
        --seed "${SEED}" \
        --target_concept original \
        --contents coco \
        --mode original \
        --num_samples 1 \
        --batch_size "${MSCOCO_BATCH_SIZE}" \
        --max_samples "${MSCOCO_NUM_PROMPTS}" \
        --total_timesteps "${INFERENCE_TIMESTEPS}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --save_root "${MSCOCO_IMAGE_ROOT}"
fi
assert_png_count "${original_dir}" "${MSCOCO_NUM_PROMPTS}" 'original MS-COCO'

for method in "${EDITED_METHODS[@]}"; do
    checkpoint="$(checkpoint_path "${method}")"
    require_file "${checkpoint}"
    image_dir="$(mscoco_image_dir "${method}")"
    clear_pngs_if_stale "${image_dir}" "${checkpoint}"
    clear_pngs_if_forced "${image_dir}"
    if [[ "$(count_pngs "${image_dir}")" != "${MSCOCO_NUM_PROMPTS}" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
            --sd_ckpt "${SD_CKPT}" \
            --seed "${SEED}" \
            --target_concept "${TASK_ID}" \
            --contents coco \
            --mode edit \
            --num_samples 1 \
            --batch_size "${MSCOCO_BATCH_SIZE}" \
            --max_samples "${MSCOCO_NUM_PROMPTS}" \
            --total_timesteps "${INFERENCE_TIMESTEPS}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --save_root "${MSCOCO_IMAGE_ROOT}/${method}" \
            --edit_ckpt "${checkpoint}"
    fi
    assert_png_count "${image_dir}" "${MSCOCO_NUM_PROMPTS}" \
        "${method} MS-COCO"
done
printf 'MS-COCO generation complete.\n'
