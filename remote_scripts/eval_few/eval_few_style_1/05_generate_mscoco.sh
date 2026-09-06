#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
require_file "${REPO_ROOT}/data/mscoco.csv"

original_dir="$(mscoco_image_dir_for_run original shared)"
clear_pngs_if_forced "${original_dir}"
if [[ "$(count_pngs "${original_dir}")" != "${MSCOCO_NUM_PROMPTS}" ]]; then
    printf 'Generating %s shared original MS-COCO images.\n' \
        "${MSCOCO_NUM_PROMPTS}"
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
assert_png_count "${original_dir}" "${MSCOCO_NUM_PROMPTS}" \
    'original MS-COCO'

for method in "${EDITED_METHODS[@]}"; do
    while IFS="${FIELD_SEPARATOR}" read -r \
        task_id erase_type target_concepts anchor_concept target_count \
        applied_rank contents prompt_templates prompt_count expected_count; do
        checkpoint_path="$(checkpoint_path_for_run "${method}" "${task_id}")"
        require_file "${checkpoint_path}"
        image_dir="$(mscoco_image_dir_for_run "${method}" "${task_id}")"
        clear_pngs_if_forced "${image_dir}"
        if [[ "$(count_pngs "${image_dir}")" == "${MSCOCO_NUM_PROMPTS}" ]]; then
            printf '%s/%s MS-COCO images are complete.\n' \
                "${method}" "${task_id}"
            continue
        fi

        printf 'Generating %s/%s MS-COCO images.\n' \
            "${method}" "${task_id}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample2.py \
            --sd_ckpt "${SD_CKPT}" \
            --seed "${SEED}" \
            --target_concept "${task_id}" \
            --contents coco \
            --mode edit \
            --num_samples 1 \
            --batch_size "${MSCOCO_BATCH_SIZE}" \
            --max_samples "${MSCOCO_NUM_PROMPTS}" \
            --total_timesteps "${INFERENCE_TIMESTEPS}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --save_root "${MSCOCO_IMAGE_ROOT}/${method}" \
            --edit_ckpt "${checkpoint_path}"
        assert_png_count "${image_dir}" "${MSCOCO_NUM_PROMPTS}" \
            "${method}/${task_id} MS-COCO"
    done < <(task_rows)
done

printf 'MS-COCO generation complete.\n'

