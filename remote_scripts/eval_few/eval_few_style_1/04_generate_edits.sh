#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

for method in "${EDITED_METHODS[@]}"; do
    while IFS="${FIELD_SEPARATOR}" read -r \
        task_id erase_type target_concepts anchor_concept target_count \
        applied_rank contents prompt_templates prompt_count expected_count; do
        checkpoint_path="$(checkpoint_path_for_run "${method}" "${task_id}")"
        require_file "${checkpoint_path}"
        pending_contents=()
        IFS=',' read -r -a content_items <<< "${contents}"
        for raw_content in "${content_items[@]}"; do
            content="${raw_content# }"
            content="${content% }"
            image_dir="$(few_image_dir_for_run \
                "${method}" "${erase_type}" "${task_id}" "${content}")"
            clear_pngs_if_forced "${image_dir}"
            if [[ "$(count_pngs "${image_dir}")" != "${expected_count}" ]]; then
                pending_contents+=("${content}")
            fi
        done

        if (( ${#pending_contents[@]} == 0 )); then
            printf '%s/%s edited images are complete.\n' \
                "${method}" "${task_id}"
            continue
        fi

        pending_raw="$(join_by_comma_space "${pending_contents[@]}")"
        printf 'Generating %s/%s edited images for: %s\n' \
            "${method}" "${task_id}" "${pending_raw}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample.py \
            --sd_ckpt "${SD_CKPT}" \
            --seed "${SEED}" \
            --erase_type "${erase_type}" \
            --target_concept "${task_id}" \
            --contents "${pending_raw}" \
            --prompts "${prompt_templates}" \
            --mode edit \
            --num_samples "${NUM_SAMPLES_PER_PROMPT}" \
            --batch_size "${BATCH_SIZE}" \
            --total_timesteps "${INFERENCE_TIMESTEPS}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --save_root "${IMAGE_ROOT}/${method}/${erase_type}" \
            --edit_ckpt "${checkpoint_path}"

        for content in "${pending_contents[@]}"; do
            image_dir="$(few_image_dir_for_run \
                "${method}" "${erase_type}" "${task_id}" "${content}")"
            assert_png_count "${image_dir}" "${expected_count}" \
                "${method}/${task_id}/${content}"
        done
    done < <(task_rows)
done

printf 'Edited few-concept generation complete.\n'

