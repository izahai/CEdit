#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

while IFS="${FIELD_SEPARATOR}" read -r \
    erase_type contents prompt_templates prompt_count expected_count; do
    pending_contents=()
    IFS=',' read -r -a content_items <<< "${contents}"
    for raw_content in "${content_items[@]}"; do
        content="${raw_content# }"
        content="${content% }"
        image_dir="$(few_image_dir_for_run original "${erase_type}" shared "${content}")"
        clear_pngs_if_forced "${image_dir}"
        if [[ "$(count_pngs "${image_dir}")" != "${expected_count}" ]]; then
            pending_contents+=("${content}")
        fi
    done

    if (( ${#pending_contents[@]} == 0 )); then
        printf '%s original images are complete.\n' "${erase_type}"
        continue
    fi

    pending_raw="$(join_by_comma_space "${pending_contents[@]}")"
    printf 'Generating original %s images for: %s\n' \
        "${erase_type}" "${pending_raw}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample.py \
        --sd_ckpt "${SD_CKPT}" \
        --seed "${SEED}" \
        --erase_type "${erase_type}" \
        --target_concept shared \
        --contents "${pending_raw}" \
        --prompts "${prompt_templates}" \
        --mode original \
        --num_samples "${NUM_SAMPLES_PER_PROMPT}" \
        --batch_size "${BATCH_SIZE}" \
        --total_timesteps "${INFERENCE_TIMESTEPS}" \
        --guidance_scale "${GUIDANCE_SCALE}" \
        --save_root "${IMAGE_ROOT}/original/${erase_type}"

    for content in "${pending_contents[@]}"; do
        image_dir="$(few_image_dir_for_run original "${erase_type}" shared "${content}")"
        assert_png_count "${image_dir}" "${expected_count}" \
            "${erase_type}/${content} original"
    done
done < <(domain_rows)

printf 'Original few-concept generation complete.\n'
