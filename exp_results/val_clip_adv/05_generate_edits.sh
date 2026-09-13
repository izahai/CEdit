#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
expected_per_content=$((STYLE_TEMPLATE_COUNT * NUM_SAMPLES_PER_PROMPT))

for method in "${EDITED_METHODS[@]}"; do
    checkpoint="$(checkpoint_path "${method}")"
    require_file "${checkpoint}"
    pending=()
    IFS=',' read -r -a content_items <<< "${CONTENTS}"
    for raw_content in "${content_items[@]}"; do
        content="${raw_content# }"
        content="${content% }"
        image_dir="$(few_image_dir "${method}" "${content}")"
        clear_pngs_if_stale "${image_dir}" "${checkpoint}"
        clear_pngs_if_forced "${image_dir}"
        if [[ "$(count_pngs "${image_dir}")" != "${expected_per_content}" ]]; then
            pending+=("${content}")
        fi
    done
    if (( ${#pending[@]} > 0 )); then
        pending_contents="$(IFS=', '; printf '%s' "${pending[*]}")"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample.py \
            --sd_ckpt "${SD_CKPT}" \
            --seed "${SEED}" \
            --erase_type "${ERASE_TYPE}" \
            --target_concept "${TASK_ID}" \
            --contents "${pending_contents}" \
            --mode edit \
            --num_samples "${NUM_SAMPLES_PER_PROMPT}" \
            --batch_size "${BATCH_SIZE}" \
            --total_timesteps "${INFERENCE_TIMESTEPS}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --save_root "${IMAGE_ROOT}/${method}/${ERASE_TYPE}" \
            --edit_ckpt "${checkpoint}"
    fi
    for raw_content in "${content_items[@]}"; do
        content="${raw_content# }"
        content="${content% }"
        assert_png_count "$(few_image_dir "${method}" "${content}")" \
            "${expected_per_content}" "${method}/${content}"
    done
done
printf 'Edited style generation complete.\n'
