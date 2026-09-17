#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"
require_file "${CHECKPOINT_PATH}"

complete=1
for state in original edit; do
    for identity in "${TARGET_A}" "${ANCHOR_B}"; do
        directory="$(image_dir "${identity}" "${state}")"
        if [[ ! -d "${directory}" || "$(count_pngs "${directory}")" != \
            "${EXPECTED_IMAGES_PER_IDENTITY}" ]]; then
            complete=0
        fi
    done
done
if [[ "${complete}" == "1" && "${FORCE_RESAMPLE:-0}" != "1" ]]; then
    printf 'All matched original/edit images already exist; skipping.\n'
    exit 0
fi

USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" sample.py \
    --sd_ckpt "${SD_CKPT}" \
    --seed "${SEED}" \
    --mode "original,edit" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --total_timesteps "${DIFFUSION_STEPS}" \
    --num_samples "${SAMPLES_PER_TEMPLATE}" \
    --batch_size "${SAMPLE_BATCH_SIZE}" \
    --prompts "${PROMPT_TEMPLATES}" \
    --erase_type celebrity \
    --target_concept "$(run_slug)" \
    --contents "${TARGET_A}, ${ANCHOR_B}" \
    --save_root "${IMAGE_ROOT}" \
    --edit_ckpt "${CHECKPOINT_PATH}"

for state in original edit; do
    for identity in "${TARGET_A}" "${ANCHOR_B}"; do
        directory="$(image_dir "${identity}" "${state}")"
        count="$(count_pngs "${directory}")"
        [[ "${count}" == "${EXPECTED_IMAGES_PER_IDENTITY}" ]] || {
            printf 'Expected %s images in %s, found %s\n' \
                "${EXPECTED_IMAGES_PER_IDENTITY}" "${directory}" "${count}" >&2
            exit 1
        }
    done
done

