#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow_smoke.yaml}"
export SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-/workspace/cedit_val_clip_adv_smoke}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/cedit_val_clip_adv_smoke_k74}"

source "${WORKFLOW_DIR}/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

if [[ "$(readlink -f "${SOURCE_OUTPUT_ROOT}")" == "$(readlink -f "${OUTPUT_ROOT}")" ]]; then
    printf 'SOURCE_OUTPUT_ROOT and OUTPUT_ROOT must differ.\n' >&2
    exit 1
fi

reuse_directory() {
    local source_path="$1"
    local destination_path="$2"
    [[ -d "${source_path}" ]] || {
        printf 'Missing reusable directory: %s\n' "${source_path}" >&2
        exit 1
    }
    if [[ -L "${destination_path}" ]]; then
        [[ "$(readlink -f "${destination_path}")" == "$(readlink -f "${source_path}")" ]] || {
            printf 'Existing symlink points elsewhere: %s\n' "${destination_path}" >&2
            exit 1
        }
        return
    fi
    [[ ! -e "${destination_path}" ]] || {
        printf 'Refusing to replace existing path: %s\n' "${destination_path}" >&2
        exit 1
    }
    mkdir -p "$(dirname -- "${destination_path}")"
    ln -s "${source_path}" "${destination_path}"
}

for method in original legacy target_global_pairwise_residual_subspace; do
    reuse_directory \
        "${SOURCE_OUTPUT_ROOT}/images/${method}" \
        "${IMAGE_ROOT}/${method}"
    reuse_directory \
        "${SOURCE_OUTPUT_ROOT}/mscoco/${method}" \
        "${MSCOCO_IMAGE_ROOT}/${method}"
done
reuse_directory \
    "${SOURCE_OUTPUT_ROOT}/metrics/fid_cache" \
    "${FID_CACHE_ROOT}"

mkdir -p "${LOG_ROOT}/anchor" "${LOG_ROOT}/train"
manifest="${LEARNED_ANCHOR_ROOT}/manifest.json"
tensors="${LEARNED_ANCHOR_ROOT}/embeddings.safetensors"
if [[ ! -f "${manifest}" || ! -f "${tensors}" ]]; then
    if [[ -e "${LEARNED_ANCHOR_ROOT}" ]]; then
        printf 'Incomplete k74 anchor artifact exists: %s\n' \
            "${LEARNED_ANCHOR_ROOT}" >&2
        exit 1
    fi
    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u learn_anchor.py \
            --config "${WORKFLOW_DIR}/learn_anchor_config_k74.yaml" \
            --save_root "${LEARNED_ANCHOR_ROOT}" \
            2>&1 | tee "${LOG_ROOT}/anchor/van_gogh_k74.log"
fi
require_file "${manifest}"
require_file "${tensors}"

checkpoint="$(checkpoint_path clip_guided_learned_anchor)"
if [[ ! -f "${checkpoint}" || "${manifest}" -nt "${checkpoint}" ]]; then
    mkdir -p "$(dirname -- "${checkpoint}")"
    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u train_erase_null.py \
            --config "${WORKFLOW_DIR}/train_config_learned_anchor.yaml" \
            --target_concepts "${TARGET_CONCEPT}" \
            --retain_path "data/${ERASE_TYPE}.csv" \
            --save_path "$(dirname -- "${checkpoint}")" \
            --learned_anchor_path "${LEARNED_ANCHOR_ROOT}" \
            2>&1 | tee "${LOG_ROOT}/train/clip_guided_learned_anchor_k74.log"
fi
require_file "${checkpoint}"

expected_per_content=$((STYLE_TEMPLATE_COUNT * NUM_SAMPLES_PER_PROMPT))
pending=()
IFS=',' read -r -a content_items <<< "${CONTENTS}"
for raw_content in "${content_items[@]}"; do
    content="${raw_content# }"
    content="${content% }"
    image_dir="$(few_image_dir clip_guided_learned_anchor "${content}")"
    clear_pngs_if_stale "${image_dir}" "${checkpoint}"
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
        --save_root "${IMAGE_ROOT}/clip_guided_learned_anchor/${ERASE_TYPE}" \
        --edit_ckpt "${checkpoint}"
fi
for raw_content in "${content_items[@]}"; do
    content="${raw_content# }"
    content="${content% }"
    assert_png_count \
        "$(few_image_dir clip_guided_learned_anchor "${content}")" \
        "${expected_per_content}" "k74/${content}"
done

coco_dir="$(mscoco_image_dir clip_guided_learned_anchor)"
clear_pngs_if_stale "${coco_dir}" "${checkpoint}"
if [[ "$(count_pngs "${coco_dir}")" != "${MSCOCO_NUM_PROMPTS}" ]]; then
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
        --save_root "${MSCOCO_IMAGE_ROOT}/clip_guided_learned_anchor" \
        --edit_ckpt "${checkpoint}"
fi
assert_png_count "${coco_dir}" "${MSCOCO_NUM_PROMPTS}" 'k74 MS-COCO'

bash "${WORKFLOW_DIR}/07_evaluate.sh"
printf 'Learned-anchor k74 evaluation complete: %s\n' "${OUTPUT_ROOT}"
