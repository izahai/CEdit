#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_DIR="${RUN_DIR:-logs/closed_form_backprop/snoopy_preview_$(date -u +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-2}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
SAMPLE_STEPS="${SAMPLE_STEPS:-15}"
SAMPLE_SEED="${SAMPLE_SEED:-1234}"

cd -- "${REPO_ROOT}"
mkdir -p -- "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

echo "Run directory: ${REPO_ROOT}/${RUN_DIR}"
echo "Training Snoopy anchor (${TRAIN_STEPS} steps)..."
"${PYTHON_BIN}" -u train_closed_form_backprop.py \
    --config configs/closed_form_backprop.yaml \
    --save_path "${RUN_DIR}" \
    --anchor_steps "${TRAIN_STEPS}" \
    --validation_samples "${VALIDATION_SAMPLES}" \
    --validation_interval 5 \
    --num_inference_steps "${DENOISING_STEPS}" \
    --max_anchor_norm target \
    2>&1 | tee "${RUN_DIR}/train.log"

EDIT_CHECKPOINT="${RUN_DIR}/weight.safetensors"
SAMPLE_ROOT="${RUN_DIR}/samples"
PROMPTS='a drawing of {};{} sitting on a red sofa;{} running through a park;{} flying an airplane;{} reading a book'

for concept in 'Snoopy' 'Pink Panther'; do
    if [[ "${concept}" == 'Snoopy' ]]; then
        group='erase'
    else
        group='retain'
    fi
    echo "Generating 5 ${group} prompt comparisons for ${concept}..."
    "${PYTHON_BIN}" -u sample.py \
        --sd_ckpt CompVis/stable-diffusion-v1-4 \
        --save_root "${SAMPLE_ROOT}" \
        --mode original,edit \
        --erase_type instance \
        --target_concept Snoopy \
        --contents "${concept}" \
        --prompts "${PROMPTS}" \
        --edit_ckpt "${EDIT_CHECKPOINT}" \
        --seed "${SAMPLE_SEED}" \
        --num_samples 1 \
        --batch_size 1 \
        --total_timesteps "${SAMPLE_STEPS}" \
        2>&1 | tee "${RUN_DIR}/sample_${group}.log"
done

echo "Finished. Checkpoint: ${REPO_ROOT}/${EDIT_CHECKPOINT}"
echo "Paired images: ${REPO_ROOT}/${SAMPLE_ROOT}/Snoopy/{Snoopy,Pink Panther}/combine"
