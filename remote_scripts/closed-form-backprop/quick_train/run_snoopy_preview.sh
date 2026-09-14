#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_ID="${GPU_ID:-0}"
CONFIG_PATH="${CONFIG_PATH:-remote_scripts/closed-form-backprop/quick_train/config.yaml}"

cd -- "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

echo "Installing Python dependencies..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" -r requirements.txt
else
    "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

mapfile -t CONFIG_VALUES < <(
    "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}

targets = config["target_concepts"]
target = targets[0] if isinstance(targets, list) else targets.split(",")[0].strip()

for value in (
    config["save_path"],
    config["file_name"],
    config["sd_ckpt"],
    target,
    config["seed"],
    config["num_inference_steps"],
    config["anchor_steps"],
):
    print(value)
PY
)

if [[ "${#CONFIG_VALUES[@]}" -ne 7 ]]; then
    echo "Failed to read required values from ${CONFIG_PATH}" >&2
    exit 1
fi

RUN_DIR="${CONFIG_VALUES[0]}"
FILE_NAME="${CONFIG_VALUES[1]}"
SD_CKPT="${CONFIG_VALUES[2]}"
TARGET_CONCEPT="${CONFIG_VALUES[3]}"
SAMPLE_SEED="${CONFIG_VALUES[4]}"
SAMPLE_STEPS="${CONFIG_VALUES[5]}"
ANCHOR_STEPS="${CONFIG_VALUES[6]}"
mkdir -p -- "${RUN_DIR}"

echo "Run directory: ${REPO_ROOT}/${RUN_DIR}"
echo "Training ${TARGET_CONCEPT} anchor (${ANCHOR_STEPS} steps)..."
"${PYTHON_BIN}" -u train_closed_form_backprop.py \
    --config "${CONFIG_PATH}" \
    2>&1 | tee "${RUN_DIR}/train.log"

EDIT_CHECKPOINT="${RUN_DIR}/${FILE_NAME}.safetensors"
SAMPLE_ROOT="${RUN_DIR}/samples"
PROMPTS='a drawing of {};{} sitting on a red sofa;{} running through a park;{} flying an airplane;{} reading a book'

for concept in "${TARGET_CONCEPT}" 'Pink Panther'; do
    if [[ "${concept}" == "${TARGET_CONCEPT}" ]]; then
        group='erase'
    else
        group='retain'
    fi
    echo "Generating 5 ${group} prompt comparisons for ${concept}..."
    "${PYTHON_BIN}" -u sample.py \
        --sd_ckpt "${SD_CKPT}" \
        --save_root "${SAMPLE_ROOT}" \
        --mode original,edit \
        --erase_type instance \
        --target_concept "${TARGET_CONCEPT}" \
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
echo "Paired images: ${REPO_ROOT}/${SAMPLE_ROOT}/${TARGET_CONCEPT}/{${TARGET_CONCEPT},Pink Panther}/combine"
