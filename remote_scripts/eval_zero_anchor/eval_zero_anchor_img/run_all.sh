#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-30}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

ART_CONFIG="${SCRIPT_DIR}/train_art.yaml"
ZERO_CONFIG="${SCRIPT_DIR}/train_zero.yaml"
ART_WEIGHT="${SCRIPT_DIR}/outputs/checkpoints/art/weight.pt"
ZERO_WEIGHT="${SCRIPT_DIR}/outputs/checkpoints/zero/weight.pt"
OUTPUT_IMG_DIR="${SCRIPT_DIR}/outputs/concat_images"

echo "=========================================================="
echo " Evaluation: Van Gogh Erasure (Original vs Art vs Zero)"
echo " Working directory: ${REPO_ROOT}"
echo " GPU:               ${GPU_ID}"
echo " Python binary:     ${PYTHON_BIN}"
echo " Art Config:        ${ART_CONFIG}"
echo " Zero Config:       ${ZERO_CONFIG}"
echo " Inference Steps:   ${STEPS}"
echo " Force Retrain:     ${FORCE_RETRAIN}"
echo " Install Deps:      ${INSTALL_DEPS}"
echo "=========================================================="

# ------------------------------------------------------------------
# Step 0: Ensure dependencies are installed (using uv if available)
# ------------------------------------------------------------------
if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo ""
  echo "--- Step 0: Ensuring dependencies are installed from requirements.txt ---"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" -r "${REPO_ROOT}/requirements.txt"
  else
    "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
  fi
fi

# ------------------------------------------------------------------
# Step 1: Train SPEED with normal anchor ("art")
# ------------------------------------------------------------------
if [[ "${FORCE_RETRAIN}" == "1" || ! -f "${ART_WEIGHT}" ]]; then
  echo ""
  echo "--- Step 1: Training SPEED with 'art' anchor ---"
  "${PYTHON_BIN}" train_erase_null.py --config "${ART_CONFIG}"
else
  echo ""
  echo "--- Step 1: Checkpoint found at ${ART_WEIGHT} (Skipping, set FORCE_RETRAIN=1 to rerun) ---"
fi

# ------------------------------------------------------------------
# Step 2: Train SPEED with zero anchor ("<zero>")
# ------------------------------------------------------------------
if [[ "${FORCE_RETRAIN}" == "1" || ! -f "${ZERO_WEIGHT}" ]]; then
  echo ""
  echo "--- Step 2: Training SPEED with '<zero>' anchor ---"
  "${PYTHON_BIN}" train_erase_null.py --config "${ZERO_CONFIG}"
else
  echo ""
  echo "--- Step 2: Checkpoint found at ${ZERO_WEIGHT} (Skipping, set FORCE_RETRAIN=1 to rerun) ---"
fi

# ------------------------------------------------------------------
# Step 3: Run 3-Way Inference & Concatenation [Original | Art | Zero]
# ------------------------------------------------------------------
echo ""
echo "--- Step 3: Generating 20 3-Way Concatenated Images [Original | Art | Zero] ---"
"${PYTHON_BIN}" "${SCRIPT_DIR}/preview_concat.py" \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --art_ckpt "${ART_WEIGHT}" \
    --zero_ckpt "${ZERO_WEIGHT}" \
    --style_csv "data/style.csv" \
    --output_dir "${OUTPUT_IMG_DIR}" \
    --steps "${STEPS}" \
    --seed 42 \
    --device "cuda"

echo ""
echo "=========================================================="
echo " Pipeline Complete!"
echo " Art Anchor weights:  ${ART_WEIGHT}"
echo " Zero Anchor weights: ${ZERO_WEIGHT}"
echo " 20 3-way concatenated images saved to:"
echo "   Van Gogh:       ${OUTPUT_IMG_DIR}/van_gogh/ (10 images)"
echo "   Random artists: ${OUTPUT_IMG_DIR}/random_artists/ (10 images)"
echo " Image dimensions: 512x1536 [Original | Art Anchor | Zero Anchor]"
echo "=========================================================="

