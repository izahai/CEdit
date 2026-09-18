#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd -- "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESAM_CONFIG="${RESAM_CONFIG:-${SCRIPT_DIR}/train_config.yaml}"
SPEED_CONFIG="${SPEED_CONFIG:-${SCRIPT_DIR}/speed_config.yaml}"
PREVIEW_STEPS="${PREVIEW_STEPS:-20}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

echo "=========================================================="
echo " Van Gogh Erasure Pipeline: SPEED vs Ours (ReSAM)"
echo " Working directory: ${REPO_ROOT}"
echo " GPU: ${GPU_ID}"
echo " Python binary: ${PYTHON_BIN}"
echo " ReSAM Config:  ${RESAM_CONFIG}"
echo " SPEED Config:  ${SPEED_CONFIG}"
echo " Preview Steps: ${PREVIEW_STEPS}"
echo " Force Retrain: ${FORCE_RETRAIN}"
echo "=========================================================="

# ------------------------------------------------------------------
# Step 1: Train Ours (ReSAM Sparse Anchor Mixture)
# ------------------------------------------------------------------
RESAM_WEIGHT="logs/resam/van_gogh/weight.safetensors"
if [[ "${FORCE_RETRAIN}" == "1" || ! -f "${RESAM_WEIGHT}" ]]; then
  echo ""
  echo "--- Step 1: Training ReSAM Sparse Anchor Mixture ---"
  "${PYTHON_BIN}" train_resam.py --config "${RESAM_CONFIG}"
else
  echo ""
  echo "--- Step 1: ReSAM checkpoint found at ${RESAM_WEIGHT} (Skipping training, set FORCE_RETRAIN=1 to rerun) ---"
fi

# ------------------------------------------------------------------
# Step 2: Train Baseline SPEED
# ------------------------------------------------------------------
SPEED_WEIGHT="logs/speed/van_gogh/weight.pt"
if [[ "${FORCE_RETRAIN}" == "1" || ! -f "${SPEED_WEIGHT}" ]]; then
  echo ""
  echo "--- Step 2: Training Baseline SPEED (rank-100 retain projection) ---"
  "${PYTHON_BIN}" train_erase_null.py --config "${SPEED_CONFIG}"
else
  echo ""
  echo "--- Step 2: SPEED checkpoint found at ${SPEED_WEIGHT} (Skipping training, set FORCE_RETRAIN=1 to rerun) ---"
fi

# ------------------------------------------------------------------
# Step 3: Generate 3-Way Preview Image Comparisons (Original | SPEED | Ours)
# ------------------------------------------------------------------
echo ""
echo "--- Step 3: Generating 3-Way Comparisons (Original | SPEED | Ours) ---"
"${PYTHON_BIN}" "${SCRIPT_DIR}/preview_samples.py" \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --speed_ckpt "${SPEED_WEIGHT}" \
    --ours_ckpt "${RESAM_WEIGHT}" \
    --style_csv "data/style.csv" \
    --save_dir "logs/resam/van_gogh/preview_samples" \
    --device "cuda" \
    --steps "${PREVIEW_STEPS}" \
    --seed 42

echo ""
echo "=========================================================="
echo " Pipeline Complete!"
echo " SPEED weights: ${SPEED_WEIGHT}"
echo " ReSAM weights: ${RESAM_WEIGHT}"
echo " 3-Way preview images:"
echo "   Van Gogh: logs/resam/van_gogh/preview_samples/van_gogh/"
echo "   Retain:   logs/resam/van_gogh/preview_samples/retain/"
echo "=========================================================="
