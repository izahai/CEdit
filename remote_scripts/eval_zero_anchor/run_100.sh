#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# 100-Image Benchmark Evaluation: Art Anchor vs Zero Anchor
# Generates 100 Target (Van Gogh) + 100 Retain (Artists) = 200 3-way Comparisons
# Inference: 20 timesteps with DPM-Solver (fast)
# Checkpoints: Reuses existing SPEED Art and Zero models (no retraining needed)
# Metrics: End-to-end CLIP evaluation on the concatenated images
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
COUNT="${COUNT:-100}"
STEPS="${STEPS:-20}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STYLE_CSV="${STYLE_CSV:-data/style_100_retain_eval.csv}"

# Resolve Python binary
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "/venv/main/bin/python" ]]; then
    PYTHON_BIN="/venv/main/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# Resolve Checkpoints
ART_CKPT="${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/checkpoints/art/weight.pt"
ZERO_CKPT="${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/checkpoints/zero/weight.pt"

if [[ ! -f "${ART_CKPT}" ]]; then
    ALT_ART="${REPO_ROOT}/remote_scripts/eval_zero_anchor/eval_zero_anchor_img/outputs/checkpoints/art/weight.pt"
    if [[ -f "${ALT_ART}" ]]; then
        ART_CKPT="${ALT_ART}"
    fi
fi

if [[ ! -f "${ZERO_CKPT}" ]]; then
    ALT_ZERO="${REPO_ROOT}/remote_scripts/eval_zero_anchor/eval_zero_anchor_img/outputs/checkpoints/zero/weight.pt"
    if [[ -f "${ALT_ZERO}" ]]; then
        ZERO_CKPT="${ALT_ZERO}"
    fi
fi

OUTPUT_IMAGES="${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/concat_images_100"
OUTPUT_METRICS="${REPO_ROOT}/remote_scripts/eval_zero_anchor/outputs/eval_metrics_100"

echo "=========================================================="
echo " 100-Image Benchmark Evaluation: Art Anchor vs Zero Anchor"
echo " Working directory: ${REPO_ROOT}"
echo " GPU ID:            ${GPU_ID}"
echo " Python binary:     ${PYTHON_BIN}"
echo " Sample Count:      ${COUNT} erase + ${COUNT} retain (${COUNT}x2 = $((COUNT * 2)) total)"
echo " Diffusion Steps:   ${STEPS} steps (DPM-Solver)"
echo " Retain CSV:        ${STYLE_CSV}"
echo " Art Checkpoint:    ${ART_CKPT}"
echo " Zero Checkpoint:   ${ZERO_CKPT}"
echo " Images Output:     ${OUTPUT_IMAGES}"
echo " Metrics Output:    ${OUTPUT_METRICS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Verify Checkpoints
if [[ ! -f "${ART_CKPT}" || ! -f "${ZERO_CKPT}" ]]; then
    echo "ERROR: Required checkpoints not found!"
    echo "  Art Checkpoint:  ${ART_CKPT} (exists: $([ -f "${ART_CKPT}" ] && echo yes || echo no))"
    echo "  Zero Checkpoint: ${ZERO_CKPT} (exists: $([ -f "${ZERO_CKPT}" ] && echo yes || echo no))"
    echo "Please train them first using run_all.sh or train_erase_null.py."
    exit 1
fi

# -------------------------------------------------------------
# Step 1: Inference & 3-Way Concatenation (200 images total)
# -------------------------------------------------------------
echo ""
echo "=== Step 1: Generating 3-Way Comparison Images (${COUNT} erase + ${COUNT} retain) ==="

SKIP_FLAG=""
if [[ "${SKIP_EXISTING}" == "1" ]]; then
    SKIP_FLAG="--skip_existing"
fi

PREVIEW_SCRIPT="${SCRIPT_DIR}/eval_zero_anchor_img/preview_concat.py"
if [[ ! -f "${PREVIEW_SCRIPT}" ]]; then
    PREVIEW_SCRIPT="${SCRIPT_DIR}/preview_concat.py"
fi

"${PYTHON_BIN}" "${PREVIEW_SCRIPT}" \
    --art_ckpt "${ART_CKPT}" \
    --zero_ckpt "${ZERO_CKPT}" \
    --style_csv "${STYLE_CSV}" \
    --output_dir "${OUTPUT_IMAGES}" \
    --count "${COUNT}" \
    --steps "${STEPS}" \
    --prompt_mode "canonical_5" \
    --device "cuda" \
    ${SKIP_FLAG}

# -------------------------------------------------------------
# Step 2: Compute CLIP Metrics
# -------------------------------------------------------------
echo ""
echo "=== Step 2: Evaluating CLIP Metrics on Generated Images ==="

"${PYTHON_BIN}" "${SCRIPT_DIR}/eval_metrics/compute_quick_metrics.py" \
    --image_dir "${OUTPUT_IMAGES}" \
    --output_dir "${OUTPUT_METRICS}" \
    --style_csv "${STYLE_CSV}" \
    --prompt_mode "canonical_5" \
    --device "cuda"

echo ""
echo "=========================================================="
echo " 100-Image Evaluation Complete!"
echo " Results Summary:"
echo "   Summary Table:  ${OUTPUT_METRICS}/summary_table.txt"
echo "   Metrics JSON:   ${OUTPUT_METRICS}/summary_metrics.json"
echo "   Per-image CSV:  ${OUTPUT_METRICS}/per_image_metrics.csv"
echo "   Image outputs:  ${OUTPUT_IMAGES}"
echo "=========================================================="
echo ""
echo "To download all 200 comparison images to your laptop, run from your local terminal:"
echo ""
echo "OUT_DIR=\"remote_scripts/eval_zero_anchor/outputs/results_100_\$(date +%Y%m%d_%H%M%S)\""
echo "mkdir -p \"\${OUT_DIR}\" && \\"
echo "ssh -p 26546 root@202.122.49.242 \\"
echo "  \"tar -C ${OUTPUT_IMAGES} \\"
echo "   --exclude='*.safetensors' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.bin' \\"
echo "   -cf - .\" | tar -C \"\${OUT_DIR}\" -xf -"
echo "=========================================================="

