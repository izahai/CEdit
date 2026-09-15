#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SPECTRUM_CSV="${ANALYSIS_DIR}/residual_spectrum.csv"
RANK_CSV="${ANALYSIS_DIR}/rank_summary.csv"
analysis_stale=0
for input_path in \
    "${WORKFLOW_CONFIG}" \
    "${WORKFLOW_DIR}/analyze_residuals.py" \
    "${REPO_ROOT}/src/residual_subspace.py"; do
    if [[ ! -f "${SPECTRUM_CSV}" || "${input_path}" -nt "${SPECTRUM_CSV}" ]]; then
        analysis_stale=1
    fi
done
if [[ -f "${SPECTRUM_CSV}" && -f "${RANK_CSV}" && "${analysis_stale}" == "0" && "${FORCE_ANALYSIS:-0}" != "1" ]]; then
    printf 'Residual analysis already exists; skipping: %s\n' "${ANALYSIS_DIR}"
    exit 0
fi

mkdir -p "${ANALYSIS_DIR}" "${LOG_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_TF=0 TRANSFORMERS_NO_TF=1 \
"${PYTHON_BIN}" "${WORKFLOW_DIR}/analyze_residuals.py" \
    --config "${WORKFLOW_CONFIG}" \
    --output-dir "${ANALYSIS_DIR}" \
    2>&1 | tee "${LOG_DIR}/analyze_residuals.log"

require_file "${SPECTRUM_CSV}"
require_file "${RANK_CSV}"
require_file "${ANALYSIS_DIR}/analysis.json"
