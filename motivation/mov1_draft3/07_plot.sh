#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

PNG_PATH="${FIGURE_DIR}/common_anchor_rank_tradeoff.png"
PDF_PATH="${FIGURE_DIR}/common_anchor_rank_tradeoff.pdf"
inputs_newer=0
for input_path in \
    "${ANALYSIS_DIR}/residual_spectrum.csv" \
    "${ANALYSIS_DIR}/rank_summary.csv" \
    "${SUMMARY_DIR}/metrics.csv"; do
    if [[ ! -f "${PDF_PATH}" || "${input_path}" -nt "${PDF_PATH}" ]]; then
        inputs_newer=1
    fi
done
if [[ -f "${PNG_PATH}" && -f "${PDF_PATH}" && "${inputs_newer}" == "0" && "${FORCE_PLOT:-0}" != "1" ]]; then
    printf 'Paper figure exists; skipping: %s\n' "${FIGURE_DIR}"
    exit 0
fi
mkdir -p "${FIGURE_DIR}"
"${PYTHON_BIN}" "${WORKFLOW_DIR}/plot_figure.py" \
    --spectrum "${ANALYSIS_DIR}/residual_spectrum.csv" \
    --rank-summary "${ANALYSIS_DIR}/rank_summary.csv" \
    --metrics "${SUMMARY_DIR}/metrics.csv" \
    --output-dir "${FIGURE_DIR}"
require_file "${PNG_PATH}"
require_file "${PDF_PATH}"
