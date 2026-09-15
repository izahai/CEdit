#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

SUMMARY_CSV="${SUMMARY_DIR}/metrics.csv"
newer_gcd=""
if [[ -f "${SUMMARY_CSV}" ]]; then
    newer_gcd="$(find "${GCD_OUTPUT_DIR}" -type f -name '*.csv' -newer "${SUMMARY_CSV}" -print -quit 2>/dev/null || true)"
fi
if [[ -f "${SUMMARY_CSV}" && -z "${newer_gcd}" && "${FORCE_ANALYSIS:-0}" != "1" ]]; then
    printf 'Summary exists; skipping: %s\n' "${SUMMARY_CSV}"
    exit 0
fi
mkdir -p "${SUMMARY_DIR}"
"${PYTHON_BIN}" "${WORKFLOW_DIR}/summarize_results.py" \
    --config "${WORKFLOW_CONFIG}" \
    --rank-summary "${ANALYSIS_DIR}/rank_summary.csv" \
    --gcd-root "${GCD_OUTPUT_DIR}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --output "${SUMMARY_CSV}"
require_file "${SUMMARY_CSV}"
