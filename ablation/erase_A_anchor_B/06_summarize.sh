#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

if [[ -f "${SUMMARY_ROOT}/four_points.csv" && \
      -f "${FIGURE_ROOT}/erase_A_anchor_B_probe.pdf" && \
      "${FORCE_SUMMARY:-0}" != "1" ]]; then
    printf 'Summary and figure exist; skipping. Set FORCE_SUMMARY=1 to rebuild.\n'
    exit 0
fi

"${PYTHON_BIN}" "${WORKFLOW_DIR}/summarize_and_plot.py" \
    --gcd-root "${GCD_ROOT}" \
    --summary-root "${SUMMARY_ROOT}" \
    --figure-root "${FIGURE_ROOT}" \
    --target-a "${TARGET_A}" \
    --anchor-b "${ANCHOR_B}" \
    --expected-count "${EXPECTED_IMAGES_PER_IDENTITY}"

