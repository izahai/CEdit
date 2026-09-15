#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

STAGES=(
    00_validate_repository.sh
    01_setup_environment.sh
    02_analyze_residuals.sh
    03_train.sh
    04_infer.sh
    05_evaluate.sh
    06_summarize.sh
    07_plot.sh
)

for stage in "${STAGES[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nMOV1 workflow complete.\n'

