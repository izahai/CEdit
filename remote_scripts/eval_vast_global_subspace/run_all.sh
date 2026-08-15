#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${WORKFLOW_DIR}/common.sh"

STAGES=(
    00_clone_repositories.sh
    01_setup_environment.sh
    02_train.sh
    03_infer.sh
    04_setup_ce_eval.sh
    05_eval.sh
    06_summarize_results.sh
)

for stage in "${STAGES[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nEvaluation workflow complete.\n'
