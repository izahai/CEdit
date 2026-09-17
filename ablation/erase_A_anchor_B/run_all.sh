#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGES=(
    00_prepare.sh
    01_setup_environment.sh
    02_setup_ce_eval.sh
    03_train.sh
    04_generate.sh
    05_evaluate.sh
    06_summarize.sh
)

for stage in "${STAGES[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nA-to-B identity-transfer probe complete.\n'

