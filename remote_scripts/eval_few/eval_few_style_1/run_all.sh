#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${WORKFLOW_DIR}/common.sh"

STAGES=(
    00_validate.sh
    01_setup_environment.sh
    02_generate_original.sh
    03_train.sh
    04_generate_edits.sh
    05_generate_mscoco.sh
    06_evaluate.sh
)

for stage in "${STAGES[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nFew-concept style comparison workflow complete.\n'
