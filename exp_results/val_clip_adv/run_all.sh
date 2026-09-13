#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stages=(
    00_validate.sh
    01_setup_environment.sh
    02_learn_anchor.sh
    03_generate_original.sh
    04_train.sh
    05_generate_edits.sh
    06_generate_mscoco.sh
    07_evaluate.sh
)

for stage in "${stages[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done
printf '\nVan Gogh learned-anchor comparison workflow complete.\n'
