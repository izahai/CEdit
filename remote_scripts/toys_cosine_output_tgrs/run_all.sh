#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for stage in \
    00_validate_repository.sh \
    01_setup_environment.sh \
    02_analyze.sh \
    03_show_results.sh; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nTGPRS output-cosine toy workflow complete.\n'
