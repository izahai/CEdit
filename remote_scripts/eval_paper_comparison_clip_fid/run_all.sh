#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${WORKFLOW_DIR}/common.sh"

STAGES=(
    00_clone_repositories.sh
    01_setup_environment.sh
    02_generate_original.sh
    03_train.sh
    04_infer_edits.sh
    05_setup_ce_eval.sh
    06_eval.sh
    07_generate_mscoco.sh
    08_eval_clip_fid.sh
    09_summarize_results.sh
)

for stage in "${STAGES[@]}"; do
    printf '\nRunning %s\n' "${stage}"
    bash "${WORKFLOW_DIR}/${stage}"
done

printf '\nPaper comparison workflow complete.\n'
