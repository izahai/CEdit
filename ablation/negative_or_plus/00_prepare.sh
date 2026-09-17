#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/train_erase_null.py"
require_file "${REPO_ROOT}/sample2.py"
require_file "${WORKFLOW_DIR}/train_config_negative.yaml"
require_file "${WORKFLOW_DIR}/train_config_positive.yaml"
require_file "${WORKFLOW_DIR}/evaluate_by_GCD_workers.py"

if [[ ! -d "${CE_EVAL_ROOT}/.git" ]]; then
    git clone --branch "${CE_EVAL_BRANCH}" \
        "${CE_EVAL_REPOSITORY}" "${CE_EVAL_ROOT}"
else
    printf 'CE-Eval checkout already exists: %s\n' "${CE_EVAL_ROOT}"
fi

printf 'Repository preparation complete.\n'
