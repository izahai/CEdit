#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

for path in \
    "${REPO_ROOT}/learn_anchor.py" \
    "${REPO_ROOT}/train_erase_null.py" \
    "${REPO_ROOT}/sample.py" \
    "${REPO_ROOT}/sample2.py" \
    "${REPO_ROOT}/data/style.csv" \
    "${REPO_ROOT}/data/mscoco.csv" \
    "${WORKFLOW_DIR}/learn_anchor_config.yaml" \
    "${WORKFLOW_DIR}/train_config_legacy.yaml" \
    "${WORKFLOW_DIR}/train_config_target_global_pairwise_residual_subspace.yaml" \
    "${WORKFLOW_DIR}/train_config_learned_anchor.yaml" \
    "${WORKFLOW_DIR}/evaluate_clip_fid.py"; do
    require_file "${path}"
done

"${PYTHON_BIN}" "${WORKFLOW_DIR}/workflow_config.py" \
    --config "${WORKFLOW_CONFIG}" validate
counts="$("${PYTHON_BIN}" "${WORKFLOW_DIR}/workflow_config.py" \
    --config "${WORKFLOW_CONFIG}" counts)"
printf 'Expected image counts: %s\n' "${counts}"
printf 'Workflow validation complete.\n'
