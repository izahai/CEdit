#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${OUTPUT_ROOT}/checkpoints"
require_directory "${GCD_OUTPUT_DIR}"
require_file "${GCD_OUTPUT_DIR}/summary.csv"

STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="${WORKSPACE_DIR}/CEdit-CE-Eval-eval-multi-1-${STAMP}"
ARCHIVE_PATH="${BUNDLE_DIR}.tar.gz"
mkdir -p "${BUNDLE_DIR}"
cp -R "${OUTPUT_ROOT}/checkpoints" "${BUNDLE_DIR}/checkpoints"
cp -R "${GCD_OUTPUT_DIR}" "${BUNDLE_DIR}/gcd"
cp "${WORKFLOW_DIR}/workflow.yaml" "${BUNDLE_DIR}/workflow.yaml"
cp "${WORKFLOW_DIR}/train_config_1.yaml" "${BUNDLE_DIR}/train_config_1.yaml"
cp "${WORKFLOW_DIR}/train_config_2.yaml" "${BUNDLE_DIR}/train_config_2.yaml"
tar -czf "${ARCHIVE_PATH}" -C "${WORKSPACE_DIR}" "$(basename -- "${BUNDLE_DIR}")"

printf 'Archive created: %s\n' "${ARCHIVE_PATH}"
printf 'Retrieve it through SCP or the Vast file browser.\n'
