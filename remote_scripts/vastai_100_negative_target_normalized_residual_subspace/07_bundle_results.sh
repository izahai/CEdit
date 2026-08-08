#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${OUTPUT_ROOT}/checkpoints/${ANCHOR_MODE}"
require_directory "${GCD_OUTPUT_DIR}"

STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="${WORKSPACE_DIR}/CEdit-CE-Eval-100-negative-target-normalized-residual-subspace-${STAMP}"
ARCHIVE_PATH="${BUNDLE_DIR}.tar.gz"
mkdir -p "${BUNDLE_DIR}"
cp -R "${OUTPUT_ROOT}/checkpoints/${ANCHOR_MODE}" "${BUNDLE_DIR}/checkpoints"
cp -R "${GCD_OUTPUT_DIR}" "${BUNDLE_DIR}/gcd"
tar -czf "${ARCHIVE_PATH}" -C "${WORKSPACE_DIR}" "$(basename -- "${BUNDLE_DIR}")"

printf 'Archive created: %s\n' "${ARCHIVE_PATH}"
printf 'Retrieve it through SCP or the Vast file browser.\n'
