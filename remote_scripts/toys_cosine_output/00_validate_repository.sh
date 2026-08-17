#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_executable "${PYTHON_BIN}"
require_file "${REPO_ROOT}/requirements.txt"
require_file "${REPO_ROOT}/src/residual_subspace.py"
require_file "${REPO_ROOT}/data/10_celebrity.csv"
require_file "${WORKFLOW_CONFIG}"
require_file "${WORKFLOW_DIR}/analyze_tgprs_output.py"

printf 'Repository validation complete: %s\n' "${REPO_ROOT}"
