#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_executable "${PYTHON_BIN}"
require_file "${REPO_ROOT}/data/50_celebrity.csv"
require_file "${WORKFLOW_CONFIG}"
require_file "${WORKFLOW_DIR}/analyze_target_pairwise_cosine.py"

printf 'Repository validation complete: %s\n' "${REPO_ROOT}"
