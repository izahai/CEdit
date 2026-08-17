#!/usr/bin/env bash
set -Eeuo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_CONFIG="${WORKFLOW_DIR}/workflow_50_celebrity.yaml" \
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd -- "${WORKFLOW_DIR}/../../.." && pwd)/cedit_toys_cosine_output_50_celebrity}" \
bash "${WORKFLOW_DIR}/02_analyze.sh"

WORKFLOW_CONFIG="${WORKFLOW_DIR}/workflow_50_celebrity.yaml" \
OUTPUT_ROOT="${OUTPUT_ROOT:-$(cd -- "${WORKFLOW_DIR}/../../.." && pwd)/cedit_toys_cosine_output_50_celebrity}" \
bash "${WORKFLOW_DIR}/03_show_results.sh"
