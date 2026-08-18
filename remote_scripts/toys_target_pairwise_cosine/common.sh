#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/cedit_toys_target_pairwise_cosine_50_celebrity}"

export REPO_ROOT WORKSPACE_DIR WORKFLOW_CONFIG PYTHON_BIN GPU_ID OUTPUT_ROOT

require_file() {
    [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

require_executable() {
    [[ -x "$1" ]] || { printf 'Missing executable: %s\n' "$1" >&2; exit 1; }
}
