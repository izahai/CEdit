#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}" 
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"

[[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Vast PyTorch interpreter not found: %s\n' "${BOOTSTRAP_PYTHON}" >&2
    printf 'Set PYTHON_BIN only if you intentionally use a different image.\n' >&2
    return 1
}
[[ -f "${WORKFLOW_CONFIG}" ]] || {
    printf 'Workflow config not found: %s\n' "${WORKFLOW_CONFIG}" >&2
    return 1
}

export REPO_ROOT WORKSPACE_DIR

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_DIR}/workflow_config.py" \
        --config "${WORKFLOW_CONFIG}" export)"
    [[ -x "${PYTHON_BIN}" ]] || {
        printf 'Configured Python interpreter not found: %s\n' "${PYTHON_BIN}" >&2
        return 1
    }
}

require_file() {
    [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

require_directory() {
    [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }
}

count_pngs() {
    find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}
