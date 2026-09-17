#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}" 
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"

export REPO_ROOT WORKSPACE_DIR

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_DIR}/workflow_config.py" \
        --config "${WORKFLOW_CONFIG}" export)"
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

identity_slug() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import re
import sys
print(re.sub(r"[^A-Za-z0-9]+", "_", sys.argv[1]).strip("_").lower())
PY
}

run_slug() {
    printf '%s_to_%s' "$(identity_slug "${TARGET_A}")" "$(identity_slug "${ANCHOR_B}")"
}

image_dir() {
    printf '%s/%s/%s/%s' "${IMAGE_ROOT}" "$(run_slug)" "$1" "$2"
}

gcd_csv() {
    printf '%s/%s_%s.csv' "${GCD_ROOT}" "$1" "$(identity_slug "$2")"
}

tensorflow_cuda_library_path() {
    local site_packages
    site_packages="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
    find "${site_packages}/nvidia" -type d -name lib -print | sort | paste -sd: -
}

