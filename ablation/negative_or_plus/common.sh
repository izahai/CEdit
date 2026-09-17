#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"
WORKFLOW_CONFIG_LOADER="${WORKFLOW_DIR}/workflow_config.py"

[[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Configured Python interpreter not found: %s\n' \
        "${BOOTSTRAP_PYTHON}" >&2
    printf 'Set PYTHON_BIN when using a different GPU image.\n' >&2
    return 1
}
[[ -f "${WORKFLOW_CONFIG}" ]] || {
    printf 'Workflow config not found: %s\n' "${WORKFLOW_CONFIG}" >&2
    return 1
}
[[ -f "${WORKFLOW_CONFIG_LOADER}" ]] || {
    printf 'Workflow config loader not found: %s\n' \
        "${WORKFLOW_CONFIG_LOADER}" >&2
    return 1
}

export REPO_ROOT WORKSPACE_DIR

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${BOOTSTRAP_PYTHON}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" export)"
    [[ -x "${PYTHON_BIN}" ]] || {
        printf 'Configured Python interpreter not found: %s\n' \
            "${PYTHON_BIN}" >&2
        return 1
    }
    read -r -a METHODS <<< "${METHODS_RAW}"
    read -r -a BENCHMARK_NAMES <<< "${BENCHMARK_NAMES_RAW}"
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

projection_direction_for_method() {
    case "$1" in
        negative_projection)
            printf 'negative'
            ;;
        positive_projection)
            printf 'positive'
            ;;
        *)
            printf 'Unknown projection method: %s\n' "$1" >&2
            return 1
            ;;
    esac
}

train_config_for_method() {
    printf '%s/train_config_%s.yaml' \
        "${WORKFLOW_DIR}" "$(projection_direction_for_method "$1")"
}

benchmark_csv_for_name() {
    printf '%s/data/%s.csv' "${REPO_ROOT}" "$1"
}

checkpoint_dir_for_run() {
    printf '%s/checkpoints/%s/%s' "${OUTPUT_ROOT}" "$1" "$2"
}

image_root_for_run() {
    printf '%s/%s/%s' "${IMAGE_ROOT}" "$1" "$2"
}

gcd_output_dir_for_run() {
    printf '%s/%s/%s' "${GCD_OUTPUT_DIR}" "$1" "$2"
}

target_concepts_for_benchmark() {
    "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import csv
import sys

path, benchmark = sys.argv[1:]
try:
    expected = int(benchmark.removesuffix("_celebrity"))
except ValueError as error:
    raise SystemExit(f"Cannot infer target count from benchmark: {benchmark}") from error

targets = []
with open(path, newline="", encoding="utf-8") as csv_file:
    for row in csv.DictReader(csv_file):
        concept = (row.get("concept") or "").strip()
        if row.get("type") == "erase" and concept and concept not in targets:
            targets.append(concept)

if len(targets) != expected:
    raise SystemExit(
        f"Expected {expected} erase concepts in {path}, found {len(targets)}"
    )
if any("," in target for target in targets):
    raise SystemExit("Target concept names cannot contain commas")
print(", ".join(targets))
PY
}

tensorflow_cuda_library_path() {
    local site_packages
    site_packages="$("${PYTHON_BIN}" -c \
        'import site; print(site.getsitepackages()[0])')"
    find "${site_packages}/nvidia" -type d -name lib -print | sort | paste -sd: -
}
