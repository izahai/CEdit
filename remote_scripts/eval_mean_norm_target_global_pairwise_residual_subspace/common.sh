#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"
WORKFLOW_CONFIG_LOADER="${WORKFLOW_DIR}/workflow_config.py"

[[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Vast PyTorch interpreter not found: %s\n' "${BOOTSTRAP_PYTHON}" >&2
    printf 'Set PYTHON_BIN only if you intentionally use another image.\n' >&2
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

TMUX_LOG_PATH="${WORKSPACE_DIR}/tmux-log.log"
TMUX_LOG_FORMAT="raw-mean-norm-target-global-v1"
export TMUX_LOG_PATH

enable_tmux_logging() {
    if [[ -z "${TMUX:-}" ]]; then
        printf 'Not inside tmux; pane logging is unavailable. Expected log: %s\n' \
            "${TMUX_LOG_PATH}" >&2
        return 0
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        printf 'tmux is unavailable; pane logging cannot be enabled.\n' >&2
        return 0
    fi

    local pane_id pane_log_format log_command
    pane_id="${TMUX_PANE:-$(tmux display-message -p '#{pane_id}')}"
    pane_log_format="$(
        tmux show-options -pqv -t "${pane_id}" @cedit_tmux_log_format \
            2>/dev/null || true
    )"
    if [[ "${pane_log_format}" == "${TMUX_LOG_FORMAT}" ]]; then
        printf 'tmux pane logging is already active: %s\n' \
            "${TMUX_LOG_PATH}"
        return 0
    fi

    printf -v log_command 'cat >> %q' "${TMUX_LOG_PATH}"
    tmux pipe-pane -t "${pane_id}" "${log_command}"
    tmux set-option -p -t "${pane_id}" \
        @cedit_tmux_log_format "${TMUX_LOG_FORMAT}"
    printf 'tmux pane logging enabled: %s\n' "${TMUX_LOG_PATH}"
}

enable_tmux_logging

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" export)"
    [[ -x "${PYTHON_BIN}" ]] || {
        printf 'Configured Python interpreter not found: %s\n' "${PYTHON_BIN}" >&2
        return 1
    }
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

benchmark_csv_for_name() {
    printf '%s/data/%s.csv' "${REPO_ROOT}" "$1"
}

checkpoint_dir_for_run() {
    printf '%s/checkpoints/%s' "${OUTPUT_ROOT}" "$1"
}

image_root_for_run() {
    printf '%s/%s' "${IMAGE_ROOT}" "$1"
}

gcd_output_dir_for_run() {
    printf '%s/%s' "${GCD_OUTPUT_DIR}" "$1"
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

configured_residual_rank() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
print(config["residual_rank"])
PY
}

tensorflow_cuda_library_path() {
    local site_packages
    site_packages="$("${PYTHON_BIN}" -c \
        'import site; print(site.getsitepackages()[0])')"
    find "${site_packages}/nvidia" -type d -name lib -print | sort | paste -sd: -
}
