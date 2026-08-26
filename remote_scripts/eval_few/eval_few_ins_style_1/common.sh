#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"
WORKFLOW_CONFIG_LOADER="${WORKFLOW_DIR}/workflow_config.py"
FIELD_SEPARATOR=$'\x1f'

[[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Configured bootstrap Python is not executable: %s\n' \
        "${BOOTSTRAP_PYTHON}" >&2
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

export REPO_ROOT WORKSPACE_DIR WORKFLOW_CONFIG

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" export)"
    [[ -x "${PYTHON_BIN}" ]] || {
        printf 'Configured Python interpreter not found: %s\n' \
            "${PYTHON_BIN}" >&2
        return 1
    }
    read -r -a METHODS <<< "${METHODS_RAW}"
    read -r -a EDITED_METHODS <<< "${EDITED_METHODS_RAW}"
}

domain_rows() {
    "${PYTHON_BIN}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" domains
}

task_rows() {
    "${PYTHON_BIN}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" tasks
}

require_file() {
    [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

require_directory() {
    [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }
}

count_pngs() {
    if [[ ! -d "$1" ]]; then
        printf '0'
        return
    fi
    find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

join_by_comma_space() {
    local first=1 item
    for item in "$@"; do
        if [[ "${first}" == "1" ]]; then
            printf '%s' "${item}"
            first=0
        else
            printf ', %s' "${item}"
        fi
    done
}

clear_pngs_if_forced() {
    local image_dir="$1"
    if [[ "${FORCE_RESAMPLE:-0}" != "1" || ! -d "${image_dir}" ]]; then
        return
    fi
    case "${image_dir}" in
        "${OUTPUT_ROOT}"/*) ;;
        *)
            printf 'Refusing to clear images outside OUTPUT_ROOT: %s\n' \
                "${image_dir}" >&2
            exit 1
            ;;
    esac
    find "${image_dir}" -maxdepth 1 -type f -name '*.png' -delete
}

assert_png_count() {
    local image_dir="$1"
    local expected="$2"
    local label="$3"
    local actual
    actual="$(count_pngs "${image_dir}")"
    [[ "${actual}" == "${expected}" ]] || {
        printf '%s expected %s PNG files in %s, found %s\n' \
            "${label}" "${expected}" "${image_dir}" "${actual}" >&2
        exit 1
    }
}

train_config_for_method() {
    case "$1" in
        legacy)
            printf '%s/train_config_legacy.yaml' "${WORKFLOW_DIR}"
            ;;
        target_global_pairwise_residual_subspace)
            printf '%s/train_config_target_global_pairwise_residual_subspace.yaml' \
                "${WORKFLOW_DIR}"
            ;;
        *)
            printf 'No training config for method: %s\n' "$1" >&2
            return 1
            ;;
    esac
}

checkpoint_path_for_run() {
    printf '%s/%s/%s/weight.pt' "${CHECKPOINT_ROOT}" "$1" "$2"
}

few_image_dir_for_run() {
    local method="$1"
    local erase_type="$2"
    local task_id="$3"
    local content="$4"
    if [[ "${method}" == "original" ]]; then
        printf '%s/original/%s/shared/%s/original' \
            "${IMAGE_ROOT}" "${erase_type}" "${content}"
    else
        printf '%s/%s/%s/%s/%s/edit' \
            "${IMAGE_ROOT}" "${method}" "${erase_type}" \
            "${task_id}" "${content}"
    fi
}

mscoco_image_dir_for_run() {
    local method="$1"
    local task_id="$2"
    if [[ "${method}" == "original" ]]; then
        printf '%s/original/coco/original' "${MSCOCO_IMAGE_ROOT}"
    else
        printf '%s/%s/%s/coco/edit' \
            "${MSCOCO_IMAGE_ROOT}" "${method}" "${task_id}"
    fi
}
