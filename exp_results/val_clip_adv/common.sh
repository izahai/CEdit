#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"

if [[ "${BOOTSTRAP_PYTHON}" != */* ]]; then
    BOOTSTRAP_PYTHON="$(command -v "${BOOTSTRAP_PYTHON}" || true)"
fi
[[ -n "${BOOTSTRAP_PYTHON}" && -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Python interpreter is unavailable: %s\n' "${PYTHON_BIN:-python3}" >&2
    return 1
}
[[ -f "${WORKFLOW_CONFIG}" ]] || {
    printf 'Workflow config not found: %s\n' "${WORKFLOW_CONFIG}" >&2
    return 1
}

export REPO_ROOT WORKSPACE_DIR WORKFLOW_CONFIG

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_DIR}/workflow_config.py" \
        --config "${WORKFLOW_CONFIG}" export)"
    if [[ "${PYTHON_BIN}" != */* ]]; then
        PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
    fi
    [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
        printf 'Configured Python interpreter is unavailable.\n' >&2
        return 1
    }
    read -r -a METHODS <<< "${METHODS_RAW}"
    read -r -a EDITED_METHODS <<< "${EDITED_METHODS_RAW}"
    export PYTHON_BIN OUTPUT_ROOT CHECKPOINT_ROOT IMAGE_ROOT
    export MSCOCO_IMAGE_ROOT METRICS_DIR LOG_ROOT FID_CACHE_ROOT
    export LEARNED_ANCHOR_ROOT
}

require_file() {
    [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

count_pngs() {
    if [[ ! -d "$1" ]]; then
        printf '0'
        return
    fi
    find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
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

clear_pngs_if_stale() {
    local image_dir="$1"
    local dependency="$2"
    local stale
    if [[ ! -d "${image_dir}" || ! -f "${dependency}" ]]; then
        return
    fi
    stale="$(find "${image_dir}" -maxdepth 1 -type f -name '*.png' \
        ! -newer "${dependency}" -print -quit)"
    if [[ -z "${stale}" ]]; then
        return
    fi
    case "${image_dir}" in
        "${OUTPUT_ROOT}"/*) ;;
        *)
            printf 'Refusing to refresh images outside OUTPUT_ROOT: %s\n' \
                "${image_dir}" >&2
            exit 1
            ;;
    esac
    printf 'Refreshing stale generated images in %s\n' "${image_dir}"
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

method_train_config() {
    case "$1" in
        legacy)
            printf '%s/train_config_legacy.yaml' "${WORKFLOW_DIR}"
            ;;
        target_global_pairwise_residual_subspace)
            printf '%s/train_config_target_global_pairwise_residual_subspace.yaml' \
                "${WORKFLOW_DIR}"
            ;;
        clip_guided_learned_anchor)
            printf '%s/train_config_learned_anchor.yaml' "${WORKFLOW_DIR}"
            ;;
        *)
            printf 'No training config for method: %s\n' "$1" >&2
            return 1
            ;;
    esac
}

checkpoint_path() {
    printf '%s/%s/%s/weight.pt' "${CHECKPOINT_ROOT}" "$1" "${TASK_ID}"
}

few_image_dir() {
    local method="$1"
    local content="$2"
    if [[ "${method}" == "original" ]]; then
        printf '%s/original/%s/shared/%s/original' \
            "${IMAGE_ROOT}" "${ERASE_TYPE}" "${content}"
    else
        printf '%s/%s/%s/%s/%s/edit' \
            "${IMAGE_ROOT}" "${method}" "${ERASE_TYPE}" \
            "${TASK_ID}" "${content}"
    fi
}

mscoco_image_dir() {
    local method="$1"
    if [[ "${method}" == "original" ]]; then
        printf '%s/original/coco/original' "${MSCOCO_IMAGE_ROOT}"
    else
        printf '%s/%s/%s/coco/edit' \
            "${MSCOCO_IMAGE_ROOT}" "${method}" "${TASK_ID}"
    fi
}
