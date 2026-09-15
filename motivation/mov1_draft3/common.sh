#!/usr/bin/env bash

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${WORKFLOW_DIR}/../.." && pwd)}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname -- "${REPO_ROOT}")}"
WORKFLOW_CONFIG="${WORKFLOW_CONFIG:-${WORKFLOW_DIR}/workflow.yaml}"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"
WORKFLOW_CONFIG_LOADER="${WORKFLOW_DIR}/workflow_config.py"
TRAIN_CONFIG="${WORKFLOW_DIR}/train_config.yaml"

export REPO_ROOT WORKSPACE_DIR

[[ -x "${BOOTSTRAP_PYTHON}" ]] || {
    printf 'Configured Vast AI Python is unavailable: %s\n' "${BOOTSTRAP_PYTHON}" >&2
    printf 'Use a PyTorch image or set PYTHON_BIN explicitly.\n' >&2
    return 1
}

load_workflow_config() {
    if ! "${BOOTSTRAP_PYTHON}" -c 'import yaml' >/dev/null 2>&1; then
        "${BOOTSTRAP_PYTHON}" -m pip install PyYAML
    fi
    eval "$("${BOOTSTRAP_PYTHON}" "${WORKFLOW_CONFIG_LOADER}" \
        --config "${WORKFLOW_CONFIG}" export)"
    read -r -a METHODS <<< "${METHODS_RAW}"
    read -r -a RESIDUAL_SCALES <<< "${RESIDUAL_SCALES_RAW}"
}

require_file() {
    [[ -f "$1" ]] || { printf 'Missing file: %s\n' "$1" >&2; exit 1; }
}

require_directory() {
    [[ -d "$1" ]] || { printf 'Missing directory: %s\n' "$1" >&2; exit 1; }
}

scale_slug() {
    printf '%s' "$1" | tr '.' 'p'
}

anchor_mode_for_method() {
    case "$1" in
        legacy_full) printf 'legacy' ;;
        legacy_svd_rank30) printf 'norm_matched_truncated_svd_residual' ;;
        tgprs_rank30) printf 'target_global_pairwise_residual_subspace' ;;
        *) printf 'Unknown MOV1 method: %s\n' "$1" >&2; return 1 ;;
    esac
}

checkpoint_dir_for_run() {
    printf '%s/%s/scale_%s' "${CHECKPOINT_ROOT}" "$1" "$(scale_slug "$2")"
}

image_root_for_run() {
    printf '%s/%s/scale_%s' "${IMAGE_ROOT}" "$1" "$(scale_slug "$2")"
}

gcd_dir_for_run() {
    printf '%s/%s/scale_%s' "${GCD_OUTPUT_DIR}" "$1" "$(scale_slug "$2")"
}

count_pngs() {
    find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

file_sha256() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

target_concepts() {
    "${PYTHON_BIN}" - "${REPO_ROOT}/data/${BENCHMARK_NAME}.csv" <<'PY'
import csv
import sys

targets = []
with open(sys.argv[1], newline="", encoding="utf-8") as csv_file:
    for row in csv.DictReader(csv_file):
        concept = (row.get("concept") or "").strip()
        if row.get("type") == "erase" and concept and concept not in targets:
            targets.append(concept)
if len(targets) != 100:
    raise SystemExit(f"Expected 100 erase concepts, found {len(targets)}")
if any("," in concept for concept in targets):
    raise SystemExit("Target names containing commas are unsupported")
print(", ".join(targets))
PY
}

tensorflow_cuda_library_path() {
    local site_packages
    site_packages="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
    find "${site_packages}/nvidia" -type d -name lib -print | sort | paste -sd: -
}
