#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
config_path="${1:-${repo_root}/configs/train.yaml}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
else
  python_bin="python3"
fi

if [[ $# -gt 0 ]]; then
  shift
fi

cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "${python_bin}" train_erase_null.py \
  --config "${config_path}" \
  "$@"
