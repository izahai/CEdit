#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

train_config="${WORKFLOW_DIR}/train_config.yaml"
require_file "${train_config}"

"${PYTHON_BIN}" - "${train_config}" "${ANCHOR_MODE}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
if config.get("anchor_mode") != sys.argv[2]:
    raise SystemExit(f"Unexpected anchor_mode in {sys.argv[1]}")
if config.get("residual_rank") != 30:
    raise SystemExit(f"Expected residual_rank 30 in {sys.argv[1]}")
if config.get("threshold") != 1e-4:
    raise SystemExit(f"Expected threshold 1e-4 in {sys.argv[1]}")
if config.get("retain_scale") != 0.05:
    raise SystemExit(f"Expected retain_scale 0.05 in {sys.argv[1]}")
if config.get("aug_num") != 0:
    raise SystemExit(f"Expected aug_num 0 in {sys.argv[1]}")
PY

for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
    benchmark_csv="$(benchmark_csv_for_name "${benchmark_name}")"
    checkpoint_dir="$(checkpoint_dir_for_run "${benchmark_name}")"
    checkpoint_path="${checkpoint_dir}/weight.pt"
    require_file "${benchmark_csv}"
    target_concepts="$(target_concepts_for_benchmark "${benchmark_csv}" "${benchmark_name}")"
    requested_rank="$(configured_residual_rank "${train_config}")"

    if [[ -f "${checkpoint_path}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
        printf '%s checkpoint exists; skipping training: %s\n' \
            "${benchmark_name}" "${checkpoint_path}"
        continue
    fi

    mkdir -p "${checkpoint_dir}"
    printf 'Training %s with global residual rank %s\n' \
        "${benchmark_name}" "${requested_rank}"
    USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" -u train_erase_null.py \
        --config "${train_config}" \
        --target_concepts "${target_concepts}" \
        --retain_path "${benchmark_csv}" \
        --subspace_concepts_path "${benchmark_csv}" \
        --save_path "${checkpoint_dir}"
    require_file "${checkpoint_path}"
    printf '%s training complete: %s\n' \
        "${benchmark_name}" "${checkpoint_path}"
done
