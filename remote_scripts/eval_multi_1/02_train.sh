#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

for config_name in "${CONFIG_NAMES[@]}"; do
    train_config="$(train_config_for_name "${config_name}")"
    require_file "${train_config}"

    "${PYTHON_BIN}" - "${train_config}" "${ANCHOR_MODE}" "${config_name}" <<'PY'
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
expected_aug_num = {"config_1": 0, "config_2": 10}[sys.argv[3]]
if config.get("aug_num") != expected_aug_num:
    raise SystemExit(
        f"Expected aug_num {expected_aug_num} for {sys.argv[3]} in {sys.argv[1]}"
    )
PY

    for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
        benchmark_csv="$(benchmark_csv_for_name "${benchmark_name}")"
        checkpoint_dir="$(checkpoint_dir_for_run "${config_name}" "${benchmark_name}")"
        checkpoint_path="${checkpoint_dir}/weight.pt"
        require_file "${benchmark_csv}"
        target_concepts="$(target_concepts_for_benchmark "${benchmark_csv}" "${benchmark_name}")"
        target_count="$(target_count_for_benchmark "${benchmark_name}")"
        requested_rank="$(configured_residual_rank "${train_config}")"
        effective_rank="${requested_rank}"
        if (( effective_rank > target_count )); then
            effective_rank="${target_count}"
            printf '%s/%s requests residual rank %s, but only %s target residuals exist; using the maximum feasible rank %s.\n' \
                "${config_name}" "${benchmark_name}" "${requested_rank}" \
                "${target_count}" "${effective_rank}"
        fi

        if [[ -f "${checkpoint_path}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
            printf '%s/%s checkpoint exists; skipping training: %s\n' \
                "${config_name}" "${benchmark_name}" "${checkpoint_path}"
            continue
        fi

        mkdir -p "${checkpoint_dir}"
        printf 'Training %s/%s\n' "${config_name}" "${benchmark_name}"
        USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u train_erase_null.py \
            --config "${train_config}" \
            --target_concepts "${target_concepts}" \
            --retain_path "${benchmark_csv}" \
            --residual_rank "${effective_rank}" \
            --save_path "${checkpoint_dir}"
        require_file "${checkpoint_path}"
        printf '%s/%s training complete: %s\n' \
            "${config_name}" "${benchmark_name}" "${checkpoint_path}"
    done
done
