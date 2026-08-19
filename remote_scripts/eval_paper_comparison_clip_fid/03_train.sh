#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

"${PYTHON_BIN}" - \
    "$(train_config_for_method legacy)" \
    "$(train_config_for_method target_global_pairwise_residual_subspace)" <<'PY'
import sys
import yaml

expected_configs = {
    sys.argv[1]: {
        "anchor_mode": "legacy",
        "anchor_concepts": ["person"],
        "params": "V",
        "aug_num": 10,
        "threshold": 1e-4,
        "retain_scale": 0.05,
        "disable_filter": True,
    },
    sys.argv[2]: {
        "anchor_mode": "target_global_pairwise_residual_subspace",
        "anchor_concepts": ["person"],
        "params": "V",
        "aug_num": 0,
        "threshold": 1e-4,
        "retain_scale": 0.05,
        "disable_filter": True,
        "residual_rank": 30,
        "residual_scale": 1.0,
    },
}
for path, expected in expected_configs.items():
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"Expected {key}={value!r} in {path}, found {config.get(key)!r}"
            )
PY

for method in legacy target_global_pairwise_residual_subspace; do
    train_config="$(train_config_for_method "${method}")"
    require_file "${train_config}"
    for benchmark_name in "${BENCHMARK_NAMES[@]}"; do
        benchmark_csv="$(benchmark_csv_for_name "${benchmark_name}")"
        checkpoint_dir="$(checkpoint_dir_for_run "${method}" "${benchmark_name}")"
        checkpoint_path="${checkpoint_dir}/weight.pt"
        require_file "${benchmark_csv}"
        target_concepts="$(
            target_concepts_for_benchmark "${benchmark_csv}" "${benchmark_name}"
        )"

        if [[ -f "${checkpoint_path}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
            printf '%s %s checkpoint exists; skipping: %s\n' \
                "${method}" "${benchmark_name}" "${checkpoint_path}"
            continue
        fi

        mkdir -p "${checkpoint_dir}"
        printf 'Training %s on %s\n' "${method}" "${benchmark_name}"
        USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" -u train_erase_null.py \
            --config "${train_config}" \
            --target_concepts "${target_concepts}" \
            --retain_path "${benchmark_csv}" \
            --save_path "${checkpoint_dir}"
        require_file "${checkpoint_path}"
        printf '%s %s training complete: %s\n' \
            "${method}" "${benchmark_name}" "${checkpoint_path}"
    done
done
