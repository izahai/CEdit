#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${BENCHMARK_CSV}"
require_file "${TRAIN_CONFIG}"
cd -- "${REPO_ROOT}"

"${PYTHON_BIN}" - "${TRAIN_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

targets = config.get("target_concepts", [])
if len(targets) != 100:
    raise SystemExit(f"Expected 100 target concepts in train config, got {len(targets)}")
if config.get("anchor_mode") != "smallest_cosine_subspace":
    raise SystemExit("Train config must use anchor_mode: smallest_cosine_subspace")
PY

if [[ -f "${CHECKPOINT_PATH}" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    printf 'Checkpoint exists; skipping training: %s\n' "${CHECKPOINT_PATH}"
    exit 0
fi

mkdir -p "${CHECKPOINT_DIR}"
export CHECKPOINT_DIR
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u train_erase_null.py \
    --config "${TRAIN_CONFIG}"

require_file "${CHECKPOINT_PATH}"
printf 'Training complete: %s\n' "${CHECKPOINT_PATH}"
