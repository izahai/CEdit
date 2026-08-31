#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/train_erase_null.py"
require_file "${REPO_ROOT}/sample.py"
require_file "${REPO_ROOT}/sample2.py"
require_file "${REPO_ROOT}/data/instance.csv"
require_file "${REPO_ROOT}/data/mscoco.csv"
require_file "$(train_config_for_method target_global_pairwise_residual_subspace)"
require_file "${WORKFLOW_DIR}/evaluate_clip_fid.py"

"${PYTHON_BIN}" "${WORKFLOW_CONFIG_LOADER}" \
    --config "${WORKFLOW_CONFIG}" validate

"${PYTHON_BIN}" - \
    "$(train_config_for_method target_global_pairwise_residual_subspace)" <<'PY'
import sys

import yaml

expected = {
    sys.argv[1]: {
        "anchor_mode": "target_global_pairwise_residual_subspace",
        "params": "V",
        "aug_num": 0,
        "threshold": 0.2,
        "retain_scale": 1.0,
        "disable_filter": False,
        "residual_rank": 30,
        "residual_scale": 1.0,
    },
}
for path, fields in expected.items():
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    for key, value in fields.items():
        if config.get(key) != value:
            raise SystemExit(
                f"Expected {key}={value!r} in {path}, found {config.get(key)!r}"
            )
PY

counts="$("${PYTHON_BIN}" "${WORKFLOW_CONFIG_LOADER}" \
    --config "${WORKFLOW_CONFIG}" counts)"
printf 'Expected generated image counts: %s\n' "${counts}"
printf 'Workflow validation complete.\n'
