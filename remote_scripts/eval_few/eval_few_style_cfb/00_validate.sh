#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/train_erase_null.py"
require_file "${REPO_ROOT}/train_closed_form_backprop.py"
require_file "${REPO_ROOT}/sample.py"
require_file "${REPO_ROOT}/sample2.py"
require_file "${REPO_ROOT}/data/style.csv"
require_file "${REPO_ROOT}/data/mscoco.csv"
require_file "$(train_config_for_method legacy)"
require_file "$(train_config_for_method cfb)"
require_file "${WORKFLOW_DIR}/evaluate_clip_fid.py"

"${PYTHON_BIN}" "${WORKFLOW_CONFIG_LOADER}" \
    --config "${WORKFLOW_CONFIG}" validate

"${PYTHON_BIN}" - \
    "$(train_config_for_method legacy)" \
    "$(train_config_for_method cfb)" <<'PY'
import sys

import yaml

expected = {
    sys.argv[1]: {
        "anchor_mode": "legacy",
        "params": "V",
        "aug_num": 10,
        "threshold": 0.1,
        "retain_scale": 1.0,
        "disable_filter": False,
    },
    sys.argv[2]: {
        "anchor_mode": "legacy",
        "erase_style": True,
        "params": "V",
        "aug_num": 0,
        "retain_projection_rank": 150,
        "threshold": 0.1,
        "retain_scale": 100.0,
        "residual_scale": 1.0,
        "use_k2": False,
        "anchor_steps": 200,
        "anchor_lr": 0.01,
        "validation_samples": 1,
        "validation_interval": 9999,
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
