#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/train_erase_null.py"
require_file "${REPO_ROOT}/sample.py"
require_file "${WORKFLOW_DIR}/train_config.yaml"
require_file "${BENCHMARK_CSV}"

mkdir -p "${OUTPUT_ROOT}/config" "${CHECKPOINT_DIR}" "${IMAGE_ROOT}" \
    "${GCD_ROOT}" "${SUMMARY_ROOT}" "${FIGURE_ROOT}"

"${PYTHON_BIN}" - "${BENCHMARK_CSV}" "${TARGET_A}" "${ANCHOR_B}" \
    "${RETAIN_CSV}" <<'PY'
import csv
import sys

source, target_a, anchor_b, destination = sys.argv[1:]
concepts = []
with open(source, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        concept = (row.get("concept") or "").strip()
        if concept and concept not in concepts:
            concepts.append(concept)
missing = [name for name in (target_a, anchor_b) if name not in concepts]
if missing:
    raise SystemExit(f"Celebrities missing from benchmark: {missing}")
retained = [name for name in concepts if name not in {target_a, anchor_b}]
with open(destination, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["concept"])
    writer.writeheader()
    writer.writerows({"concept": name} for name in retained)
print(f"Prepared {len(retained)} unrelated retain concepts")
PY

if [[ ! -d "${CE_EVAL_ROOT}/.git" ]]; then
    git clone --branch "${CE_EVAL_BRANCH}" \
        "${CE_EVAL_REPOSITORY}" "${CE_EVAL_ROOT}"
else
    printf 'CE-Eval checkout already exists: %s\n' "${CE_EVAL_ROOT}"
fi

printf 'Prepared A=%s, B=%s, expected images per identity/state=%s\n' \
    "${TARGET_A}" "${ANCHOR_B}" "${EXPECTED_IMAGES_PER_IDENTITY}"

