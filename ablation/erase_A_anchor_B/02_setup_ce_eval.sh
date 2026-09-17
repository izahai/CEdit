#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss"
RESNET_FILE="${CELEB_DIR}/model_training/helpers/resnet_model.py"
require_file "${RESNET_FILE}"

"${PYTHON_BIN}" - "${RESNET_FILE}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "checkpoint = torch.load(self.weights_path)",
    "checkpoint = torch.load(self.weights_path, weights_only=False)",
)
text = text.replace(
    "checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage)",
    "checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage, weights_only=False)",
)
text = text.replace("nn.Softmax()(fc2_output)", "nn.Softmax(dim=1)(fc2_output)")
path.write_text(text)
PY

RESOURCES_DIR="${CELEB_DIR}/examples/resources"
if [[ ! -f "${RESOURCES_DIR}/face_recognition/labels.csv" || \
      ! -f "${RESOURCES_DIR}/face_recognition/best_model_states.pkl" ]]; then
    bash "${CE_EVAL_ROOT}/run/download_resources_colab.sh" "${CELEB_DIR}"
fi
require_file "${RESOURCES_DIR}/face_recognition/labels.csv"
require_file "${RESOURCES_DIR}/face_recognition/best_model_states.pkl"

"${PYTHON_BIN}" - "${RESOURCES_DIR}/face_recognition/labels.csv" \
    "${TARGET_A}" "${ANCHOR_B}" <<'PY'
import csv
import re
import sys

def normalize(value):
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()

path, target_a, anchor_b = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as file:
    labels = [row[0].split("_[", 1)[0].replace("_", " ") for row in list(csv.reader(file))[1:]]
normalized = {normalize(label) for label in labels}
missing = [name for name in (target_a, anchor_b) if normalize(name) not in normalized]
if missing:
    raise SystemExit(f"Celebrities missing from GCD labels: {missing}")
print(f"GCD recognizes both requested identities among {len(labels)} classes")
PY

