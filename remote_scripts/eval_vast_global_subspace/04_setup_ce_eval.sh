#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_directory "${CE_EVAL_ROOT}/celeb-detection-oss"
RESNET_FILE="${CE_EVAL_ROOT}/celeb-detection-oss/model_training/helpers/resnet_model.py"
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
print(f"Applied compatibility patch: {path}")
PY

RESOURCES_DIR="${CE_EVAL_ROOT}/celeb-detection-oss/examples/resources"
if [[ ! -f "${RESOURCES_DIR}/face_recognition/labels.csv" || ! -f "${RESOURCES_DIR}/face_recognition/best_model_states.pkl" ]]; then
    bash "${CE_EVAL_ROOT}/run/download_resources_colab.sh" "${CE_EVAL_ROOT}/celeb-detection-oss"
fi

require_file "${RESOURCES_DIR}/face_recognition/labels.csv"
require_file "${RESOURCES_DIR}/face_recognition/best_model_states.pkl"
printf 'CE-Eval resources ready: %s\n' "${RESOURCES_DIR}"
