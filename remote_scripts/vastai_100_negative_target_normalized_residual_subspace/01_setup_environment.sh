#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install python-dotenv openpyxl scikit-image scikit-learn opencv-python-headless tensorflow-cpu

printf 'Environment setup complete.\n'

