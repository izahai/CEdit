#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install \
    python-dotenv openpyxl scikit-image scikit-learn matplotlib \
    opencv-python-headless "tensorflow[and-cuda]==2.21.0"

TF_CUDA_LIBRARY_PATH="$(tensorflow_cuda_library_path)"
PTXAS_PATH="$(find "$("${PYTHON_BIN}" -c \
    'import site; print(site.getsitepackages()[0])')/nvidia" \
    -name ptxas -type f -print -quit)"
[[ -n "${PTXAS_PATH}" ]] || {
    printf 'TensorFlow CUDA ptxas was not installed.\n' >&2
    exit 1
}
ln -sf "${PTXAS_PATH}" "$(dirname -- "${PYTHON_BIN}")/ptxas"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
LD_LIBRARY_PATH="${TF_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
"${PYTHON_BIN}" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("TensorFlow cannot see a GPU")
print("TensorFlow:", tf.__version__, "GPUs:", gpus)
PY
