#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/requirements.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
capability = torch.cuda.get_device_capability()
cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
if capability[0] >= 10 and cuda_version < (12, 8):
    raise SystemExit(
        f"GPU capability {capability} requires a CUDA 12.8+ PyTorch wheel; "
        f"found CUDA {torch.version.cuda}"
    )
x = torch.ones((256, 256), device="cuda")
y = x @ x
print(
    "PyTorch:", torch.__version__,
    "CUDA:", torch.version.cuda,
    "GPU:", torch.cuda.get_device_name(0),
    "capability:", capability,
    "matmul:", y.device,
)
PY

"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install \
    python-dotenv openpyxl scikit-image scikit-learn \
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
x = tf.matmul(tf.ones((256, 256)), tf.ones((256, 256)))
if "GPU" not in x.device:
    raise SystemExit(f"TensorFlow matmul did not use the GPU: {x.device}")
print("TensorFlow:", tf.__version__, "GPUs:", gpus, "matmul:", x.device)
PY

printf 'Environment setup complete.\n'
