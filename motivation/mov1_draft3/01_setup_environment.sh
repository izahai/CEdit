#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
capability = torch.cuda.get_device_capability()
cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
if capability[0] >= 10 and cuda_version < (12, 8):
    raise SystemExit(
        f"GPU capability {capability} requires CUDA 12.8+; found {torch.version.cuda}"
    )
x = torch.ones((256, 256), device="cuda")
print(
    "PyTorch:", torch.__version__,
    "CUDA:", torch.version.cuda,
    "GPU:", torch.cuda.get_device_name(0),
    "matmul:", (x @ x).device,
)
PY

"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install \
    matplotlib python-dotenv openpyxl scikit-image scikit-learn \
    opencv-python-headless "scipy<1.18" "tensorflow[and-cuda]==2.21.0"

if [[ ! -d "${CE_EVAL_ROOT}/.git" ]]; then
    git clone --branch "${CE_EVAL_BRANCH}" \
        "${CE_EVAL_REPOSITORY}" "${CE_EVAL_ROOT}"
else
    printf 'CE-Eval checkout already exists: %s\n' "${CE_EVAL_ROOT}"
fi

CELEB_DIR="${CE_EVAL_ROOT}/celeb-detection-oss"
RESOURCES_DIR="${CELEB_DIR}/examples/resources"
GCD_WEIGHTS="${RESOURCES_DIR}/face_recognition/best_model_states.pkl"
if [[ ! -f "${GCD_WEIGHTS}" ]]; then
    require_file "${CE_EVAL_ROOT}/run/download_resources_colab.sh"
    bash "${CE_EVAL_ROOT}/run/download_resources_colab.sh" "${CELEB_DIR}"
fi
require_file "${GCD_WEIGHTS}"
require_file "${RESOURCES_DIR}/face_recognition/labels.csv"

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
import matplotlib
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("TensorFlow cannot access a GPU")
x = tf.matmul(tf.ones((256, 256)), tf.ones((256, 256)))
if "GPU" not in x.device:
    raise SystemExit(f"TensorFlow matmul did not use the GPU: {x.device}")
print("TensorFlow:", tf.__version__, "matplotlib:", matplotlib.__version__)
PY

printf 'MOV1 environment setup complete.\n'
