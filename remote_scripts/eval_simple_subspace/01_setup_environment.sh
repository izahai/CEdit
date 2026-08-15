#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

require_file "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
"${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install python-dotenv openpyxl scikit-image scikit-learn opencv-python-headless "tensorflow[and-cuda]==2.21.0"

TF_CUDA_LIBRARY_PATH="$(tensorflow_cuda_library_path)"
PTXAS_PATH="$(find "$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name ptxas -type f -print -quit)"
[[ -n "${PTXAS_PATH}" ]] || { printf 'TensorFlow CUDA ptxas was not installed.\n' >&2; exit 1; }
ln -sf "${PTXAS_PATH}" "$(dirname -- "${PYTHON_BIN}")/ptxas"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
LD_LIBRARY_PATH="${TF_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
"${PYTHON_BIN}" -c 'import tensorflow as tf; gpus = tf.config.list_physical_devices("GPU"); assert gpus, "TensorFlow cannot see a GPU"; x = tf.matmul(tf.ones((256, 256)), tf.ones((256, 256))); assert "GPU" in x.device, x.device; print("TensorFlow:", tf.__version__, "GPUs:", gpus, "matmul:", x.device)'

printf 'Environment setup complete.\n'
