#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
x = torch.ones((64, 64), device="cuda")
print(
    "PyTorch:", torch.__version__,
    "CUDA:", torch.version.cuda,
    "GPU:", torch.cuda.get_device_name(0),
    "matmul:", (x @ x).device,
)
PY

if ! "${PYTHON_BIN}" -c \
    'import diffusers, matplotlib, torch, transformers, yaml' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install \
        diffusers==0.32.2 transformers==4.48.0 accelerate matplotlib PyYAML
fi

printf 'Environment setup complete.\n'
