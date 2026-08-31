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

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" \
        -r "${REPO_ROOT}/requirements.txt" "scipy<1.18"
else
    "${PYTHON_BIN}" -m pip install \
        -r "${REPO_ROOT}/requirements.txt" "scipy<1.18"
fi

"${PYTHON_BIN}" - <<'PY'
import diffusers
import scipy
import torch_fidelity
import transformers

print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("scipy", scipy.__version__)
print("torch_fidelity", torch_fidelity.__version__)
PY

printf 'Environment setup complete.\n'
