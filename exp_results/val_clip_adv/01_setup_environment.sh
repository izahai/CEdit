#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access a CUDA GPU")
device = torch.device("cuda:0")
value = torch.ones((256, 256), device=device) @ torch.ones((256, 256), device=device)
print(
    "PyTorch:", torch.__version__,
    "CUDA:", torch.version.cuda,
    "GPU:", torch.cuda.get_device_name(0),
    "matmul:", value.device,
)
PY

mkdir -p "${OUTPUT_ROOT}"
available_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
required_kib=$((MINIMUM_FREE_DISK_GIB * 1024 * 1024))
if (( available_kib < required_kib )) && [[ "${ALLOW_LOW_DISK:-0}" != "1" ]]; then
    printf 'At least %s GiB free is recommended for profile %s; only %s GiB is available.\n' \
        "${MINIMUM_FREE_DISK_GIB}" "${WORKFLOW_PROFILE}" \
        "$((available_kib / 1024 / 1024))" >&2
    printf 'Set ALLOW_LOW_DISK=1 only if another output volume is configured.\n' >&2
    exit 1
fi

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
