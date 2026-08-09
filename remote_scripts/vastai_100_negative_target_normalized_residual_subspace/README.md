# Vast AI: 100 Celebrity Negative-Target Normalized Residual Subspace

This workflow trains, samples, evaluates, and summarizes
`negative_target_normalized_residual_subspace` for subspace ranks
`k = 10, 20, ..., 100` on the 100-celebrity benchmark.

Each rank is isolated under its own checkpoint, image, and CE-Eval output
directory, so the workflow can resume independently per `k`.

## Server setup

The expected layout on Vast AI is:

```text
/workspace/
├── CEdit/
└── CE-Eval/
```

`CEdit` must be cloned first because this workflow is stored inside that
checkout. Script `00_clone_repositories.sh` then clones `CE-Eval` as its sibling
under `/workspace/CE-Eval`.

On a fresh server:

```bash
cd /workspace
git clone --branch main https://github.com/izahai/CEdit.git CEdit
```

From the local machine, sync the working code into the checkout. Run this from
the local CEdit repository:

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='CE-Eval/' \
  --exclude='__pycache__/' \
  --exclude='logs/' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  -e "ssh -p 38410" \
  /Users/hainguyen/Repo/2026/ConceptErasure/Working/CEdit/ \
  root@182.224.239.168:/workspace/CEdit/
```

## Save log from tmux 
```bash
ssh -p 38410 root@182.224.239.168 \
  "tmux pipe-pane -t ssh_tmux:0.0 -o 'cat >> /workspace/train.log'"
```  

If the server, SSH port, username, or local path differs, update those values.
Keep `/workspace/CEdit/.git` from the server-side clone; the workflow checks
that it exists before cloning `CE-Eval`.

## Run the workflow

Run these commands from `/workspace/CEdit` on Vast AI:

```bash
cd /workspace/CEdit

bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/00_clone_repositories.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/01_setup_environment.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/02_train.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/03_infer.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/04_setup_ce_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/05_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/06_summarize_results.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/07_bundle_results.sh
```

The sweep uses:

```text
k = 10, 20, 30, ..., 100
```

`01_setup_environment.sh` installs TensorFlow GPU support with
`tensorflow[and-cuda]==2.21.0`, configures CUDA library paths, and validates
that TensorFlow can execute on the GPU. `04_setup_ce_eval.sh` downloads the
CE-Eval model resources if they are missing.

The final comparison table is saved as `gcd/summary.csv` under the configured
output root. Use `FORCE_RETRAIN=1`, `FORCE_RESAMPLE=1`, or `FORCE_EVAL=1` to
rerun completed stages. Override `GPU_ID`, `PYTHON_BIN`, `OUTPUT_ROOT`, or
`CE_EVAL_ROOT` through the environment when needed.

## Resume or rerun

Each `k` has independent output paths. If a run stops, rerun the same command;
completed checkpoints, image sets, and evaluation CSVs are skipped by default.

```bash
FORCE_RETRAIN=1 bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/05_eval.sh
```

The default output root is:

```text
/workspace/cedit_ce_eval_outputs_100_celebrity_negative_target_normalized_residual_subspace/
```

Checkpoints, images, and GCD results are separated by `k_10`, `k_20`, ...,
`k_100`.

## Common checks

Verify the GPU environment before starting a long run:

```bash
nvidia-smi
/venv/main/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The CE-Eval wrapper must use `/venv/main/bin/python`, not the system
`/usr/bin/python3`. The evaluation script sets the venv path and CUDA library
path automatically. On an RTX 50-series GPU, TensorFlow may print a warning
about JIT-compiling kernels from PTX during the first run; this is expected if
the wheel has no native kernel for the GPU architecture.
