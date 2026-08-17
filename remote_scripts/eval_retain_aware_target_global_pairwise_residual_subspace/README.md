# Vast AI: Retain-Aware Target-Global Pairwise Residual Evaluation

This workflow trains and evaluates
`retain_aware_target_global_pairwise_residual_subspace` for the 10-, 50-, and
100-celebrity benchmarks. It evaluates only the new edited model—there are no
original or legacy runs. For each edited layer, raw residuals built from erase
targets, `""`, and `"person"` are projected through that layer's retain-low
projector before normalization and truncated SVD.

The training configuration matches
`eval_paper_comparison/train_config_target_global_pairwise_residual_subspace.yaml`
except for `anchor_mode` and the retain-low threshold, which is `0.00001`.

## 1. Create the Vast instance

Choose an RTX 5090 PyTorch image with CUDA 12.8 or newer and at least 50 GB of
disk. Vast normally opens the SSH shell inside tmux.

Set the new instance connection details on your local Mac:

```bash
export VAST_HOST='180.189.55.38'
export VAST_PORT='43995'
```

Upload the repository with the macOS-compatible rsync command:

```bash
rsync -az --progress \
  --exclude='.git/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  -e "ssh -p ${VAST_PORT}" \
  /Users/hainguyen/Repo/2026/ConceptErasure/Working/CEdit/ \
  "root@${VAST_HOST}:/workspace/CEdit/"
```

Keep the trailing `/` after `CEdit/`.

## 2. Connect and run

```bash
ssh -p 43995 -L 8080:localhost:8080 root@180.189.55.38
cd /workspace/CEdit
nvidia-smi
/venv/main/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 10'
bash remote_scripts/eval_retain_aware_target_global_pairwise_residual_subspace/run_all.sh
```

The workflow is resumable. Completed checkpoints, image sets, and evaluations
are skipped. Force an individual stage when needed:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_retain_aware_target_global_pairwise_residual_subspace/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_retain_aware_target_global_pairwise_residual_subspace/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/eval_retain_aware_target_global_pairwise_residual_subspace/05_eval.sh
```

GCD defaults to 8 CPU loader workers, batch size 32, and prefetch factor 2. A
5090 with sufficient CPU and RAM can be tuned without editing YAML:

```bash
GCD_NUM_WORKERS=16 GCD_BATCH_SIZE=64 GCD_PREFETCH_FACTOR=2 \
  FORCE_EVAL=1 \
  bash remote_scripts/eval_retain_aware_target_global_pairwise_residual_subspace/05_eval.sh
```

Use `GCD_NUM_WORKERS=0 GCD_BATCH_SIZE=1` for serial debugging.

## 3. Monitor and download

Open a second local terminal to monitor the GPU without polluting the tmux log:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" nvidia-smi dmon -s pucm
```

To track GPU utilization every second in a background process on the server:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'nohup nvidia-smi dmon -i 0 -s u -d 1 -o T \
   >> /workspace/gpu_sm.log 2>&1 & echo $! > /workspace/gpu_sm.pid'
```

View or stop the background monitor from the local machine:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'tail -f /workspace/gpu_sm.log'

ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'kill "$(cat /workspace/gpu_sm.pid)"'
```

Count generated images from the local machine:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'find /workspace/cedit_ce_eval_outputs_retain_aware_target_global_pairwise_residual_subspace/images -type f -name "*.png" 2>/dev/null | wc -l'
```

A complete run contains 3,000 images: 500 erase and 500 retain images for each
benchmark.

Outputs are stored under:

```text
/workspace/cedit_ce_eval_outputs_retain_aware_target_global_pairwise_residual_subspace/
├── checkpoints/<benchmark>/weight.pt
├── images/<benchmark>/<benchmark>/{erase,retain}/edit/*.png
└── gcd/
    ├── <benchmark>/*.csv
    └── summary.csv
```

Download the tmux pane log and final summary into the current local directory:

```bash
scp -P "${VAST_PORT}" "root@${VAST_HOST}:/workspace/tmux-log.log" .
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/cedit_ce_eval_outputs_retain_aware_target_global_pairwise_residual_subspace/gcd/summary.csv" \
  .
```

Do not run `tail -f /workspace/tmux-log.log` inside the logged tmux pane; that
would feed the log back into itself.
