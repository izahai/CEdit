# Vast AI: Target-Global Pairwise Residual Evaluation

This workflow trains and evaluates `target_global_pairwise_residual_subspace`
for the 10, 50, and 100 celebrity benchmarks. The residual subspace uses only
erase targets, `""`, and `"person"`.

This is a single-method workflow. It does not generate or evaluate the original
model or the legacy SPEED method. Its training settings match
`eval_paper_comparison/train_config_target_global_pairwise_residual_subspace.yaml`.

## Quick retain-low rank inspection

On a fresh Vast PyTorch server, upload this repository as described below and
run the standalone inspection script:

```bash
cd /workspace/CEdit
bash remote_scripts/eval_target_global_pairwise_residual_subspace/inspect_retain_low_rank.sh
```

The script installs only the packages needed for inspection, downloads only the
Stable Diffusion tokenizer and text encoder, and prints the retain-low rank for
the 10-, 50-, and 100-celebrity retain sets. It does not train, sample, clone
CE-Eval, or run GCD. By default it reads `threshold` from `train_config.yaml`.
Override it without editing the config:

```bash
THRESHOLD=0.00001 \
  bash remote_scripts/eval_target_global_pairwise_residual_subspace/inspect_retain_low_rank.sh
```

## 1. Create the Vast instance

Choose an RTX 5090 PyTorch image with CUDA 12.8 or newer and at least 50 GB of
disk. Vast normally opens the SSH shell inside tmux.

Set the new instance connection details on your local Mac:

```bash
export VAST_HOST='175.121.93.64'
export VAST_PORT='45291'
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
ssh -p 45291 -L 8080:localhost:8080 root@175.121.93.64
cd /workspace/CEdit
nvidia-smi
/venv/main/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 10'
bash remote_scripts/eval_target_global_pairwise_residual_subspace/run_all.sh
```

The workflow is resumable. Completed checkpoints, image sets, and evaluations
are skipped. Force an individual stage when needed:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_target_global_pairwise_residual_subspace/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_target_global_pairwise_residual_subspace/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/eval_target_global_pairwise_residual_subspace/05_eval.sh
```

GCD defaults to 8 CPU loader workers, batch size 32, and prefetch factor 2. A
5090 with sufficient CPU and RAM can be tuned without editing YAML:

```bash
GCD_NUM_WORKERS=16 GCD_BATCH_SIZE=64 GCD_PREFETCH_FACTOR=2 \
  FORCE_EVAL=1 \
  bash remote_scripts/eval_target_global_pairwise_residual_subspace/05_eval.sh
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
  'find /workspace/cedit_ce_eval_outputs_target_global_pairwise_residual_subspace/images -type f -name "*.png" 2>/dev/null | wc -l'
```

A complete run contains 3,000 images: 500 erase and 500 retain images for each
benchmark.

Outputs are stored under:

```text
/workspace/cedit_ce_eval_outputs_target_global_pairwise_residual_subspace/
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
  "root@${VAST_HOST}:/workspace/cedit_ce_eval_outputs_target_global_pairwise_residual_subspace/gcd/summary.csv" \
  .
```

Do not run `tail -f /workspace/tmux-log.log` inside the logged tmux pane; that
would feed the log back into itself.

Retain-low threshold: 1e-04
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=0.0001 | retain_low_rank=668/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=0.0001 | retain_low_rank=668/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=0.0001 | retain_low_rank=668/768

Retain-low threshold: 1e-05
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=1e-05 | retain_low_rank=644/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=1e-05 | retain_low_rank=644/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=1e-05 | retain_low_rank=644/768

Retain-low threshold: 1e-06
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=1e-06 | retain_low_rank=340/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=1e-06 | retain_low_rank=340/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=1e-06 | retain_low_rank=340/768

Retain-low threshold: 1e-07
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=1e-07 | retain_low_rank=41/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=1e-07 | retain_low_rank=41/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=1e-07 | retain_low_rank=41/768

Retain-low threshold: 1e-08
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=1e-08 | retain_low_rank=6/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=1e-08 | retain_low_rank=6/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=1e-08 | retain_low_rank=6/768

Retain-low threshold: 1e-09
10_celebrity: retain_texts=100 | erase_targets=10 | threshold=1e-09 | retain_low_rank=0/768
50_celebrity: retain_texts=100 | erase_targets=50 | threshold=1e-09 | retain_low_rank=0/768
100_celebrity: retain_texts=100 | erase_targets=100 | threshold=1e-09 | retain_low_rank=0/768


