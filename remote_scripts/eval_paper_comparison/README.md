# Vast AI: Paper Comparison

This workflow compares original Stable Diffusion v1.4, the paper's legacy
SPEED configuration, and `target_global_pairwise_residual_subspace` on the
10-, 50-, and 100-celebrity erase/retain benchmarks.

## Upload and run

Create an RTX 5090 Vast instance with a CUDA 12.8+ PyTorch image and at least
100 GB of disk. Set its connection details on your Mac:

```bash
export VAST_HOST='180.189.55.43'
export VAST_PORT='19845'
```

Upload the repository using the macOS-compatible rsync progress option:

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

Connect to the instance and run the workflow inside Vast's tmux session:

```bash
ssh -p "${VAST_PORT}" -L 8080:localhost:8080 "root@${VAST_HOST}"
cd /workspace/CEdit
nvidia-smi
bash remote_scripts/eval_paper_comparison/run_all.sh
```

The workflow is resumable. Force individual work when required:

```bash
FORCE_RESAMPLE=1 bash remote_scripts/eval_paper_comparison/02_generate_original.sh
FORCE_RETRAIN=1 bash remote_scripts/eval_paper_comparison/03_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_paper_comparison/04_infer_edits.sh
FORCE_EVAL=1 bash remote_scripts/eval_paper_comparison/06_eval.sh
```

GCD uses the GPU plus 8 CPU image workers, batch size 32, and prefetch factor
2. Override these without editing the workflow:

```bash
GCD_NUM_WORKERS=16 GCD_BATCH_SIZE=64 GCD_PREFETCH_FACTOR=2 \
  FORCE_EVAL=1 bash remote_scripts/eval_paper_comparison/06_eval.sh
```

## Monitor progress

Start one-second GPU utilization logging from a local terminal:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'nohup nvidia-smi dmon -i 0 -s u -d 1 -o T \
   >> /workspace/gpu_sm.log 2>&1 & echo $! > /workspace/gpu_sm.pid'
```

View or stop the monitor:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'tail -f /workspace/gpu_sm.log'

ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'kill "$(cat /workspace/gpu_sm.pid)"'
```

Count generated images from the local machine:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'find /workspace/cedit_ce_eval_outputs_paper_comparison/images -type f -name "*.png" 2>/dev/null | wc -l'
```

A complete run has 9,000 images: 3 methods, 3 benchmarks, and 500 erase plus
500 retain images per benchmark.

## Results

Outputs are written under:

```text
/workspace/cedit_ce_eval_outputs_paper_comparison/
├── checkpoints/{legacy,target_global_pairwise_residual_subspace}/<benchmark>/weight.pt
├── images/<method>/<benchmark>/<benchmark>/{erase,retain}/{original,edit}/*.png
└── gcd/
    ├── <method>/<benchmark>/<method>_{erase,retain}.{csv,xlsx,log}
    └── summary.csv
```

Download the tmux log and unified 18-row summary into the current local
directory:

```bash
scp -P "${VAST_PORT}" "root@${VAST_HOST}:/workspace/tmux-log.log" .
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/cedit_ce_eval_outputs_paper_comparison/gcd/summary.csv" \
  .
```

Lower erase accuracy and higher retain accuracy are better. The legacy rows
use the paper configuration with `aug_num=10`; target-global rows use the
existing new-method configuration with `aug_num=0` and residual rank 30.
