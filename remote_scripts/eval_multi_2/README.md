# Vast AI: Global Pairwise Residual Subspace Evaluation

This workflow trains and evaluates three rank-30
`global_pairwise_residual_subspace` models for `10_celebrity`, `50_celebrity`,
and `100_celebrity`. It uses the single `train_config.yaml` configuration.

## New Vast instance

Create a CUDA-capable Vast PyTorch instance with at least 50 GB of disk. Vast
automatically opens the SSH shell inside tmux.

From your **local machine**, upload this repository to the empty instance:

```bash
rsync -az --progress \
  --exclude='.git/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  -e 'ssh -p 41959' \
  /Users/hainguyen/Repo/2026/ConceptErasure/Working/CEdit/ \
  root@39.119.10.242:/workspace/CEdit/
```

Keep the trailing `/` after the local `CEdit/`. Re-run the same command whenever
you want to upload newer local changes.

Connect to Vast and start the complete resumable workflow:

```bash
ssh -p 41959 root@39.119.10.242 -L 8080:localhost:8080
cd /workspace/CEdit
bash remote_scripts/eval_multi_2/run_all.sh
```

The workflow clones CE-Eval, installs dependencies, trains three checkpoints,
generates images, evaluates them, and writes the summary. Re-running
`run_all.sh` skips completed work.

To force individual stages:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_multi_2/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_multi_2/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/eval_multi_2/05_eval.sh
```

## Log and outputs

The Vast tmux pane is logged automatically to readable plain text at:

```text
/workspace/tmux-log.log
```

Do not use `tail -f` on this file inside the logged pane because it creates a
feedback loop.

To clean an older raw log already downloaded into the repository directory:

```bash
python3 remote_scripts/eval_multi_2/clean_tmux_log.py \
  tmux-log-clean.log --input tmux-log.log
```

From your local machine, download the log into the current directory:

```bash
scp -P 41959 root@39.119.10.242:/workspace/tmux-log.log .
```

Results are written under:

```text
/workspace/cedit_ce_eval_outputs_eval_multi_2/
├── checkpoints/<benchmark>/weight.pt
├── images/<benchmark>/<benchmark>/{erase,retain}/edit/*.png
└── gcd/
    ├── <benchmark>/*.csv
    └── summary.csv
```

Count the generated images directly from your local machine:

```bash
ssh -p 41959 root@39.119.10.242 \
  'find /workspace/cedit_ce_eval_outputs_eval_multi_2/images \
    -type f -name "*.png" 2>/dev/null | wc -l'
```

A complete run contains 3,000 images: 500 erase and 500 retain images for each
of the three benchmarks.

Download the final summary into the current local directory:

```bash
scp -P 41959 \
  root@39.119.10.242:/workspace/cedit_ce_eval_outputs_eval_multi_2/gcd/summary.csv \
  .
```
