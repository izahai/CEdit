# MOV1: Common-anchor residual rank

This experiment tests whether a common anchor actually produces a compact
multi-concept edit. With target matrix `T`, residual matrix `R`, and the SPEED
edit statistic `D`, the update satisfies

\[
D=R^\top T/N,\qquad
\operatorname{rank}(\Delta W)
\leq \operatorname{rank}(D)
\leq \operatorname{rank}(R).
\]

All 100 celebrity targets use the same `person` anchor. The controlled
comparison contains:

- `legacy_full`: the original residuals `person - target`;
- `legacy_svd_rank30`: a rank-30 SVD of the legacy residual matrix with every
  target residual rescaled back to its original norm;
- `tgprs_rank30`: the existing target-global pairwise residual subspace at
  rank 30.

Each method is evaluated at residual scales `0.2`, `0.4`, `0.6`, `0.8`, and
`1.0`. All other SPEED and sampling settings are shared. A complete run trains
15 checkpoints and generates 15,000 images: 500 erase and 500 retain images
for each checkpoint at 50 diffusion steps.

## Start from an empty Vast AI server

Use a PyTorch image with CUDA 12.8 or newer for an RTX 5090 and provision at
least 100 GB of disk. On the local machine, define the new instance address:

```bash
export VAST_HOST='your.vast.host'
export VAST_PORT='12345'
```

Upload the current checkout. This is preferred to cloning because it includes
the local MOV1 implementation even before it is pushed:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" 'mkdir -p /workspace/CEdit'
rsync -az --progress \
  --exclude='.git/' \
  --exclude='logs/' \
  --exclude='motivation/mov1/outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  -e "ssh -p ${VAST_PORT}" \
  /Users/hainguyen/Repo/2026/ConceptErasure/Working/CEdit/ \
  "root@${VAST_HOST}:/workspace/CEdit/"
```

Connect and start a persistent run:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}"
cd /workspace/CEdit
nvidia-smi
tmux new -s mov1
mkdir -p motivation/mov1/outputs/logs
bash motivation/mov1/run_all.sh 2>&1 | tee motivation/mov1/outputs/logs/run_all.log
```

Detach from tmux with `Ctrl-b d`. Reconnect later with:

```bash
tmux attach -t mov1
```

The setup stage validates CUDA, installs the repository and plotting
dependencies, configures TensorFlow CUDA for GCD, and clones CE-Eval into
`/workspace/CE-Eval`. Model downloads happen automatically on first use.

## Resume or rerun individual stages

Every stage checks its expected artifacts and skips completed work:

```bash
bash motivation/mov1/02_analyze_residuals.sh
bash motivation/mov1/03_train.sh
bash motivation/mov1/04_infer.sh
bash motivation/mov1/05_evaluate.sh
bash motivation/mov1/06_summarize.sh
bash motivation/mov1/07_plot.sh
```

Checkpoint hashes connect training, sampling, and GCD evaluation. If a
checkpoint changes, its images and metrics are automatically regenerated
instead of silently reusing stale downstream artifacts.

After the environment stage, the focused validation suite can be run on the
Vast AI server with:

```bash
python -m unittest \
  tests.test_residual_subspace \
  tests.test_target_anchor_statistics \
  tests.test_train_config \
  tests.test_mov1_workflow
```

Force only the required stage when replacing an artifact:

```bash
FORCE_ANALYSIS=1 bash motivation/mov1/02_analyze_residuals.sh
FORCE_RETRAIN=1 bash motivation/mov1/03_train.sh
FORCE_RESAMPLE=1 bash motivation/mov1/04_infer.sh
FORCE_EVAL=1 bash motivation/mov1/05_evaluate.sh
FORCE_ANALYSIS=1 bash motivation/mov1/06_summarize.sh
FORCE_PLOT=1 bash motivation/mov1/07_plot.sh
```

Runtime settings can be overridden without editing YAML, for example:

```bash
GPU_ID=0 GCD_NUM_WORKERS=16 GCD_BATCH_SIZE=64 \
  bash motivation/mov1/05_evaluate.sh
```

## Monitor

From another local terminal:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 2'
```

Count generated images:

```bash
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  'find /workspace/CEdit/motivation/mov1/outputs/images -type f -name "*.png" 2>/dev/null | wc -l'
```

A complete run reports `15000`.

## Outputs

All runtime artifacts stay under `motivation/mov1/outputs/`, which is ignored
by the local `.gitignore`:

```text
outputs/
├── analysis/
│   ├── analysis.json
│   ├── rank_summary.csv
│   └── residual_spectrum.csv
├── checkpoints/<method>/scale_<value>/weight.pt
├── images/<method>/scale_<value>/100_celebrity/{erase,retain}/edit/*.png
├── gcd/<method>/scale_<value>/{erase,retain}.{csv,xlsx,log}
├── summary/metrics.csv
├── figures/
│   ├── common_anchor_rank_tradeoff.pdf
│   └── common_anchor_rank_tradeoff.png
└── logs/
```

The PDF is vector output for the paper; the PNG is rendered at 600 DPI. Panel
(a) reports residual spectra and residual/edit-statistic ranks. Panel (b)
reports the erase-retain Pareto curves with Wilson 95% confidence intervals.

Download the final evidence bundle:

```bash
mkdir -p motivation/mov1/outputs/downloaded
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/CEdit/motivation/mov1/outputs/figures/common_anchor_rank_tradeoff.pdf" \
  motivation/mov1/outputs/downloaded/
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/CEdit/motivation/mov1/outputs/figures/common_anchor_rank_tradeoff.png" \
  motivation/mov1/outputs/downloaded/
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/CEdit/motivation/mov1/outputs/summary/metrics.csv" \
  motivation/mov1/outputs/downloaded/
scp -P "${VAST_PORT}" \
  "root@${VAST_HOST}:/workspace/CEdit/motivation/mov1/outputs/analysis/rank_summary.csv" \
  motivation/mov1/outputs/downloaded/
```

Interpret the result conservatively:

- rank-30 Legacy-SVD above/right of full-rank legacy supports a causal benefit
  from controlling residual rank;
- TGPRS above/right of Legacy-SVD at the same rank supports better pairwise
  subspace selection;
- if Legacy-SVD does not improve on legacy, residual rank should not be claimed
  as the primary mechanism.
