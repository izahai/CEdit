# TGPRS negative vs. positive projection

This ablation keeps the target-global pairwise residual subspace setup fixed
and changes only the sign of each target projection. It trains and evaluates
`negative_projection` and `positive_projection` on `10_celebrity`,
`50_celebrity`, and `100_celebrity`.

The two residual directions are:

```text
negative: r_i = -||a_i - t_i|| normalize(Proj_B(t_i))
positive: r_i = +||a_i - t_i|| normalize(Proj_B(t_i))
```

## Run on a GPU server

The default workflow expects the Vast PyTorch interpreter at
`/venv/main/bin/python`, GPU 0, and a CUDA 12.8+ PyTorch wheel for RTX 50-series
GPUs. Upload the repository from the local machine if needed:

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

Then, from the repository root on the GPU server, run:

```bash
bash ablation/negative_or_plus/run_all.sh
```

If the 10- and 100-celebrity runs already exist, run only the newly added
50-celebrity ablation and then rebuild the combined summary with:

```bash
bash ablation/negative_or_plus/run_50.sh
```

The workflow is resumable. Individual expensive stages skip complete outputs;
force them when needed:

```bash
FORCE_RETRAIN=1 bash ablation/negative_or_plus/03_train.sh
FORCE_RESAMPLE=1 bash ablation/negative_or_plus/04_infer.sh
FORCE_EVAL=1 bash ablation/negative_or_plus/05_eval.sh
```

Common overrides can be passed without editing the YAML:

```bash
GPU_ID=1 \
PYTHON_BIN=/venv/main/bin/python \
OUTPUT_ROOT=/workspace/my_negative_or_plus_run \
bash ablation/negative_or_plus/run_all.sh
```

The complete run creates six checkpoints and 6,000 images: two projection
directions, three benchmarks, and 500 erase plus 500 retain images per run.

## Outputs

By default, results are stored under:

```text
/workspace/cedit_ce_eval_outputs_negative_or_plus/
├── checkpoints/{negative_projection,positive_projection}/<benchmark>/weight.pt
├── images/<method>/<benchmark>/<benchmark>/{erase,retain}/edit/*.png
└── gcd/
    ├── <method>/<benchmark>/<method>_{erase,retain}.{csv,xlsx,log}
    └── summary.csv
```

`summary.csv` contains twelve rows: two directions by three benchmarks by two
splits. For erase rows, lower identity accuracy/hit rate is better. For retain
rows, higher accuracy/hit rate is better.
