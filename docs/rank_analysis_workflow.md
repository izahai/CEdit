# Residual-rank analysis workflow

This workflow implements the geometry and causal rank interventions described in
`idea-analysis.md`. It loads Stable Diffusion once, evaluates closed-form SPEED
updates in memory, and never generates images or saves edited checkpoints.

## Run

Install the project requirements in a CUDA-enabled PyTorch environment. Start
with the pilot grid:

```bash
python scripts/run_rank_analysis.py \
  --config configs/rank_analysis_pilot.yaml
```

Then generate figures without loading the model:

```bash
python scripts/plot_rank_analysis.py \
  logs/rank_analysis/celebrity_rank_analysis_pilot
```

Run the main grid after validating the pilot output:

```bash
python scripts/run_rank_analysis.py \
  --config configs/rank_analysis.yaml
```

The runner appends records and skips completed configuration IDs, so rerunning
the same command resumes an interrupted analysis. Use `--fresh` to fail early
when result records already exist. `--part a` and `--part b` run one section.

## Output contract

Each run directory contains:

- `manifest.json`: resolved configuration, dataset hash, git revision, status,
  model information, and artifact names;
- `geometry_metrics.jsonl`: one Part A record per target subset and residual
  construction;
- `edit_metrics.jsonl`: per-layer and aggregate Part B records;
- `spectra.npz`: complete Part A singular-value arrays keyed by record ID;
- `retain_singular_values.npy`: retain-covariance spectrum;
- `figure_data/*.csv`: flattened and aggregated data used by plotting;
- `figures/*.pdf` and `figures/*.png`: rank, spectral-energy, Pareto, and
  directional-realization figures.

JSONL is the canonical metric format. Every row has a stable `record_id`, which
makes interrupted runs recoverable and allows outputs from separate jobs to be
audited. Long singular-value arrays live in compressed NumPy files instead of
being duplicated in JSON.

## Metrics

Part A reports numerical rank, stable rank, and spectral effective rank for the
actual residual matrix, its row-normalized version, and SPEED's edit statistic
`D = R.T @ T / N`.

Part B compares full legacy residuals, norm-matched legacy SVD truncation,
seeded random low-rank controls, and norm-matched TGPRS. For each `to_v` layer
it reports target-relative change, target output rotation, retain-relative
change, cosine realization of `delta_W @ t` versus `W @ r`, and relative
directional error.

The default configuration requires `retain.aug_num: 0`. This makes the retain
projection layer-independent and permits exact caching across residual methods.
Set `part_b.compute_delta_spectrum: true` only for a small diagnostic run;
performing an SVD of every layer update is substantially more expensive and is
not required for the Pareto figures.

## Hardware

The default float32 pipeline is comfortable on a 12 GB CUDA GPU. An 8 GB GPU
may work because no denoising pass is run, but has less SVD workspace headroom.
The output itself is normally below 100 MB; allow roughly 5--8 GB separately
for the Stable Diffusion v1.4 model cache. Plotting is CPU-only.
