# Vast AI: Multi-Concept Evaluation 1

This workflow trains and fully evaluates six SPEED checkpoints using
`negative_target_normalized_residual_subspace` with residual rank 30:

| Config | Benchmark tasks | `aug_num` | `threshold` | `retain_scale` |
|---|---|---:|---:|---:|
| `config_1` | `10_celebrity`, `50_celebrity`, `100_celebrity` | 0 | 1e-4 | 0.05 |
| `config_2` | `10_celebrity`, `50_celebrity`, `100_celebrity` | 10 | 1e-4 | 0.05 |

Each checkpoint is sampled on all 500 erase and 500 retain prompts from its
benchmark CSV, evaluated with CE-Eval GCD, and included in `gcd/summary.csv`.
The summary contains 12 rows: two splits for each of the six models.

Rank 30 is the requested rank for every model. A residual matrix built from 10
targets has maximum rank 10, and the repository rejects a larger rank. The
workflow therefore uses effective rank 10 for `10_celebrity` and rank 30 for
`50_celebrity` and `100_celebrity`. Both requested and effective ranks are
recorded in the final summary CSV.

## Expected server layout

```text
/workspace/
├── CEdit/
└── CE-Eval/
```

On a fresh server, clone CEdit first:

```bash
cd /workspace
git clone --branch main https://github.com/izahai/CEdit.git CEdit
```

## Run the complete workflow

From `/workspace/CEdit`:

```bash
bash remote_scripts/eval_multi_1/00_clone_repositories.sh
bash remote_scripts/eval_multi_1/01_setup_environment.sh
bash remote_scripts/eval_multi_1/02_train.sh
bash remote_scripts/eval_multi_1/03_infer.sh
bash remote_scripts/eval_multi_1/04_setup_ce_eval.sh
bash remote_scripts/eval_multi_1/05_eval.sh
bash remote_scripts/eval_multi_1/06_summarize_results.sh
bash remote_scripts/eval_multi_1/07_bundle_results.sh
```

The default output root is:

```text
/workspace/cedit_ce_eval_outputs_eval_multi_1/
```

Outputs are isolated by config and task:

```text
checkpoints/<config>/<benchmark>/weight.pt
images/<config>/<benchmark>/<benchmark>/{erase,retain}/edit/*.png
gcd/<config>/<benchmark>/*.csv
gcd/summary.csv
```

Training target concepts are derived from the unique `concept` values in only
the `type=erase` rows of each benchmark CSV. The script validates that this
produces exactly 10, 50, and 100 targets respectively.

## Resume and overrides

Completed stages are skipped independently for each config/task pair. Force a
stage to rerun with:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_multi_1/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_multi_1/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/eval_multi_1/05_eval.sh
```

`GPU_ID`, `PYTHON_BIN`, `OUTPUT_ROOT`, and `CE_EVAL_ROOT` can be overridden
through environment variables.
