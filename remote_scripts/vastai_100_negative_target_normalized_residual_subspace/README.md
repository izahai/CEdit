# Vast AI: 100 Celebrity Negative-Target Normalized Residual Subspace

This workflow trains, samples, evaluates, and summarizes
`negative_target_normalized_residual_subspace` for subspace ranks
`k = 10, 20, ..., 100` on the 100-celebrity benchmark.

Each rank is isolated under its own checkpoint, image, and CE-Eval output
directory, so the workflow can resume independently per `k`.

Run from the CEdit checkout root on Vast AI:

```bash
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/00_clone_repositories.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/01_setup_environment.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/02_train.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/03_infer.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/04_setup_ce_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/05_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/06_summarize_results.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/07_bundle_results.sh
```

The final comparison table is saved as `gcd/summary.csv` under the configured
output root. Use `FORCE_RETRAIN=1`, `FORCE_RESAMPLE=1`, or `FORCE_EVAL=1` to
rerun completed stages. Override `GPU_ID`, `PYTHON_BIN`, `OUTPUT_ROOT`, or
`CE_EVAL_ROOT` through the environment when needed.
