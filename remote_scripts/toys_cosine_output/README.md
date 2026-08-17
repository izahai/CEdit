# TGPRS output-cosine toy analysis

This workflow measures the existing
`target_global_pairwise_residual_subspace` method without training or editing
any model weight. It compares each target embedding with its TGPRS-shifted
embedding before and after every original Stable Diffusion v1.4
`attn2.to_v.weight` matrix.

The default experiment uses the first five erase concepts from
`data/10_celebrity.csv`. Five targets produce 30 ordered target/anchor residuals,
so the configured TGPRS rank of 30 is valid.

## Run on the GPU server

From an existing CEdit checkout on the server:

```bash
cd /workspace/CEdit
bash remote_scripts/toys_cosine_output/run_all.sh
```

To rerun only the analysis:

```bash
cd /workspace/CEdit
bash remote_scripts/toys_cosine_output/02_analyze.sh
```

To analyze all targets in `data/50_celebrity.csv` and write to the isolated
`/workspace/cedit_toys_cosine_output_50_celebrity` folder:

```bash
cd /workspace/CEdit
bash remote_scripts/toys_cosine_output/run_50_celebrity.sh
```

Environment overrides are supported:

```bash
GPU_ID=0 \
OUTPUT_ROOT=/workspace/cedit_toys_cosine_output_test \
bash remote_scripts/toys_cosine_output/02_analyze.sh
```

Edit `workflow.yaml` to change the checkpoint, concepts, target count, residual
rank, dtype, or device. The requested residual rank must not exceed
`N * (N + 1)` for `N` targets.

## Outputs

By default, artifacts are written to
`/workspace/cedit_toys_cosine_output`:

- `target_layer_metrics.csv`: one row per target and `to_v` layer;
- `layer_summary.csv`: indexed mean, minimum, maximum, and standard deviation by
  layer;
- `target_summary.csv`: indexed statistics by target;
- `analysis.json`: configuration, TGPRS diagnostics, and overall statistics;
- `output_cosine_heatmap.png`: target-by-layer output cosine similarities;
- `output_angle_heatmap.png`: target-by-layer output angles in degrees;
- `residual_norm_vs_output_angle.png`: residual scale versus output rotation;
- `embedding_angle_vs_output_angle.png`: embedding rotation versus output
  rotation.

The output-space quantities are calculated with the original frozen weight:

\[
\cos\left(W_\ell t_i, W_\ell(t_i+r_i)\right),
\qquad
\frac{\lVert W_\ell r_i\rVert_2}{\lVert W_\ell t_i\rVert_2}.
\]

No checkpoint, generated image, or model edit is produced.
