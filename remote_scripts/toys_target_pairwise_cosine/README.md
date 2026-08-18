# Natural target and anchor pairwise-cosine analysis

This control experiment analyzes the geometry of the 50 unique erase concepts
in `data/50_celebrity.csv` together with the `person` and empty-prompt
embeddings. It does not construct a residual, run TGPRS, edit weights, train a
model, or generate images.

The empty prompt is encoded with the same last-subject-token extraction used by
the existing TGPRS workflow. The vector order is 50 targets, `person`, then
`<empty>`.

For the embedding space and every original Stable Diffusion v1.4
`attn2.to_v.weight`, the workflow calculates a complete 52-by-52 cosine matrix:

\[
C^{(\ell)}_{ij}=\cos(W_\ell x_i,W_\ell x_j).
\]

Summary statistics remain separated into target-target, target-person,
target-empty, and person-empty pairs.

## Run on the GPU server

```bash
cd /workspace/CEdit
bash remote_scripts/toys_target_pairwise_cosine/run_all.sh
```

The default output directory is:

```text
/workspace/cedit_toys_target_pairwise_cosine_50_celebrity
```

## Outputs

- `pairwise_metrics.csv`: every unordered pair in embedding and output spaces;
- `space_summary.csv`: pair-class statistics, norms, correlation to embedding
  geometry, and RMSE from embedding geometry;
- `labels.csv`: matrix index, label, and target/anchor kind;
- `matrices/embedding.csv`: the embedding-space 52-by-52 matrix;
- `matrices/layer_00.csv` through `layer_15.csv`: output-space matrices;
- `embedding_pairwise_cosine_heatmap.png`: embedding-space matrix;
- `layer_pairwise_cosine_heatmaps.png`: all 16 output matrices using one color
  scale;
- `layer_pair_type_summary.png`: target-target, target-person, target-empty,
  and person-empty cosine trends across layers;
- `analysis.json`: configuration and artifact counts.
