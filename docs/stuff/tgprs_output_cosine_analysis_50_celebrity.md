# TGPRS Output-Cosine Analysis on 50 Celebrities

## Purpose

This toy experiment examines how the existing
`target_global_pairwise_residual_subspace` (TGPRS) residual behaves after each
original Stable Diffusion v1.4 cross-attention value-projection weight. It does
not edit model weights, train a model, or generate images.

The analysis uses all 50 erase concepts from `data/50_celebrity.csv`, TGPRS
rank 30, the conventional `person` anchor, and the fixed empty-prompt and
`person` global anchors. It evaluates 50 targets across 16 original
`attn2.to_v` layers, producing 800 target-layer measurements.

## Measured output angle

For target embedding \(t_i\), existing TGPRS residual \(r_i\), and original
frozen value-projection weight \(W_\ell\), the experiment calculates:

\[
\theta_{i,\ell}
=
\arccos\left(
\frac{
(W_\ell t_i)^\top W_\ell(t_i+r_i)
}{
\lVert W_\ell t_i\rVert_2
\lVert W_\ell(t_i+r_i)\rVert_2
}
\right).
\]

Therefore, each heatmap cell is the angle between:

1. the original target output, \(W_\ell t_i\); and
2. the shifted-target output, \(W_\ell(t_i+r_i)\).

It is **not** the angle between \(W_\ell t_i\) and the isolated output
residual \(W_\ell r_i\).

In the heatmap, rows represent target concepts, columns represent original
`to_v` weights, and color represents \(\theta_{i,\ell}\) in degrees.

## Main result

The most important insight is that TGPRS output rotation is overwhelmingly
determined by the particular `to_v` layer rather than by target identity.

For the balanced 50-target by 16-layer result:

- mean output cosine: 0.256628;
- mean output angle: 73.950 degrees;
- complete output-angle range: 34.463 to 112.427 degrees;
- layer mean-angle range: 42.545 to 101.008 degrees;
- target mean-angle range: 66.026 to 80.472 degrees;
- target-layer pairs rotated beyond 90 degrees: 258 of 800, or 32.25%;
- effective TGPRS basis rank: 30.

A balanced variance decomposition attributes approximately:

- 95.7% of output-angle variation to the layer;
- 2.2% to the target concept;
- 2.1% to the remaining target-layer interaction.

The output-angle heatmap displays this result as strong vertical bands. For a
given layer, different targets usually receive similar rotations, while the
same target receives substantially different rotations across layers.

![TGPRS output angle heatmap](../logs/toys_cosine_output_50_celebrity/output_angle_heatmap.png)

## Embedding-space measurements do not predict output rotation

The existing TGPRS residual is constructed entirely from embedding-space
geometry, but those measurements are weak predictors of its effect after a
specific value-projection weight:

- correlation between embedding angle and output angle: approximately 0.11;
- correlation between raw embedding residual norm and output angle:
  approximately 0.04.

Consequently, choosing a residual direction or magnitude only from embedding
geometry cannot reliably control the actual output rotation of every layer.

![Embedding angle versus output angle](../logs/toys_cosine_output_50_celebrity/embedding_angle_vs_output_angle.png)

By contrast, the relative output residual

\[
\frac{\lVert W_\ell r_i\rVert_2}{\lVert W_\ell t_i\rVert_2}
\]

has a correlation of approximately 0.986 with output angle. This is the most
useful scale for controlling how strongly a residual affects a particular
weight.

## Implication for the learnable residual method

The current method constructs one residual \(r_i\) for each target and shares
it across every edited layer. The experiment shows that the same residual can
produce a moderate rotation in one layer and an angle greater than 90 degrees
in another. A shared embedding residual therefore cannot provide consistent
output-space behavior.

The proposed method should instead learn a target- and layer-specific residual:

\[
r_i \longrightarrow r_{i,\ell}.
\]

For each frozen original weight \(W_\ell\), it should find the smallest
residual in the TGPRS subspace that reaches a controlled output-angle margin:

\[
\min_{r_{i,\ell}}
\frac{\lVert W_\ell r_{i,\ell}\rVert_2}
     {\lVert W_\ell t_i\rVert_2}
\quad
\text{subject to}
\quad
\cos\left(
W_\ell t_i,
W_\ell(t_i+r_{i,\ell})
\right)
\leq \tau.
\]

This changes the objective from producing a large directional difference to
producing the minimum layer-specific change needed to satisfy a chosen angular
constraint. The fact that 32.25% of existing target-layer pairs already rotate
beyond 90 degrees indicates that minimizing the residual is important: the
existing fixed-norm construction frequently produces more rotation than is
likely necessary.

## Reproduction and artifacts

Run the 50-celebrity workflow on the GPU server with:

```bash
cd /workspace/CEdit
bash remote_scripts/toys_cosine_output/run_50_celebrity.sh
```

The server writes results to:

```text
/workspace/cedit_toys_cosine_output_50_celebrity
```

The downloaded local results are under
`logs/toys_cosine_output_50_celebrity/`. Important artifacts include:

- `target_layer_metrics.csv`: all 800 measurements;
- `layer_summary.csv`: statistics grouped by `to_v` layer;
- `target_summary.csv`: statistics grouped by target;
- `analysis.json`: experiment metadata and TGPRS diagnostics;
- `output_angle_heatmap.png`: target-by-layer output angles;
- `embedding_angle_vs_output_angle.png`: comparison of embedding and output
  rotation.
