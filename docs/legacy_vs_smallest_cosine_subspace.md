# Legacy vs. Smallest-Cosine Subspace Residuals

This note summarizes the two target-to-anchor residual construction methods
implemented by `train_erase_null.py`. Both methods use the same SPEED editing
procedure after the residuals have been constructed; they differ only in how
the residual for each target concept is defined.

## Shared notation

For target concept `i`, let:

- `t_i` be its text embedding;
- `a_i` be the embedding of its anchor concept; and
- `r_i` be the residual used to construct the edit statistic.

The training code computes:

```text
S_tt = mean_i(t_i^T t_i)
D    = mean_i(r_i^T t_i)
```

`S_tt` and `D`, together with retain-set statistics, are used by SPEED to
calculate the update for each selected cross-attention projection matrix. The
methods below therefore affect the edit through `D`.

## `legacy`

The legacy method independently uses the direct anchor displacement for every
target:

```text
r_i = a_i - t_i
```

This is the original, pairwise interpretation of the edit: each target is
moved toward its corresponding anchor. If the anchor is the null prompt, every
target is edited toward the null embedding.

### Characteristics

- Each target-anchor pair contributes its own direction and magnitude.
- The residual matrix can contain many unrelated directions when the target
  concepts are diverse.
- No residual selection, subspace construction, or projection is performed.
- It is the simplest baseline and has no `residual_top_k` dependency.

### YAML example

```yaml
anchor_mode: legacy
anchor_concepts:
  - person
```

For multiple targets, a single anchor is broadcast to all targets. Alternatively,
one anchor can be supplied for each target.

## `smallest_cosine_subspace`

This method starts from the same legacy residuals:

```text
d_i = a_i - t_i
```

It then identifies the residuals that are least aligned with the rest of the
residual set. First, it computes the mean pairwise cosine score for each
normalized legacy residual:

```text
s_i = mean_{j != i}(cos(d_i, d_j))
```

The `k = residual_top_k` residuals with the smallest scores are selected. Their
rows are factorized with SVD to construct an orthonormal basis `B` for a
residual subspace.

For each target, the method projects the target embedding into this subspace
and chooses the opposite direction:

```text
p_i = projection_B(t_i)
r_i = -||d_i|| * p_i / ||p_i||
```

Thus, the new residual remains inside the selected subspace and preserves the
norm of the corresponding legacy residual. Its direction is chosen to oppose
the target's projection onto that subspace.

If a target has a near-zero subspace projection, the implementation falls back
to its projected legacy residual. If that is also near zero, it uses the first
basis vector. These fallbacks are reported in the training diagnostics.

### Characteristics

- Uses a shared low-dimensional subspace instead of independent pairwise
  directions.
- Prioritizes residuals that have low average cosine similarity with the other
  residuals; these are the most directionally distinct candidates under this
  criterion.
- Keeps the magnitude of each legacy residual, so the method changes direction
  rather than simply shrinking or expanding the edit signal.
- Adds `residual_top_k` as a method-specific hyperparameter. `k` must be at
  least one and no larger than the number of target concepts.

### YAML example

```yaml
anchor_mode: smallest_cosine_subspace
residual_top_k: 10
residual_scale: 1.0
anchor_concepts:
  - person
```

For the 100-celebrity Vast workflow, `residual_top_k: 10` selects the ten
lowest-mean-cosine legacy residuals before building the subspace.

## Practical comparison

| Aspect | `legacy` | `smallest_cosine_subspace` |
| --- | --- | --- |
| Residual direction | Directly from each anchor-target pair | Opposite each target's projection into a selected residual subspace |
| Residual magnitude | `||a_i - t_i||` | Preserves `||a_i - t_i||` |
| Cross-target coupling | None during residual construction | Shared basis learned from selected residuals |
| Extra hyperparameter | None | `residual_top_k` |
| Diagnostics | General edit/residual statistics | Plus selected indices, cosine scores, basis rank, singular values, and fallback counts |

Neither method changes the sampling scheduler, diffusion steps, retain set, or
the remaining SPEED matrix-update logic. To make a controlled comparison, keep
all of those settings fixed and change only `anchor_mode` (and
`residual_top_k` for the subspace method).
