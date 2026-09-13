# How TGPRS Runs in `eval_few_style_1`

This document explains the **Target-Global Pairwise Residual Subspace
(TGPRS)** branch executed by
`remote_scripts/eval_few/eval_few_style_1/run_all.sh`. It describes the
current code and configuration, not just the general TGPRS method.

> **In one sentence:** for each artist, TGPRS learns a rank-30 semantic
> subspace from 100 artist-neutral directions, constructs one erasure residual
> inside that subspace, and supplies that residual to SPEED's closed-form
> cross-attention value-matrix update.

## 1. What the workflow compares

The full workflow compares three model states:

| Model | Anchor construction | Retain augmentation | Retain threshold |
|---|---|---:|---:|
| Original SD v1.4 | No edit | — | — |
| Legacy SPEED | Direct residual `art - artist` | 10 | 0.1 |
| TGPRS + SPEED | Rank-30 residual subspace | 0 | 0.4 |

It creates three independent single-concept editing tasks:

| Task ID | Erased target | Direct anchor |
|---|---|---|
| `van_gogh` | `Van Gogh` | `art` |
| `picasso` | `Picasso` | `art` |
| `monet` | `Monet` | `art` |

Each task starts again from the unedited
`CompVis/stable-diffusion-v1-4`; the edits are **not accumulated** into one
model.

The source-of-truth settings are
[`workflow.yaml`](../remote_scripts/eval_few/eval_few_style_1/workflow.yaml)
and
[`train_config_target_global_pairwise_residual_subspace.yaml`](../remote_scripts/eval_few/eval_few_style_1/train_config_target_global_pairwise_residual_subspace.yaml).
At the time of writing, the TGPRS config requests `residual_rank: 30` and
contains 100 neutral subspace anchors. The directory README's statement that
the applied rank is 100 is stale; the loader currently applies rank 30.

## 2. End-to-end flow

```text
Validate configuration
        │
        ├── Generate shared original style images
        │
        ├── For each artist, train a fresh TGPRS/SPEED checkpoint
        │       ├── Encode artist, `art`, and 100 neutral phrases
        │       ├── Build and compress the residual matrix
        │       ├── Synthesize the artist's erasure residual
        │       └── Edit every cross-attention `to_v` weight in closed form
        │
        ├── Generate edited style images
        ├── Generate original and edited MS-COCO images
        └── Compute CLIP score and FID; write comparison CSV files
```

`run_all.sh` executes stages `00` through `06` in this order. The stages are
resumable: existing checkpoints, complete image directories, and metric rows
are reused unless the corresponding `FORCE_*` environment variable is set.

## 3. Text embeddings used by TGPRS

Let the CLIP text-embedding width be `d = 768`. For one task, define:

- `t`, with shape `1 x d`: the last subject-token embedding of the
  target artist, such as `Van Gogh`;
- `a`, with shape `1 x d`: the last subject-token embedding of the
  direct anchor `art`;
- `q_j`, with shape `1 x d`, for `j = 1, ..., 100`: embeddings of the neutral
  phrases in the TGPRS YAML.

Examples of the neutral phrases are `painting`, `bold brushwork`, `watercolor
wash`, `geometric composition`, `dramatic lighting`, and `gallery art`. They
span medium, technique, color, composition, lighting, and presentation. None
of the five artists used in evaluation appears in this list.

The code uses `erase_style: false`, so it encodes the artist name itself for
training. This flag does not prevent style evaluation; it controls only how
the target embedding is formed.

## 4. Constructing the TGPRS residual subspace

### 4.1 Build directed differences

In the general multi-target algorithm, every target is a source and the
destinations are all *other* targets plus all extra anchors:

```text
g_(i -> j) = t_j - t_i
g_(i -> q) = q - t_i
```

This workflow erases only one artist per checkpoint. Therefore, there are no
other targets and no target-to-target rows. Its residual matrix is exactly

```text
    [ q_1   - t ]
    [ q_2   - t ]
G = [     ...   ]    shape: 100 x 768
    [ q_100 - t ]
```

This is an important interpretation of the name **pairwise** in this
experiment: the implemented general method supports ordered target pairs, but
the single-target style tasks use only target-to-neutral-anchor pairs.

### 4.2 Normalize before SVD

Every row is normalized independently:

```text
g_hat_j = g_j / max(norm(g_j), epsilon)
epsilon = 1e-8
```

The normalization prevents long text-embedding differences from dominating
solely because of magnitude. The SVD therefore summarizes frequently shared
**directions** among the 100 residuals:

```text
G_hat = U Sigma V^T
```

The code requests the first 30 right-singular vectors. If the numerical rank
is smaller, it uses the smaller effective rank:

```text
B = first k rows of V^T                 shape: k x 768
k = min(30, numerical_rank(G_hat))
```

The associated orthogonal subspace projector is

```text
P_B = B^T B                             shape: 768 x 768
```

The workflow loader bounds the requested rank as follows:

```text
N + Q - 1 = 1 + 100 - 1 = 100
```

The configured rank 30 is therefore passed unchanged. This loader bound is
only a structural maximum; the SVD code independently handles a lower
numerical rank.

### 4.3 Construct one residual for the artist

Legacy SPEED would use the direct residual

```text
r_legacy = a - t
```

TGPRS uses its **length**, but normally replaces its direction. It projects
the target into the learned subspace, reverses that projection, normalizes it,
and restores the legacy length:

```text
projected_target = t P_B

          norm(a - t)
r_TGPRS = ----------- (-projected_target)
          norm(projected_target)
```

Equivalently, `r_TGPRS` points opposite the projected target and has length
`norm(a - t)`.

Because `residual_scale: 1.0`, no further scaling occurs. Consequently:

- the 100 neutral anchors decide the allowed semantic subspace;
- the projected artist embedding decides the direction within that subspace;
- the direct `art` anchor decides the residual magnitude;
- `art` also happens to be one of the 100 neutral anchors, so it contributes
  to the learned subspace as well.

If `t P_B` is numerically zero, the implementation falls back first to the
normalized projection of `a - t`, and then, if that is also zero, to the first
basis vector. Training logs report both fallback counts.

The conceptual replacement is therefore:

```text
Legacy: target ───────────────► art
         residual = art - target

TGPRS:  target ───────────────► target + r_TGPRS
         direction = opposite the target's projection in the neutral subspace
         length    = distance from target to art
```

TGPRS does not train this basis by gradient descent. Building `G`, taking its
SVD, and constructing `r_TGPRS` are deterministic tensor
operations once the text embeddings are fixed.

## 5. Turning the residual into SPEED edit statistics

For `N` targets, the editor forms

```text
S_tt = (1 / N) sum_i(t_i^T t_i)
D_ta = (1 / N) sum_i(r_i^T t_i)
```

Here `N = 1`, so

```text
S_tt = t^T t
D_ta = r_TGPRS^T t
```

This reveals another useful distinction: the learned residual **basis** can
have rank 30, but a single-target run emits only one residual. Thus the
residual output and the outer-product statistic `D_ta` each have rank at
most one. Rank 30 enriches the space from which that one direction is chosen;
it does not make the final single-target edit statistic rank 30.

## 6. Retain preservation in SPEED

The retain set comes from the unique values in the `concept` column of
`data/style.csv`, after prompts containing the target artist are removed. If
the retained embeddings are `z_1, ..., z_L`, SPEED computes their
uncentered second moment

```text
C_R = (1 / L) sum_l(z_l^T z_l)
```

It decomposes `C_R = U_R Sigma_R V_R^T` and keeps directions whose singular
values are below the absolute threshold `0.4`:

```text
U_low = columns of U_R for which singular_value < 0.4
P_R   = U_low U_low^T
```

This is the **retain-low projector**. It favors update directions with low
energy in retained concepts. Do not confuse it with `P_B`:

| Projector | Learned from | Selection | Purpose |
|---|---|---|---|
| `P_B` | normalized target-to-neutral residuals | top 30 SVD directions | construct the TGPRS erasure residual |
| `P_R` | style retain embeddings | all singular values below 0.4 | constrain the SPEED weight update |

The TGPRS config sets `aug_num: 0`. No noisy retain embeddings are generated,
and `C_R` and `P_R` are therefore layer-independent. This also means the
experiment is a system-level comparison with legacy SPEED, not a pure
anchor-mode ablation: legacy uses `aug_num: 10` and threshold `0.1`.

## 7. Closed-form UNet weight update

Only cross-attention value projections are edited because `params: V`.
For each UNet parameter whose name contains `attn2.to_v`, let its original
weight be `W`. Define

```text
M = inverse(S_tt P_R + gamma I)
gamma = retain_scale = 1
```

SPEED also constructs `K_2` from the beginning-of-sequence empty-prompt
embedding and three k-means centers of the remaining empty-prompt token
embeddings. With `lamb: 0.0`, the implemented update is

```text
middle = I - M K_2 inverse(K_2^T P_R M K_2 + lambda I) K_2^T P_R

delta_W = W D_ta P_R middle M

lambda = 0
```

followed by

```text
W_edited = W + delta_W
```

This is a closed-form edit: there is no optimizer, loss loop, or
back-propagation. The resulting dictionary of edited `to_v` tensors is saved
as

```text
<output_root>/checkpoints/
  target_global_pairwise_residual_subspace/<task_id>/weight.pt
```

At sampling time, this partial state dictionary is loaded into a copy of the
original UNet with `strict=False`; all parameters absent from the checkpoint
remain unchanged.

## 8. Sampling protocol

Every checkpoint is tested on five style contents:

```text
Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio
```

The target artist is the erasure case; the other four artists are preservation
cases. Each content fills all 30 style templates in `src/template.py`, giving
30 images per content and 150 style images per checkpoint.

Sampling uses:

| Setting | Value |
|---|---:|
| Seed | 0 |
| Scheduler | DPM-Solver Multistep |
| Denoising steps | 20 |
| Classifier-free guidance | 7.5 |
| Samples per prompt | 1 |
| Style batch size | 1 |

The original and edited runs use the same seed and prompt ordering, producing
the same latent sequence for paired comparisons. Preservation is additionally
tested on the first 100 configured MS-COCO records, sampled through
`sample2.py`.

For the full three-method workflow, the expected image count is:

| Images | Count |
|---|---:|
| Shared original style images: `5 x 30` | 150 |
| Six edited checkpoints: `2 methods x 3 tasks x 5 x 30` | 900 |
| Original MS-COCO images | 100 |
| Six edited MS-COCO sets: `6 x 100` | 600 |
| **Total** | **1,750** |

## 9. Evaluation and interpretation

The evaluator computes two metrics:

- **CLIP score:** cosine similarity between each generated image and its
  actual generation prompt, multiplied by 100;
- **FID versus original:** distributional distance between images from an
  edited checkpoint and the matching original-model images.

For an erased target, a **lower target CLIP score** is treated as better
erasure. For non-target artists and MS-COCO, a **lower FID** and a **higher
CLIP score** indicate better preservation.

The main files are:

| File | Contents |
|---|---|
| `detailed_metrics.csv` | per task, model, and content CLIP/FID rows |
| `summary.csv` | target means, non-target means, and MS-COCO metrics |
| `comparison.csv` | legacy and TGPRS side by side; positive `*_improvement` values favor TGPRS |

The comparison formulas are worth reading literally:

```text
target CLIP improvement = legacy target CLIP - TGPRS target CLIP

non-target FID improvement = legacy non-target FID - TGPRS non-target FID

COCO CLIP improvement = TGPRS COCO CLIP - legacy COCO CLIP
```

Thus a positive number consistently favors TGPRS even though the preferred
direction differs by metric.

## 10. Compact pseudocode

```python
for target_name in ["Van Gogh", "Picasso", "Monet"]:
    t = encode_last_subject_token(target_name)
    a = encode_last_subject_token("art")
    neutral = encode_last_subject_tokens(the_100_neutral_phrases)

    # Single-target workflow: 100 target-to-anchor rows.
    G = stack([q - t for q in neutral])
    G_normalized = row_normalize(G)
    _, singular_values, Vh = svd(G_normalized)
    B = Vh[:min(30, numerical_rank(G_normalized))]

    projected_target = (t @ B.T) @ B
    if norm(projected_target) > epsilon:
        direction = -projected_target / norm(projected_target)
    else:
        direction = fallback_direction(B, a - t)
    residual = norm(a - t) * direction

    S_tt = t.T @ t
    D_ta = residual.T @ t

    retain_embeddings = encode_style_retain_set_without(target_name)
    C_R = mean([z.T @ z for z in retain_embeddings])
    U, singular_values, _ = svd(C_R)
    P_R = U[:, singular_values < 0.4] @ U[:, singular_values < 0.4].T

    for W in unet_cross_attention_value_weights:
        delta_W = speed_closed_form_update(W, S_tt, D_ta, P_R)
        save(W + delta_W)
```

## 11. Code map

- Workflow orchestration:
  [`run_all.sh`](../remote_scripts/eval_few/eval_few_style_1/run_all.sh)
- Task/rank resolution:
  [`workflow_config.py`](../remote_scripts/eval_few/eval_few_style_1/workflow_config.py)
- Training invocation:
  [`03_train.sh`](../remote_scripts/eval_few/eval_few_style_1/03_train.sh)
- TGPRS matrix, SVD, projection, and norm matching:
  [`src/residual_subspace.py`](../src/residual_subspace.py)
- SPEED statistics and closed-form layer edits:
  [`train_erase_null.py`](../train_erase_null.py)
- Style and MS-COCO sampling:
  [`sample.py`](../sample.py) and [`sample2.py`](../sample2.py)
- CLIP/FID aggregation:
  [`evaluate_clip_fid.py`](../remote_scripts/eval_few/eval_few_style_1/evaluate_clip_fid.py)
