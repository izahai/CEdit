# Residual Subspace-Based Anchor Construction

## Overview

This document describes the proposed anchor-construction procedure. The
implementation mode is currently named
`negative_target_normalized_residual_subspace`; a paper name has not yet been
chosen.

The method is designed for concept erasure in text-to-image diffusion models.
In SPEED-style editing, a target concept embedding is edited toward an anchor
concept embedding. The anchor provides the semantic direction that replaces
or suppresses the target concept.

The usual construction uses one residual direction for each target-anchor
pair:

```text
r_i = a_i - t_i
```

where `t_i` is a target prompt embedding and `a_i` is its anchor prompt
embedding. This can be restrictive because the residuals may contain several
different directions, while a single residual or an unstructured average can
discard the geometry shared by the examples.

The proposed procedure addresses this by learning a low-rank subspace from the
residuals and then constructing a new anchor direction that explicitly points
away from the target embedding inside that subspace.

## Problem context

Let:

- `t_i` be the embedding of a target prompt containing an erased concept;
- `a_i` be the embedding of the corresponding anchor prompt;
- `r_i = a_i - t_i` be the original target-to-anchor residual;
- `N` be the number of target-anchor pairs;
- `D` be the flattened embedding dimension.

The editing objective uses statistics derived from these residuals. A useful
anchor should satisfy three properties:

1. It should lie in directions supported by the target-anchor residual data.
2. It should move away from the target concept rather than reinforce it.
3. It should preserve a scale comparable to the original residual so that the
   editing strength does not change unexpectedly.

The method therefore separates direction selection from magnitude selection:

- the residual examples determine the direction subspace;
- the negative target projection determines the direction within the subspace;
- the original residual norm determines the magnitude.

## Subspace construction

### 1. Compute residual embeddings

For every target-anchor pair, compute:

\[
r_i = a_i - t_i.
\]

Flatten each residual into a vector in `R^D`.

### 2. Normalize every residual before SVD

Compute the residual norm:

\[
\rho_i = \lVert r_i \rVert_2.
\]

Then normalize each residual independently:

\[
\tilde r_i = \frac{r_i}{\max(\rho_i, \varepsilon)}.
\]

This prevents large-norm residuals from dominating the subspace purely because
of magnitude. The SVD therefore captures variation in residual directions
rather than mainly variation in residual lengths.

Stack the normalized residuals into a matrix:

\[
R_{norm} =
\begin{bmatrix}
\tilde r_1^T\\
\tilde r_2^T\\
\vdots\\
\tilde r_N^T
\end{bmatrix}
\in \mathbb{R}^{N \times D}.
\]

### 3. Truncated SVD

Compute:

\[
R_{norm} = U\Sigma V^T.
\]

For a requested subspace rank `k`, retain the first `k` right singular
vectors:

\[
B = V_{1:k} \in \mathbb{R}^{k \times D}.
\]

The rows of `B` form an orthonormal basis for the normalized residual
subspace. In the experiment, `k` is swept over:

```text
10, 20, 30, ..., 100
```

The valid range is:

```text
1 <= k <= min(N, D)
```

## New anchor construction

For every target embedding, project the target onto the learned residual
subspace:

\[
p_i = P_B(t_i) = (t_i B^T)B.
\]

The negative target direction is:

\[
d_i = -\frac{p_i}{\lVert p_i \rVert_2}.
\]

This direction is inside the residual subspace but points opposite to the
target embedding. It is therefore an explicit negative-target anchor
direction.

To preserve the scale of the original target-anchor relationship, multiply
the unit direction by the original residual norm:

\[
\Delta_i = \rho_i d_i.
\]

The resulting new anchor embedding is:

\[
a_i^{new} = t_i + \Delta_i.
\]

In the implementation, `Delta_i` is returned as the replacement residual and
is then used by the SPEED edit-statistics calculation.

## Numerical fallbacks

If the target projection is approximately zero, the method uses the following
fallbacks:

1. Project the original residual `r_i` onto the learned subspace.
2. If that projection is also zero, use the first basis vector of `B`.

These fallbacks ensure that every example receives a valid finite direction.
The implementation records the number of fallback cases in diagnostics.

## Difference from smallest-cosine subspace anchoring

The two methods share the idea of constructing a residual subspace, but they
choose directions differently:

| Method | Residual preprocessing | Direction inside subspace |
| --- | --- | --- |
| Smallest-cosine subspace | SVD of selected raw residuals | Negative normalized target projection |
| Proposed procedure | Normalize every residual before SVD | Negative normalized target projection |

The central design choice is the normalization before SVD. This makes the
learned subspace less sensitive to residual magnitude and more focused on the
directional structure of the residual embeddings. The final paper terminology
can be selected after comparing this procedure with the other anchor
construction baselines.

## Code entry points

The method is implemented through:

```text
src/residual_subspace.py
build_negative_target_normalized_residual_subspace_residuals(...)
```

and selected with:

```yaml
anchor_mode: negative_target_normalized_residual_subspace
residual_rank: ${SUBSPACE_K}
```
