# Target-Global Pairwise Residual Subspace (TGPRS)

## Purpose of this document

This document is a method-level specification of
`target_global_pairwise_residual_subspace` as implemented in this repository.
It is intended to serve as source material for writing a scientific paper. It
separates the general algorithm from the particular hyperparameters used in
the current Stable Diffusion v1.4 experiments and avoids making empirical
claims that are not part of the method definition.

TGPRS is not a standalone model-editing optimizer. It is a replacement for the
target-to-anchor residual construction used inside SPEED. The retain-set
estimation and the closed-form SPEED update remain unchanged.

## 1. Motivation

In anchor-based concept erasure, an erase target with text embedding
\(t_i\in\mathbb{R}^{D}\) is associated with an anchor embedding
\(a_i\in\mathbb{R}^{D}\). Legacy SPEED uses the direct displacement

\[
\ell_i=a_i-t_i
\]

as the semantic residual for target \(i\). This treats each target-anchor pair
independently. In multi-concept erasure, it does not explicitly use the
relations among all concepts in the erase set, and the geometry of the edit is
largely determined by the selected anchors.

TGPRS uses the erase set itself to estimate a shared, low-rank space of
semantically supported directions. It constructs all directed differences
between erase targets, adds directed differences from each target to a small
set of reference anchors, and compresses these directions using a
row-normalized truncated singular value decomposition (SVD). The resulting
subspace is shared globally, while the final residual remains target-specific.

The design separates two roles:

- **direction** is inferred from the global geometry of the erase targets and
  reference anchors;
- **magnitude** is inherited from the corresponding legacy target-to-anchor
  residual.

This separation constrains the edit directions without increasing their norms
relative to the original anchor construction.

## 2. Notation and inputs

Let

\[
T=\{t_1,\ldots,t_N\}, \qquad t_i\in\mathbb{R}^{D},
\]

be the embeddings of the \(N\) concepts erased jointly. Let

\[
A=\{a_1,\ldots,a_N\}
\]

be their conventional SPEED anchors, and let

\[
Q=\{q_1,\ldots,q_L\}
\]

be the extra anchors used to estimate the global residual subspace. The two
anchor collections have different roles:

- \(a_i\) determines the magnitude of the final residual for target \(i\);
- the \(q_l\) vectors contribute directions to the shared subspace.

They may overlap. In the current celebrity experiments, every conventional
anchor is `person`, while the default extra-anchor prompts are the empty prompt
and `person`.

The implementation accepts embeddings with additional non-batch dimensions.
It flattens every embedding to dimension \(D\) during subspace construction
and restores the original shape afterward. For ordinary instance and celebrity
erasure in this repository, the input is the CLIP hidden state at the final
subject token, so \(D=768\) for Stable Diffusion v1.4.

## 3. Method

### 3.1 Directed target-global residual matrix

For each source target \(t_i\), TGPRS constructs residuals to every other erase
target and to every extra anchor:

\[
\mathcal{G}_i=
\{t_j-t_i\mid j\ne i\}
\;\cup\;
\{q_l-t_i\mid 1\le l\le L\}.
\]

The residuals are stacked by source target to form

\[
G=
\begin{bmatrix}
\mathcal{G}_1\\
\vdots\\
\mathcal{G}_N
\end{bmatrix}
\in\mathbb{R}^{M\times D},
\qquad
M=N(N-1+L).
\]

The target-to-target construction is directed: both \(t_j-t_i\) and
\(t_i-t_j\) are included. These two rows have opposite signs. They do not
expand the linear span, but their inclusion gives both source-target
orientations equal weight in the SVD objective.

With the default two extra anchors, \(L=2\), and therefore

\[
M=N(N+1).
\]

For \(N=10\), \(50\), and \(100\), the matrix contains \(110\), \(2{,}550\),
and \(10{,}100\) rows, respectively.

### 3.2 Row normalization and truncated SVD

Each global residual is normalized independently before factorization:

\[
\widehat g_m=
\frac{g_m}{\max(\lVert g_m\rVert_2,\varepsilon)},
\]

where \(\varepsilon>0\) is a numerical stability constant. Stacking the
normalized rows gives \(\widehat G\). TGPRS computes the thin SVD

\[
\widehat G=U\Sigma V^\top.
\]

For a requested rank \(k\), let \(r_{\mathrm{eff}}\) be the numerical rank of
\(\widehat G\). The implemented basis rank is

\[
\bar{k}=\min(k,r_{\mathrm{eff}}),
\]

and the basis is formed from the leading right singular vectors:

\[
B=V_{1:\bar{k}}^\top\in\mathbb{R}^{\bar{k}\times D}.
\]

The rows of \(B\) are orthonormal. For a row vector \(x\), its orthogonal
projection into the learned subspace is

\[
\Pi_B(x)=(xB^\top)B.
\]

Normalizing before SVD makes the factorization emphasize frequently supported
directions rather than large Euclidean displacements. Without normalization,
long residuals would contribute quadratically more energy to the SVD.

The requested rank must satisfy \(1\le k\le\min(M,D)\). If the numerical rank
is smaller than \(k\), the implementation uses the effective rank. A global
residual matrix with numerical rank zero is rejected.

Because all rows of \(G\) are differences among at most \(N+L\) points, its
exact rank is bounded by

\[
\operatorname{rank}(G)\le N+L-1.
\]

Thus, \(k\) is a requested upper bound rather than a guarantee that the learned
basis will contain exactly \(k\) dimensions. For example, with ten targets and
the default two extra anchors, the exact rank cannot exceed eleven even if
`residual_rank` is set to 30.

### 3.3 Target-specific direction selection

For target \(t_i\), TGPRS projects the absolute target embedding into the
shared basis:

\[
p_i=\Pi_B(t_i).
\]

When \(\lVert p_i\rVert_2>\varepsilon\), the target-specific unit direction is
chosen opposite to this projection:

\[
u_i=-\frac{p_i}{\lVert p_i\rVert_2}.
\]

This direction minimizes the inner product with the target among unit vectors
in the learned subspace:

\[
u_i
=\arg\min_{u\in\operatorname{span}(B),\,\lVert u\rVert_2=1}
\langle t_i,u\rangle,
\]

provided that \(\Pi_B(t_i)\ne 0\). Hence, all targets use one global subspace,
but they need not share one residual direction or one effective anchor.

### 3.4 Legacy-norm matching

The legacy residual norm is

\[
\rho_i=\lVert a_i-t_i\rVert_2.
\]

TGPRS combines this magnitude with the target-specific direction:

\[
\Delta_i=\rho_i u_i.
\]

The corresponding implicit anchor is

\[
a_i^{\mathrm{TGPRS}}=t_i+\Delta_i.
\]

Up to floating-point error, the construction satisfies

\[
\Delta_i\in\operatorname{span}(B),
\qquad
\lVert\Delta_i\rVert_2=\lVert a_i-t_i\rVert_2.
\]

Consequently, if the TGPRS residuals are stacked into
\(\Delta\in\mathbb{R}^{N\times D}\), then

\[
\operatorname{rank}(\Delta)\le\bar{k}.
\]

An optional scalar `residual_scale` \(s\) is applied after TGPRS constructs
\(\Delta_i\). The residual actually passed to the SPEED statistic is then
\(s\Delta_i\). Norm equality with the legacy residual therefore holds before
this optional scaling, and also afterward when \(s=1\).

### 3.5 Deterministic fallbacks

The negative projected target is undefined when \(\Pi_B(t_i)\) has near-zero
norm. The implementation handles this case deterministically:

1. If \(\lVert\Pi_B(t_i)\rVert_2\le\varepsilon\) but
   \(\lVert\Pi_B(\ell_i)\rVert_2>\varepsilon\), use

   \[
   u_i=
   \frac{\Pi_B(\ell_i)}{\lVert\Pi_B(\ell_i)\rVert_2}.
   \]

2. If both projections have near-zero norm, use the first basis vector,
   \(u_i=B_1\).

In every branch, the direction lies in \(\operatorname{span}(B)\) and is
multiplied by \(\rho_i\). The code reports counts for both fallback paths as
well as maximum norm and subspace-projection errors.

## 4. Integration with SPEED

Let the target embeddings and TGPRS residuals be represented as row vectors.
SPEED forms the target second moment

\[
C_T=\frac{1}{N}\sum_{i=1}^{N}t_i^\top t_i
\]

and the target-residual interaction

\[
D_{\mathrm{TGPRS}}
=\frac{1}{N}\sum_{i=1}^{N}(s\Delta_i)^\top t_i.
\]

Legacy SPEED uses the same expression with \(a_i-t_i\) in place of
\(\Delta_i\). TGPRS changes only this residual-dependent statistic. Since

\[
D_{\mathrm{TGPRS}}=\frac{s}{N}\Delta^\top T,
\]

its rank is also bounded by the learned basis rank:

\[
\operatorname{rank}(D_{\mathrm{TGPRS}})\le\bar{k}.
\]

The remainder of the SPEED update is unchanged. For completeness, the current
implementation first constructs a retain second-moment matrix \(C_R\), obtains
the projector \(P_R\) onto singular directions of \(C_R\) whose singular
values fall below the configured threshold, and applies the existing
closed-form update. Using the names \(\gamma=\texttt{retain_scale}\),
\(\lambda=\texttt{lamb}\), and \(K_2\) for SPEED's null-text constraint
matrix, the code computes

\[
M_R=(C_TP_R+\gamma I)^{-1}
\]

and, for an edited layer with weight \(W\),

\[
\Delta W=
W D_{\mathrm{TGPRS}} P_R
\left[
I-M_RK_2
\left(K_2^\top P_RM_RK_2+\lambda I\right)^{-1}
K_2^\top P_R
\right]
M_R.
\]

The edited parameter is \(W'=W+\Delta W\). In the reported TGPRS
configuration, `params: V`, so the edited parameters are the cross-attention
value projections (`attn2.to_v`). This choice is an experimental setting, not
an intrinsic restriction of the residual construction.

The TGPRS basis projector \(\Pi_B\) and SPEED's retain projector \(P_R\) must
not be conflated:

- \(\Pi_B\) is learned once from normalized target-global residuals, retains
  leading directions up to a requested rank, and constructs the target
  residuals;
- \(P_R\) is learned from the retain distribution using a singular-value
  threshold and constrains the closed-form layer update.

## 5. Algorithm

```text
Input:
  target embeddings T = {t_i}_{i=1}^N
  conventional anchor embeddings A = {a_i}_{i=1}^N
  extra-anchor embeddings Q = {q_l}_{l=1}^L
  requested rank k, residual scale s, numerical constant epsilon

1. Construct the global residual matrix G:
   for each source target t_i:
       append t_j - t_i for every j != i
       append q_l - t_i for every extra anchor q_l

2. Normalize every row of G independently.

3. Compute a thin SVD of the normalized matrix and retain the leading
   min(k, effective_rank) right singular vectors as an orthonormal basis B.

4. For each target t_i:
       rho_i <- ||a_i - t_i||_2
       p_i   <- projection of t_i onto span(B)

       if ||p_i||_2 > epsilon:
           u_i <- -p_i / ||p_i||_2
       else if ||projection of (a_i - t_i) onto span(B)||_2 > epsilon:
           u_i <- normalized projection of (a_i - t_i)
       else:
           u_i <- first basis vector

       Delta_i <- rho_i * u_i

5. Form D_TGPRS = mean_i [(s * Delta_i)^T t_i].

6. Supply C_T and D_TGPRS to the unchanged SPEED closed-form update.

Output:
  edited model parameters and residual/subspace diagnostics
```

## 6. Computational properties

The explicit residual matrix has \(M=N(N-1+L)\) rows. Its construction costs
\(O(MD)\) time and memory. A dense thin SVD costs

\[
O\!\left(\min(M^2D,MD^2)\right),
\]

and projecting all targets costs \(O(N\bar{k}D)\). The SVD is performed once
for the ordinary TGPRS mode, before the layer-editing loop.

The current implementation materializes the full directed matrix and uses
`torch.linalg.svd`. It does not use randomized SVD, chunked covariance
accumulation, or deduplication of opposite target-pair rows. Such changes may
improve scalability but are not equivalent implementation details unless their
weighting is matched carefully.

## 7. Current experimental configuration

The paper-comparison workflow currently uses:

```yaml
sd_ckpt: CompVis/stable-diffusion-v1-4
anchor_concepts:
  - person
anchor_mode: target_global_pairwise_residual_subspace
residual_rank: 30
residual_scale: 1.0
baseline: SPEED
params: V
aug_num: 0
threshold: 0.0001
retain_scale: 0.05
lamb: 0.0
disable_filter: true
seed: 0
```

Unless explicitly overridden, global residual-subspace modes use two extra
anchor prompts:

```text
""        (empty prompt)
"person"
```

The CLI option `--subspace_anchor_concepts` can replace this list or can be
provided with no values to construct the basis from target-to-target residuals
only. Therefore, the empty prompt and `person` are defaults used by the current
experiments, not mathematical requirements of TGPRS.

For the default \(L=2\) construction and `residual_rank: 30`, note that the
ten-target problem can have basis rank at most eleven. The 50- and 100-target
problems can realize rank 30 if their normalized residual matrices have
sufficient numerical rank.

## 8. Interpretation and important caveats

### What the method guarantees

- Every nonzero output residual has the legacy residual magnitude before
  optional global scaling.
- Every output residual lies in the learned target-global subspace.
- The stacked residual matrix and the residual-dependent SPEED statistic have
  rank at most \(\bar{k}\).
- The basis is constructed only from erase-target embeddings and configured
  extra-anchor embeddings; it does not use the retain dataset.
- The ordinary TGPRS basis is layer-independent and is computed once.

### What the method does not guarantee

- A rank-\(k\) request does not guarantee an effective rank of \(k\).
- Opposing the projected text embedding does not by itself prove concept
  erasure in the image distribution.
- Matching residual norms does not guarantee equal update norms after the
  retain projection and the remaining SPEED matrix operations.
- A target-derived subspace can overlap with semantics needed for retained
  concepts; retention must be evaluated empirically.
- The quadratic number of pairwise rows may become expensive for very large
  erase sets.

### Coordinate-origin subtlety

The basis learned from pairwise differences is invariant to translating all
input points by the same vector. However, the direction-selection step projects
the absolute embedding \(t_i\), not a centered target. Consequently, the final
direction \(-\Pi_B(t_i)\) depends on the embedding coordinate origin. This is
the implemented method and should be stated accurately rather than described
as a fully translation-invariant construction.

### Distinction from nearby variants

This document describes exactly
`target_global_pairwise_residual_subspace`.

- `global_pairwise_residual_subspace` may learn the basis from an external
  concept CSV rather than from the erase targets themselves.
- `mean_norm_target_global_pairwise_residual_subspace` replaces the legacy
  residual magnitude with the mean norm of each target's outgoing global
  residuals.
- `retain_aware_target_global_pairwise_residual_subspace` projects the raw
  global residual matrix through a layer-specific retain-low projector before
  normalization and SVD.

These variants should not be described as properties of ordinary TGPRS.

## 9. Implementation mapping and verification

The implementation is concentrated in the following locations:

- `src/residual_subspace.py`
  - `build_global_pairwise_residual_matrix` constructs \(G\);
  - `_build_normalized_residual_basis` normalizes rows and computes the SVD;
  - `_build_norm_matched_directions` selects directions, applies fallbacks, and
    restores magnitudes;
  - `build_target_global_pairwise_residual_subspace_residuals` exposes TGPRS.
- `train_erase_null.py`
  - `encode_last_subject_embeddings` encodes targets and extra anchors;
  - `build_target_anchor_statistics` constructs \(C_T\) and
    \(D_{\mathrm{TGPRS}}\);
  - `edit_model` applies the unchanged SPEED layer update.
- `tests/test_residual_subspace.py` verifies directed pair construction,
  row normalization before SVD, target-only basis inputs, norm preservation,
  subspace membership, rank handling, and fallback behavior.
- `tests/test_target_anchor_statistics.py` and `tests/test_train_config.py`
  verify integration and configuration behavior.

Focused CPU tests can be run with:

```bash
python -m unittest \
  tests.test_residual_subspace \
  tests.test_target_anchor_statistics \
  tests.test_train_config
```

## 10. Recommended terminology for a paper

The following wording is consistent with the implementation:

> We construct a target-global residual matrix from all directed differences
> between erase-target embeddings and from each erase target to a set of fixed
> reference anchors. After independently normalizing the residuals, we extract
> a low-rank directional basis using truncated SVD. For each erase target, we
> select the unit vector opposite to its projection into this shared subspace
> and scale it to match the norm of the original target-to-anchor residual. The
> resulting target-specific residuals replace the legacy residuals in SPEED's
> target-anchor interaction statistic, while the retain projector and
> closed-form model update remain unchanged.

Avoid calling TGPRS a learned optimizer or claiming that it uses the retain set
in its basis construction. It is more precise to describe it as a
**target-conditioned, low-rank residual construction for SPEED**.
