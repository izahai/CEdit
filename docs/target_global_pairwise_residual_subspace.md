# Target-Global Pairwise Residual Subspaces for Scalable Concept Erasure

## Abstract

Concept erasure methods edit a text-to-image diffusion model so that prompts
containing selected concepts no longer reproduce those concepts, while prompts
for unrelated concepts remain useful. In closed-form editors such as SPEED,
the target-to-anchor residual determines the semantic direction of the edit.
Using each target's direct residual to a generic anchor, however, treats the
targets independently and does not explicitly model the geometry shared by a
multi-concept erase set. We introduce **Target-Global Pairwise Residual
Subspace (TGPRS)**, an anchor-construction method that learns a low-rank
directional subspace from the erase targets themselves. TGPRS forms all ordered
differences from every target to every other target and to two fixed anchors,
the empty prompt and `person`. It normalizes every difference before truncated
singular value decomposition, projects each target into the resulting
subspace, reverses that projection, and matches its magnitude to the original
target-to-anchor residual. The construction is target-only: it requires no
external concept vocabulary and introduces no iterative optimization. In the
repository's Stable Diffusion v1.4 experiments, a rank-30 TGPRS configuration
reduces conditional erased-identity accuracy from 2.22% to 1.03%, 3.30% to
1.85%, and 5.70% to 0.61% relative to the evaluated legacy SPEED
configurations at 10, 50, and 100 erased celebrities, respectively. Retained
accuracy is slightly higher at 10 concepts and lower at 50 and 100 concepts.
Because the compared training configurations also differ in retain
augmentation, these results should be read as a system-level comparison rather
than a controlled method-only ablation.

## 1. Introduction

Text-to-image diffusion models can reproduce identities, visual styles, and
other concepts learned from their training data. Concept erasure modifies a
pretrained model to suppress a chosen set of concepts without retraining the
model from scratch. A successful editor must balance two competing goals:

1. **Erasure:** generated images should no longer express a requested target.
2. **Retention:** unrelated concepts and the model's general generation
   ability should remain intact.

SPEED performs efficient closed-form updates to cross-attention projections.
Its target statistic depends on residuals between target text embeddings and
anchor text embeddings. For a target embedding \(t_i\) and anchor embedding
\(a_i\), the conventional residual is

\[
r_i = a_i - t_i.
\]

This construction is local to a single target-anchor pair. In a multi-concept
setting, it does not use the relational structure of the complete erase set.
It also places the full burden of defining an erasure direction on a single
generic anchor such as `person`.

TGPRS instead uses the erase targets as a small, task-specific semantic
dictionary. Pairwise target differences describe how the targets vary, while
differences to the empty prompt and `person` connect that variation to
unconditional and class-level semantics. A normalized SVD compresses these
directions into a global low-rank subspace. Each target then receives a
different residual direction inside the same subspace, chosen to point opposite
to that target and scaled to preserve the magnitude of the original edit.

The method makes four contributions:

- a target-only global subspace that needs no external concept corpus;
- an ordered pairwise construction that captures relations among all erased
  concepts and two fixed semantic endpoints;
- separation of direction and magnitude through row normalization before SVD
  and legacy-norm matching after projection; and
- direct compatibility with SPEED's existing edit statistic and closed-form
  weight update.

## 2. Background and motivation

### 2.1 Text embeddings and SPEED statistics

Let \(N\) be the number of concepts erased together and \(d\) the text
embedding dimension. In the implemented celebrity setting, the text encoder's
last subject-token embedding is used, so \(t_i,a_i\in\mathbb{R}^{1\times d}\)
with \(d=768\) for Stable Diffusion v1.4. The method also supports tensors with
additional non-batch dimensions by flattening them during subspace
construction.

SPEED aggregates the target embeddings into

\[
S_{tt}=\frac{1}{N}\sum_{i=1}^{N}t_i^{\top}t_i
\]

and the target-to-anchor interaction into

\[
D_{ta}=\frac{1}{N}\sum_{i=1}^{N}r_i^{\top}t_i.
\]

The retain-set projection and the remaining closed-form model update use these
statistics. TGPRS leaves \(S_{tt}\), retain-set estimation, and the SPEED
weight-update rule unchanged. It replaces only \(r_i\) in \(D_{ta}\).

### 2.2 How the original SPEED code computes its low-singular projector

The original SPEED update already contains an SVD-based subspace, but it is
different from the TGPRS residual subspace. For each edited cross-attention
layer, SPEED first collects retain embeddings \(z_1,\ldots,z_L\in
\mathbb{R}^{1\times d}\) and forms their uncentered second-moment matrix

\[
C_R=\frac{1}{L}\sum_{\ell=1}^{L}z_\ell^{\top}z_\ell
\in\mathbb{R}^{d\times d}.
\]

In `train_erase_null.py`, this matrix is named `sum_ret_ret`. It is symmetric
positive semidefinite, so its singular vectors are also its eigenvectors and
its singular values are nonnegative eigenvalues. The code computes

\[
C_R=U_R\Sigma_R V_R^{\top}
\]

with `torch.svd`, selects the columns of \(U_R\) whose singular values are
strictly below the absolute cutoff \(\tau=\texttt{threshold}\), and constructs

\[
U_{\mathrm{low}}=U_R[:,\,\sigma_j<\tau],
\qquad
P_R=U_{\mathrm{low}}U_{\mathrm{low}}^{\top}.
\]

The implementation is the two-line operation

```python
U, S, V = torch.svd(sum_ret_ret)
P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
```

Thus, \(P_R\) is an orthogonal projector onto the directions with low energy
under the empirical retain distribution: directions that are approximately in
the null space of \(C_R\). The selection is threshold-based, not a request for
a fixed rank, and `threshold` is an absolute singular-value cutoff. Therefore,
the rank of \(P_R\) is

\[
\operatorname{rank}(P_R)=\#\{j:\sigma_j(C_R)<\tau\},
\]

which can vary with the retain set, augmentation, numerical scale, and layer.
If no singular value is below the cutoff, the selected basis has zero columns
and \(P_R\) is the zero matrix. The experiments in this document set
`threshold: 0.0001`.

The retain matrix can itself be layer-dependent. When `aug_num > 0`, SPEED
augments retain embeddings using

\[
P_{W,\min}=v_{W,\min}v_{W,\min}^{\top},
\]

where \(v_{W,\min}\) is the last right singular vector returned by the reduced
`torch.svd` of the current layer weight \(W\). Random perturbations are projected
through \(P_{W,\min}\), filtered by their response to the preliminary erase
update, and included when accumulating \(C_R\). This
`P0_min` projector is used only to generate retain augmentations; it is not the
low-singular retain projector \(P_R\) used in the final closed-form update.
With the reported TGPRS setting, `aug_num: 0`, no such augmented embeddings are
added.

Finally, the code inserts \(P_R\) into

\[
M=(S_{tt}P_R+\gamma I)^{-1},
\]

and into the remaining SPEED correction terms, where \(\gamma\) is
`retain_scale`. Consequently, SPEED's \(P_R\) constrains the model update using
low-energy **retain** directions. It should not be confused with TGPRS's
\(P_B=B^{\top}B\), which is learned from normalized target-pair residuals,
keeps the leading singular directions at a requested rank, and is used to
construct \(\Delta_i\). TGPRS changes the target-anchor statistic but leaves
the original SPEED computation of \(C_R\), \(P_R\), and \(M\) intact.

### 2.3 Why a global pairwise subspace?

A residual matrix formed only from \(a_i-t_i\) has one row per target. With a
shared class anchor, those rows encode how each identity differs from the same
endpoint, but they omit direct relations among identities. TGPRS adds every
ordered target-to-target displacement. This creates a richer set of supported
directions while retaining a compact representation through a rank-\(k\)
subspace.

Normalizing each displacement before SVD is essential to the intended
geometry. Without normalization, long differences contribute quadratically
more energy and can dominate the singular vectors. Row normalization makes the
basis describe prevalent *directions*, while the magnitude of each final edit
is restored separately from its original residual.

## 3. Method

### 3.1 Notation

Let

\[
T=\{t_1,\ldots,t_N\}
\]

be the erase-target embeddings. Let \(q_0\) denote the empty-prompt embedding
and \(q_p\) the embedding of `person`. The conventional anchor for target
\(i\) is \(a_i\); in the reported celebrity experiments, \(a_i=q_p\) for all
\(i\). Define the legacy magnitude

\[
\rho_i=\lVert a_i-t_i\rVert_2.
\]

### 3.2 Ordered global residual matrix

For every source target \(t_i\), TGPRS constructs differences to all other
targets and both fixed anchors:

\[
\mathcal{G}_i =
\{t_j-t_i\mid j\ne i\}
\cup\{q_0-t_i,\ q_p-t_i\}.
\]

Stacking the sets in source-target order gives

\[
G=
\begin{bmatrix}
\mathcal{G}_1\\
\vdots\\
\mathcal{G}_N
\end{bmatrix}
\in\mathbb{R}^{M\times D},
\qquad
M=N[(N-1)+2]=N(N+1),
\]

where \(D\) is the flattened embedding dimension. The target-to-target block
is ordered: if \(t_j-t_i\) is present, then \(t_i-t_j\) is also present. These
opposite rows do not change the linear span, but their joint presence gives
each unordered target relation equal treatment from both source directions.

For 10, 50, and 100 targets, the construction produces 110, 2,550, and 10,100
rows, respectively.

### 3.3 Direction-normalized truncated SVD

Each row \(g_m\) is normalized independently:

\[
\tilde g_m=
\frac{g_m}{\max(\lVert g_m\rVert_2,\varepsilon)}.
\]

The method computes the exact thin SVD

\[
\tilde G=U\Sigma V^{\top}.
\]

Let \(r_{\mathrm{eff}}\) be the numerical rank estimated with the standard
floating-point tolerance used by the implementation. For requested rank
\(k\), the actual basis rank is

\[
\hat{k}=\min(k,r_{\mathrm{eff}}),
\qquad
B=V_{1:\hat{k}}^{\top}\in\mathbb{R}^{\hat{k}\times D}.
\]

The rows of \(B\) are orthonormal. The corresponding projector is

\[
P_B=B^{\top}B,
\]

or, for row vectors as implemented,

\[
\Pi_B(x)=(xB^{\top})B.
\]

The requested rank must be positive and cannot exceed
\(\min(M,D)\). A numerically zero-rank global matrix is rejected.
Because every row is a difference among \(N+2\) embedded points, the exact
linear span has dimension at most \(N+1\). Thus, `residual_rank: 30` is a
*requested* rank: the 10-target experiment can realize at most rank 11, while
the implementation automatically caps the basis at the measured effective
rank.

### 3.4 Target-opposing, norm-matched residuals

For each target, project its flattened embedding into the global subspace:

\[
p_i=\Pi_B(t_i).
\]

When \(\lVert p_i\rVert_2>\varepsilon\), select the unit direction opposite to
the projected target:

\[
u_i=-\frac{p_i}{\lVert p_i\rVert_2}.
\]

The TGPRS residual is

\[
\Delta_i=\rho_i u_i,
\]

and its implicit new anchor is

\[
a_i^{\mathrm{TGPRS}}=t_i+\Delta_i.
\]

This gives two invariants up to numerical precision:

\[
\Delta_i\in\operatorname{span}(B),
\qquad
\lVert\Delta_i\rVert_2=\lVert a_i-t_i\rVert_2.
\]

The first constrains edits to directions supported by the global target
geometry. The second prevents rank selection from unintentionally changing the
per-target edit scale. An optional global `residual_scale` is applied only
after this construction; it is 1.0 in the reported experiments.

### 3.5 Degenerate projections

The implementation uses deterministic fallbacks when a negative target
projection cannot define a direction:

1. If \(\Pi_B(t_i)\) is near zero, use the normalized projection of the legacy
   residual, \(\Pi_B(a_i-t_i)\), when nonzero.
2. If both projections are near zero, use the first basis vector of \(B\).

The selected direction is always multiplied by \(\rho_i\). Diagnostics report
target-projection, legacy-projection, and first-basis fallback counts, as well
as maximum norm and projection errors.

### 3.6 Integration with SPEED

TGPRS replaces the legacy interaction statistic with

\[
D_{\mathrm{TGPRS}}
=\frac{1}{N}\sum_{i=1}^{N}\Delta_i^{\top}t_i.
\]

This statistic is passed to the same SPEED update used by the legacy anchor
mode. Only the value-projection matrices (`attn2.to_v`) are edited in the
reported configuration. Thus, TGPRS is an anchor-residual construction rather
than a separate model-editing optimizer.

### 3.7 Algorithm

```text
Input: targets T, conventional anchors A, fixed anchors {empty, person}, rank k

1. Flatten the embeddings.
2. For each target t_i:
     append t_j - t_i for every j != i
     append empty - t_i
     append person - t_i
3. Normalize every appended row to unit length.
4. Compute a thin SVD and retain the leading k effective right singular vectors.
5. For each target t_i:
     rho_i <- ||a_i - t_i||_2
     p_i   <- projection of t_i onto the retained basis
     u_i   <- normalized -p_i, with deterministic fallbacks if p_i is zero
     Delta_i <- rho_i u_i
6. Form the SPEED interaction statistic from Delta_i^T t_i.
7. Build SPEED's low-singular retain projector P_R from the retain second moment.
8. Apply the unchanged SPEED closed-form weight update using P_R.

Output: edited cross-attention value projections
```

## 4. Computational analysis

The residual matrix contains \(M=N(N+1)\) rows. Explicit construction therefore
uses \(O(N^2D)\) memory and arithmetic before factorization. A thin exact SVD
costs approximately

\[
O(\min(M^2D,MD^2)).
\]

For last-token Stable Diffusion v1.4 embeddings, \(D=768\), while the largest
reported experiment has \(M=10{,}100\). The SVD is therefore the main
additional preprocessing cost at large \(N\), although it is performed once
per model edit and is separate from image generation. The final projection of
all targets costs \(O(NkD)\).

The explicit quadratic matrix is simple and exact but is not the only possible
implementation. Chunked covariance accumulation, randomized SVD, or removal of
sign-duplicate target pairs could reduce memory or runtime. Those variants are
not part of the evaluated implementation and may change singular-value
weighting.

## 5. Experimental setup

### 5.1 Model and benchmarks

The repository evaluates Stable Diffusion v1.4 on the 10-, 50-, and
100-celebrity benchmarks. Each benchmark contains an erase split and a retained
identity split. Five hundred images are evaluated for each split and each
benchmark. CE-Eval's GIPHY Celebrity Detector supplies face detection and
identity matching.

Three systems are available in the unified results: the unedited model, legacy
SPEED, and TGPRS-SPEED. The main table below compares the two edited systems.
Lower erased-identity accuracy is better; higher retained-identity accuracy is
better.

### 5.2 Training configuration

The evaluated TGPRS configuration is:

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

Legacy SPEED uses the same base model, anchor, edited parameter family,
threshold, retain scale, regularization, and seed, but uses `anchor_mode:
legacy` and `aug_num: 10`. TGPRS uses `aug_num: 0`. Consequently, the available
comparison changes both residual construction and retain augmentation. A
future controlled ablation should hold `aug_num` fixed.

### 5.3 Metrics

Let \(n\) be the number of generated images, \(n_d\) the number with a detected
face, and \(n_c\) the number whose top-1 detected identity is correct. The
reported metrics are:

\[
\text{Face detection rate}=n_d/n,
\]

\[
\text{conditional identity accuracy}=n_c/n_d,
\]

and

\[
\text{identity hit rate}=n_c/n.
\]

Conditional accuracy matches the repository's principal CE-Eval comparison.
Identity hit rate treats failed detections as incorrect. For a compact measure
of the erasure-retention trade-off, we additionally report

\[
H=\frac{2(1-\mathrm{Acc}_e)\mathrm{Acc}_r}
{(1-\mathrm{Acc}_e)+\mathrm{Acc}_r},
\]

computed from conditional erased accuracy \(\mathrm{Acc}_e\) and conditional
retained accuracy \(\mathrm{Acc}_r\).

## 6. Results

### 6.1 Conditional CE-Eval identity accuracy

| Erased celebrities | Method | Erase accuracy ↓ | Retain accuracy ↑ | Harmonic score ↑ |
| ---: | --- | ---: | ---: | ---: |
| 10 | Legacy SPEED | 2.22% | 91.85% | 94.72% |
| 10 | TGPRS-SPEED | **1.03%** | **92.06%** | **95.39%** |
| 50 | Legacy SPEED | 3.30% | **91.45%** | **94.00%** |
| 50 | TGPRS-SPEED | **1.85%** | 89.61% | 93.69% |
| 100 | Legacy SPEED | 5.70% | **87.85%** | 90.96% |
| 100 | TGPRS-SPEED | **0.61%** | 86.99% | **92.78%** |

TGPRS strengthens erasure at every tested scale. The absolute reductions in
conditional erased accuracy are 1.19, 1.45, and 5.09 percentage points for 10,
50, and 100 targets. The relative reductions are approximately 54%, 44%, and
89%, respectively. The strongest scaling result occurs at 100 concepts, where
TGPRS reduces residual identity recognition by nearly an order of magnitude.

Retention shows a scale-dependent trade-off. At 10 concepts, TGPRS improves
retained accuracy by 0.21 percentage points. At 50 and 100 concepts, it lowers
retained accuracy by 1.84 and 0.86 points. The harmonic score therefore
improves at 10 and 100 concepts but is 0.31 points lower at 50 concepts.

### 6.2 Detection-inclusive identity hit rate

| Erased celebrities | Method | Erase hit rate ↓ | Retain hit rate ↑ | Harmonic score ↑ |
| ---: | --- | ---: | ---: | ---: |
| 10 | Legacy SPEED | 2.20% | 90.20% | 93.85% |
| 10 | TGPRS-SPEED | **1.00%** | **90.40%** | **94.50%** |
| 50 | Legacy SPEED | 3.20% | **89.80%** | **93.17%** |
| 50 | TGPRS-SPEED | **1.80%** | 88.00% | 92.82% |
| 100 | Legacy SPEED | 5.60% | **86.80%** | 90.44% |
| 100 | TGPRS-SPEED | **0.60%** | 85.60% | **91.99%** |

The detection-inclusive results support the same conclusion: TGPRS consistently
improves erasure, with a modest retention cost at 50 and 100 concepts. Face
detection rates remain high across the edited runs (97.0%--99.0% for legacy and
97.2%--98.8% for TGPRS), so failed detection does not explain the large
100-concept erasure gain.

## 7. Discussion

### 7.1 Why the method may scale

The number of pairwise observations grows quadratically with the number of
targets, but the retained representation remains rank \(k\). Adding targets
therefore provides more evidence about the geometry of the erase set without
increasing the final subspace dimension. At 100 concepts, 10,100 normalized
directions inform a rank-30 basis. This may explain why the largest measured
erasure improvement appears in the 100-concept experiment.

The final directions are target-specific even though the basis is shared. A
single shared residual would push every target identically; TGPRS instead uses
\(-\Pi_B(t_i)\), so each target follows the component of its own representation
supported by the common subspace.

### 7.2 Role of the fixed anchors

Target-to-target differences capture within-set variation but are invariant to
a global translation of all target embeddings. The empty prompt and `person`
provide semantic reference points outside those relative differences. The
empty prompt connects the basis to unconditional text conditioning, while
`person` supplies a broad class-level endpoint for celebrity erasure.

The anchor choices are currently fixed in code for the global modes. Other
domains may need different class anchors. For example, artist, object, or style
erasure may benefit from domain-specific endpoints rather than `person`.

### 7.3 Direction versus magnitude

TGPRS deliberately estimates direction twice and magnitude once:

- normalized pairwise rows determine which directions enter the basis;
- the negative target projection selects a direction for each target; and
- the legacy target-anchor norm sets edit magnitude.

This separation makes the method easier to interpret. Improvements cannot be
attributed merely to uniformly increasing residual norms when
`residual_scale=1.0`, because every TGPRS residual matches its corresponding
legacy norm.

## 8. Limitations and threats to validity

The present evidence has several limitations.

1. **Configuration confound.** Legacy SPEED uses `aug_num=10`, whereas TGPRS
   uses `aug_num=0`. The results do not isolate the subspace construction from
   retain augmentation.
2. **Single seed.** All reported runs use seed 0. Variance across seeds is
   unknown.
3. **Single concept domain.** The evaluation covers celebrity identities only;
   generalization to objects, styles, nudity, or other implicit concepts is not
   established.
4. **Single rank.** The comparison uses \(k=30\). A rank sweep is needed to
   characterize the erasure-retention frontier and sensitivity to \(N\).
5. **Quadratic construction.** Materializing \(N(N+1)\) residuals may become
   expensive for much larger erase sets.
6. **No image-quality metrics in the unified comparison.** The available
   summary measures identity erasure and retention, but not FID, CLIP score,
   prompt fidelity, or human preference.
7. **Detector dependence.** Celebrity accuracy inherits the biases and failure
   modes of face detection and the GIPHY identity classifier.
8. **Target-derived basis.** The basis is intentionally fitted to the erase
   set. It may include directions shared with nearby retained identities,
   contributing to the observed retention loss.

The minimum next experiment is a factorial comparison of legacy and TGPRS at
both `aug_num=0` and `aug_num=10`, repeated over several seeds. Rank sweeps and
image-quality evaluation should follow before making broad state-of-the-art
claims.

## 9. Reproducibility and implementation mapping

The implementation entry points are:

- `src/residual_subspace.py`:
  `build_global_pairwise_residual_matrix`,
  `build_global_pairwise_residual_subspace_residuals`, and
  `build_target_global_pairwise_residual_subspace_residuals`;
- `train_erase_null.py`: target/anchor encoding, construction of
  \(D_{\mathrm{TGPRS}}\), retain second-moment accumulation, construction of
  the low-singular projector \(P_R\), and the SPEED weight update;
- `remote_scripts/eval_target_global_pairwise_residual_subspace/`: standalone
  10/50/100-celebrity workflow;
- `remote_scripts/eval_paper_comparison/`: original/legacy/TGPRS comparison;
  and
- `summary.csv`: unified per-split evaluation results used in this paper.

The focused implementation tests verify ordered pair construction,
normalization before exact SVD, target-only basis data, legacy-norm
preservation, subspace membership, both numerical fallbacks, residual counts,
and invalid-rank behavior. They can be run with:

```bash
python -m unittest tests.test_residual_subspace \
  tests.test_target_anchor_statistics \
  tests.test_train_config
```

The full GPU workflow is documented under
`remote_scripts/eval_target_global_pairwise_residual_subspace/README.md`.

## 10. Conclusion

TGPRS reframes multi-concept anchor construction as low-rank geometric
modeling of the erase set. It combines normalized pairwise target relations,
two fixed semantic endpoints, target-opposing projections, and legacy-norm
matching in a form that plugs directly into SPEED. The available celebrity
experiments show consistently stronger erasure and especially large gains at
100 concepts, with small but non-negligible retention trade-offs at larger
scales. The method is simple, interpretable, and optimization-free, but a
controlled augmentation ablation, multiple seeds, rank sweeps, and broader
quality evaluation are required to establish its independent contribution.

## References

1. Li, O., Wang, Y., Hu, X., Jiang, H., Liang, T., Hao, Y., Ma, G., and Feng,
   F. **SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion
   Models.** The Fourteenth International Conference on Learning
   Representations, 2026. [OpenReview](https://openreview.net/forum?id=aoEtzdRkGh).
2. Rombach et al. **High-Resolution Image Synthesis with Latent Diffusion
   Models.** Proceedings of CVPR, 2022.
3. Radford et al. **Learning Transferable Visual Models From Natural Language
   Supervision.** Proceedings of ICML, 2021.
