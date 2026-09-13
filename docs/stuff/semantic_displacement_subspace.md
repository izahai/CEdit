# Using a Semantic Displacement Subspace for Concept Erasure

## Motivation

The current target-global pairwise construction forms a low-rank subspace from
pairwise embedding residuals and obtains a target-specific direction by
projecting the negative target embedding into that subspace. If the target
embedding is $t_i$, the current direction is approximately

$$
    r_i \propto P_{\mathcal S}(-t_i),
$$

where $P_{\mathcal S}$ is the orthogonal projector onto the retained
subspace $\mathcal S$.

This operation treats the embedding origin as a meaningful representation of
"no concept." That assumption is difficult to justify. In particular, the
pairwise differences used to construct $\mathcal S$ are invariant to a
translation of the embedding space,

$$
    (z_j+c)-(z_i+c)=z_j-z_i,
$$

but the projected negative target is not:

$$
    P_{\mathcal S}(-(t_i+c))
    =P_{\mathcal S}(-t_i)-P_{\mathcal S}(c).
$$

It is therefore more natural to interpret $\mathcal S$ as a **semantic
displacement subspace**: a low-dimensional space of meaningful movements
between concepts. The erasure problem then becomes one of selecting an
appropriate displacement inside this space, rather than simply projecting the
negative target.

## Semantic displacement subspace

Let $z_1,\ldots,z_m$ contain the target concepts and any additional neutral
anchors used to define meaningful semantic variation. Construct pairwise
displacements

$$
    d_{pq}=z_q-z_p.
$$

After normalization, stack the displacements into a matrix

$$
    D=
    \begin{bmatrix}
        \hat d_{12}^{\top} \\
        \hat d_{13}^{\top} \\
        \vdots
    \end{bmatrix},
    \qquad
    \hat d_{pq}=\frac{d_{pq}}{\lVert d_{pq}\rVert_2}.
$$

Let $B\in\mathbb R^{k\times d}$ contain the top $k$ right singular
vectors of $D$. The semantic displacement subspace and its projector are

$$
    \mathcal S=\operatorname{span}(B^{\top}),
    \qquad
    P_{\mathcal S}=B^{\top}B,
$$

assuming that the rows of $B$ are orthonormal. Every permitted displacement
can then be parameterized as

$$
    r_i=B^{\top}c_i.
$$

This parameterization guarantees that the displacement remains in
$\mathcal S$, while leaving its direction and magnitude available for a
target-specific objective.

## 1. Anchor-directed displacement

The simplest alternative is to move a target toward a meaningful neutral
anchor $a_i$:

$$
    r_i=P_{\mathcal S}(a_i-t_i).
$$

This answers a direct geometric question: which displacement available inside
$\mathcal S$ brings the target closest to its neutral counterpart?

Possible anchors include:

- the empty prompt;
- a generic superclass, such as "a person" for a celebrity;
- an artist-neutral prompt for artistic-style erasure;
- the centroid of several neutral prompts.

Unlike $P_{\mathcal S}(-t_i)$, the expression
$P_{\mathcal S}(a_i-t_i)$ is invariant to a shared translation of the
embedding space.

## 2. Adaptive neutral-centroid displacement

A single anchor may introduce prompt-specific bias. Given neutral anchors
$a_1,\ldots,a_M$, define their centroid as

$$
    \mu_A=\frac{1}{M}\sum_{j=1}^{M}a_j
$$

and use

$$
    r_i=P_{\mathcal S}(\mu_A-t_i).
$$

A target-adaptive centroid can give greater weight to semantically relevant
anchors:

$$
    w_{ij}
    =
    \frac{
        \exp\!\left(\beta\,\cos(t_i,a_j)\right)
    }{
        \sum_{q=1}^{M}
        \exp\!\left(\beta\,\cos(t_i,a_q)\right)
    },
    \qquad
    \mu_i=\sum_{j=1}^{M}w_{ij}a_j,
$$

followed by

$$
    r_i=P_{\mathcal S}(\mu_i-t_i).
$$

The temperature parameter $\beta$ controls whether the displacement uses a
broad average of anchors or concentrates on the nearest anchors.

## 3. Target-versus-retain contrast

Instead of moving toward an explicit anchor, remove the component that makes a
target distinct from the retained distribution. Let $\mu_R$ be the mean
embedding of the retain set. A simple contrastive displacement is

$$
    r_i=-P_{\mathcal S}(t_i-\mu_R).
$$

This suppresses the target relative to retained concepts without treating the
embedding origin as neutral.

The retain covariance can also be used to discount directions that naturally
vary within the retain set. With a regularized covariance

$$
    \widetilde\Sigma_R=\Sigma_R+\gamma I,
$$

one possible direction is

$$
    r_i
    =-P_{\mathcal S}
      \widetilde\Sigma_R^{-1}(t_i-\mu_R).
$$

This is a Mahalanobis-style contrast: directions with low retain variance are
treated as more target-specific than directions with high retain variance.

## 4. Retain-safe semantic displacement space

The semantic displacement space can be refined before selecting a direction.
Let $U_R$ be an orthonormal basis for the leading retain directions and let

$$
    P_R=U_RU_R^{\top}.
$$

Remove these directions from the semantic displacement basis:

$$
    \widetilde B^{\top}=(I-P_R)B^{\top}.
$$

After orthonormalizing the nonzero columns of $\widetilde B^{\top}$, obtain a
retain-safe space $\mathcal S_{\mathrm{safe}}$. An anchor-directed residual
then becomes

$$
    r_i=P_{\mathcal S_{\mathrm{safe}}}(a_i-t_i).
$$

This construction explicitly restricts erasure to semantic movements with
limited overlap with dominant retain variation. A soft alternative is to
penalize, rather than remove, the retain component:

$$
    \min_{r_i\in\mathcal S}
    \mathcal L_{\mathrm{erase}}(r_i)
    +\lambda_R\lVert P_Rr_i\rVert_2^2.
$$

## 5. Minimal output-changing displacement

The subspace can define feasible directions while the diffusion model decides
which direction is most effective. Let $W_\ell$ be the original frozen
cross-attention value-projection weight at layer $\ell$. For target $t_i$,
find the smallest displacement in $\mathcal S$ that produces a required
output-angle change:

$$
    \begin{aligned}
    \min_{r_{i,\ell}}\quad
        & \lVert r_{i,\ell}\rVert_2^2 \\
    \text{subject to}\quad
        & r_{i,\ell}\in\mathcal S, \\
        & \cos\!\left(
            W_\ell(t_i+r_{i,\ell}),
            W_\ell t_i
          \right)\leq\tau.
    \end{aligned}
$$

Here, $\tau=\cos(\theta_{\min})$ specifies a minimum output-angle change.
Using $r_{i,\ell}=B^{\top}c_{i,\ell}$, an unconstrained penalty formulation
is

$$
    \begin{aligned}
    \mathcal L_{i,\ell}
    ={}&
    \lambda_{\mathrm{angle}}
    \operatorname{ReLU}\!\left(
        \cos\!\left(
            W_\ell(t_i+B^{\top}c_{i,\ell}),
            W_\ell t_i
        \right)-\tau
    \right)^2 \\
    &+\lambda_{\mathrm{anchor}}
      \lVert t_i+B^{\top}c_{i,\ell}-a_i\rVert_2^2 \\
    &+\lambda_{\mathrm{retain}}
      \lVert P_RB^{\top}c_{i,\ell}\rVert_2^2 \\
    &+\lambda_{\mathrm{norm}}
      \lVert B^{\top}c_{i,\ell}\rVert_2^2.
    \end{aligned}
$$

The anchor and retain terms are optional. This formulation gives each layer a
target-specific direction while keeping the search inside a shared,
interpretable semantic space.

## 6. Joint displacements for multiple targets

For $N$ erased concepts, collect the coefficient vectors into

$$
    C=
    \begin{bmatrix}
        c_1^{\top} \\
        \vdots \\
        c_N^{\top}
    \end{bmatrix}.
$$

Rather than learning each displacement independently, optimize all targets
jointly:

$$
    \min_C
    \mathcal L_{\mathrm{erase}}(C)
    +\lambda_R\mathcal L_{\mathrm{retain}}(C)
    +\lambda_F\lVert C\rVert_F^2
    +\lambda_*\lVert C\rVert_*.
$$

The Frobenius penalty controls displacement magnitude. The nuclear-norm
penalty $\lVert C\rVert_*$ encourages related concepts to reuse a small
number of coefficient patterns. This may make large-scale multi-concept
erasure more stable and parameter-efficient.

## Recommended experimental progression

The following progression isolates the benefit of each added assumption:

1. **Anchor projection**

$$
       r_i=P_{\mathcal S}(a_i-t_i).
$$

2. **Adaptive anchor projection**

$$
       r_i=P_{\mathcal S}(\mu_i-t_i).
$$

3. **Retain-safe anchor projection**

$$
       r_i=P_{\mathcal S_{\mathrm{safe}}}(a_i-t_i).
$$

4. **Layer-aware optimized displacement**

   Find a minimum-norm $r_{i,\ell}\in\mathcal S_{\mathrm{safe}}$ that
   satisfies an output-change margin.

Each variant should be compared with the existing negative-target projection
using identical residual rank, threshold, retain set, and evaluation prompts.
Useful diagnostics include:

- erasure and retain accuracy;
- CLIP score and FID for target, non-target, and MS-COCO prompts;
- displacement norm in embedding and output space;
- achieved output angle by layer and target;
- cosine similarity between the displacement and anchor direction;
- overlap with the retain subspace;
- convergence rate and optimization cost.

## Preferred interpretation

The central claim can be expressed without making "residual subspace" the
public-facing method name:

> We construct a low-rank semantic displacement space from pairwise concept
> differences, then search within it for a small, retain-safe displacement that
> suppresses each target.

In implementation and derivations, **pairwise residual subspace** remains a
precise description of how the basis is built. In the paper narrative,
**semantic displacement space** better describes what the basis represents and
how it can be used.

# New idea
I have new idea to finding the suitable residual vector r, suppose we have 
erase_norm

we have learnable_anchor = t + erase_norm * P_erase * learnable unit vector

from the orginal list of target clip tokens

we replace last token before eos with our learnable_anchor then use it to create an image
then at the last forward step when the image completes we use the final image to compute clip score 
with our current clip embedding tokens, we want to maximize this clip score loss by gradient only on 
the last step of image generation.

So we use this loss to learn the learnable unit vector in Semantic Displacement Subspace to improve 
the probability density of the learnable_anchor

P_erase is Semantic Displacement Subspace Projection

If you optimize

$$
\max_{v_i};
\operatorname{CLIP}
\left(
G(t_i+\rho v_i),
\text{original target}
\right),
$$


$$
\min_{v_i};
-\operatorname{sim}
\left(
f_I(G(t_i+\rho v_i)),
e_i^{\mathrm{neutral}}
\right)
+
\lambda_T
\operatorname{sim}
\left(
f_I(G(t_i+\rho v_i)),
e_i^{\mathrm{target}}
\right),
$$

subject to

$$
v_i\in\mathcal S,
\qquad
\lVert v_i\rVert_2=1.
$$


