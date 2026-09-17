# ReSAM: Retain-Guided Sparse Anchor Mixture

## Status

Research idea. ReSAM has not been implemented or evaluated.

## Motivation

The current `train_closed_form_backprop.py` learns a continuous anchor direction
and magnitude by minimizing similarity between the frozen model's and edited
model's predicted noise on target prompts. Even with a norm bound, the anchor
can move freely in contextual embedding space. It may become an adversarial
embedding that satisfies this objective without behaving like a meaningful
concept anchor.

ReSAM limits the anchor to a sparse weighted mixture of fixed vocabulary
concept embeddings. The candidate concepts should be chosen because their
individual anchors already erase the target effectively under the closed-form
SPEED edit. Optimization then focuses on preserving retain concepts.

## Proposed method

For a target embedding \(c\), prepare a frozen candidate bank
\(E = \{e_1, \ldots, e_V\}\) using the same contextual embedding extraction
as the current editor. Learn one score \(s_j\) per candidate, but use only the
\(k\) highest-scoring candidates in the forward pass. Normalize their weights
to sum to one:

\[
I_k = \operatorname{TopK}(s, k), \qquad
w_j = \frac{\exp(s_j / \tau)}{\sum_{i \in I_k}\exp(s_i / \tau)}
\quad (j \in I_k), \qquad
a(s) = \sum_{j \in I_k} w_j e_j.
\]

All other weights are zero. The temperature \(\tau\) controls how evenly the
selected candidates contribute. Use \(a(s)\) as the anchor in the existing
differentiable closed-form SPEED update, whose target residual is \(a(s)-c\).
Keep the teacher U-Net, text encoder, candidate embeddings, and closed-form
geometry frozen. Optimize only the candidate scores.

For retain prompts, run the teacher and edited U-Nets with the same noisy
latent, timestep, and text conditioning. Minimize their predicted-noise L2
error:

\[
\mathcal L_{\mathrm{retain}}(s) =
\mathbb E_{(x_t,t,h_r)}
\left[\left\|\epsilon_{W'(a(s))}(x_t,t,h_r)
- \epsilon_W(x_t,t,h_r)\right\|_2^2\right].
\]

The premise is that candidate selection supplies erasure, while this loss
finds a mixture that preserves retain behavior. Target erasure must still be
measured throughout training: a mixture could approach the target embedding,
produce a near-zero edit, and obtain a low retain loss without erasing.

## Gradients through top-k

With ordinary hard top-k selection, only selected scores receive direct
gradients. A straight-through estimator can use the sparse mixture in the
forward pass and a dense softmax surrogate in the backward pass:

```python
soft = torch.softmax(scores / temperature, dim=0)
indices = scores.topk(k).indices
hard = torch.zeros_like(soft)
hard[indices] = torch.softmax(scores[indices] / temperature, dim=0)
weights = hard.detach() - soft.detach() + soft
anchor = weights @ candidate_embeddings
```

This gives every score a gradient, but the backward gradient is an
approximation to the discontinuous top-k operation. Monitor whether the
selected set changes and whether training remains stable.

## Capacity and evaluation

Keeping \(k\) small limits the anchor's accessible directions and may reduce
its ability to exploit the retain objective. A larger \(k\) gives the
optimizer more flexibility and may allow mixtures that cancel the edit. In
the existing SPEED equation, however, one synthesized anchor still contributes
one target residual and a rank-one target-anchor statistic. Increasing \(k\)
expands the feasible anchor set; it does not directly increase the edit's
matrix rank for a single target.

Compare several values of \(k\) against individual fixed candidates and the
current continuous-anchor method. Report retain predicted-noise L2, target
erasure, generated-image results, selected candidates and weights, anchor
distance from the target, and edit magnitude. A retain-only optimum is useful
only if target erasure stays above an explicit acceptance threshold.
