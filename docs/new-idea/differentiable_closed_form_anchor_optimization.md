# Differentiable Closed-Form Anchor Optimization for SPEED

## Status

This document proposes a research direction and an implementation plan. The
method has not yet been implemented or validated. Its purpose is to test
whether an anchor optimized through the actual SPEED edit can produce a better
erasure-preservation trade-off than a manually selected anchor or an anchor
learned independently of the edit.

## Summary

SPEED currently computes a closed-form cross-attention weight update from a
target embedding and an anchor embedding. Existing anchor choices are fixed in
advance, while the current learned-anchor method optimizes an anchor against a
frozen, unedited diffusion model and only applies SPEED afterward. The anchor
learner therefore has no direct information about the edit that its anchor
will produce.

This proposal turns the closed-form SPEED update into a differentiable
function of a continuous anchor. At each optimization step, the method:

1. constructs the effective edited weights analytically from the current
   anchor;
2. runs the U-Net with those effective weights;
3. measures erasure and preservation directly in predicted-noise space; and
4. backpropagates through the U-Net and the analytical edit into the anchor.

The base diffusion model, text encoder, and VAE remain frozen. Only the anchor
parameters are optimized. The final artifact is a continuous anchor or anchor
residual together with the ordinary closed-form SPEED checkpoint derived from
it.

The central hypothesis is:

> An anchor selected by the downstream behavior of the edited model will find
> a better feasible trade-off between target erasure and non-target
> preservation than an anchor selected by a proxy objective before editing.

## Motivation

Let an anchor be denoted by \(a\), a target text representation by \(c\), and
the original U-Net weights by \(W\). In the current two-stage learned-anchor
pipeline, anchor optimization and model editing are separate:

\[
a^\star = \arg\min_a \mathcal L_{\mathrm{proxy}}(W,a),
\qquad
W' = \operatorname{SPEED}(W,c,a^\star).
\]

This separation creates an objective mismatch. A low proxy loss under the
original model does not imply that the resulting edited model strongly erases
the target, and it does not measure damage to retained concepts. The mismatch
is especially important because SPEED consumes only a particular contextual
text representation and transforms it through layer-specific preservation
geometry.

The proposed method optimizes the anchor through the edit that it induces:

\[
a^\star = \arg\min_a
\mathcal L_{\mathrm{behavior}}
\left(\operatorname{SPEED}(W,c,a)\right).
\]

Here the anchor is an internal parameterization of the edit. The optimization
objective is defined by the desired behavior of the edited model rather than
by proximity to a predetermined anchor embedding.

## Scope and non-goals

The first prototype targets Stable Diffusion v1.4 and the existing SPEED path
for editing cross-attention `attn2.to_v` weights. It should support one target
before extending to multiple targets.

The first prototype does not attempt to differentiate through hard retain-set
filtering, threshold-based rank selection, top-k residual selection, or
discrete prompt sampling. These operations introduce discontinuities that
would obscure whether differentiating through the analytical edit is useful.
Their outputs should be calculated once and treated as frozen edit statistics
during anchor optimization.

The proposal does not claim that a perfectly erasing and perfectly preserving
anchor must exist. If the target direction overlaps the subspace that must be
preserved, the selected edit family may have no exact solution. The experiment
instead searches for the best feasible trade-off under an explicit erasure
requirement.

## Notation

Let:

- \(W_\ell\in\mathbb R^{o_\ell\times d}\) be the frozen original weight of
  edited layer \(\ell\);
- \(c_i\in\mathbb R^{1\times d}\) be target embedding \(i\);
- \(a_i\in\mathbb R^{1\times d}\) be its learnable continuous anchor;
- \(r_i=a_i-c_i\) be the learnable target-anchor residual;
- \(C\) be the target covariance used by SPEED;
- \(P_\ell\) be the frozen retain-low projector;
- \(K_2\) be the existing invariant-constraint basis;
- \(M_\ell=(CP_\ell+\gamma I)^{-1}\), where \(\gamma\) is the existing
  retain scale; and
- \(\epsilon_W(x_t,t,h)\) be the noise predicted by a U-Net with weights
  \(W\), noisy latent \(x_t\), timestep \(t\), and text hidden states \(h\).

For multiple targets, the current legacy target-anchor interaction statistic
is:

\[
D(a)=\frac{1}{N}\sum_{i=1}^{N}r_i^\top c_i.
\]

Define the frozen right-hand SPEED operator for layer \(\ell\) as:

\[
B_\ell =
P_\ell
\left[
I-M_\ell K_2
\left(
K_2^\top P_\ell M_\ell K_2+\lambda I
\right)^{-1}
K_2^\top P_\ell
\right]
M_\ell.
\]

The effective edited weight is then:

\[
W'_\ell(a)=W_\ell+W_\ell D(a)B_\ell.
\]

With \(C\), \(P_\ell\), \(K_2\), and \(B_\ell\) frozen, the effective weight
is differentiable with respect to every \(a_i\). PyTorch can therefore
backpropagate through the U-Net prediction into the anchor parameters.

## Low-rank form of the differentiable edit

The implementation should avoid materializing \(D(a)\in\mathbb R^{d\times d}\)
for every optimization step. By associativity:

\[
W_\ell D(a)B_\ell
=
\frac{1}{N}\sum_{i=1}^{N}
\left[W_\ell r_i^\top\right]
\left[c_iB_\ell\right].
\]

Each target contributes a rank-one update. The target-dependent row
\(c_iB_\ell\) and all frozen edit statistics can be cached. Each iteration
only needs to compute \(W_\ell r_i^\top\), form the outer products, and run the
U-Net. This form reduces memory use, shortens the autograd graph, and makes the
relationship between the anchor and each layer update explicit.

## Anchor parameterization

The first implementation should optimize the residual \(r_i\) directly because
SPEED uses the anchor only through \(a_i-c_i\). A bounded parameterization
prevents an optimizer from satisfying the erasure loss by increasing the edit
without limit:

\[
r_i = \rho_i\frac{u_i}{\lVert u_i\rVert_2+\varepsilon},
\qquad
\rho_i=\rho_{\max}\,\sigma(s_i).
\]

Here \(u_i\) and \(s_i\) are learnable. The direction and magnitude can be
initialized from a legacy text anchor, the empty anchor, a TGPRS residual, or a
small random residual. Initialization should be an experimental variable.

This direct contextual-space parameterization gives the editor maximum
freedom, but the resulting anchor may not correspond to any discrete text
prompt. That is acceptable if the anchor is treated as an internal edit
parameter. A later variant can optimize a continuous input token embedding
through the frozen CLIP text transformer when a prompt-like, transferable
anchor is required.

## Predicted-noise objective

### Why ordinary diffusion loss is insufficient

Minimizing

\[
\left\|
\epsilon_{W'(a)}(x_t,t,h_t)-\epsilon
\right\|_2^2
\]

on target images is ordinary diffusion training. It teaches the edited model
to reconstruct the target and therefore conflicts with erasure. Maximizing the
same loss is also unsuitable: it can reward arbitrary model damage and
unbounded anchor norms.

The loss needs a target-independent description of the desired post-edit
behavior.

### Counterfactual target construction

For each target prompt \(p_t\), construct a context-matched counterfactual
prompt \(p_{\neg t}\) in which only the erased identity or style is removed.
Examples include:

| Target prompt | Counterfactual prompt |
|---|---|
| `Snoopy riding a bicycle` | `a cartoon dog riding a bicycle` |
| `a village in Van Gogh style` | `a village painting` |
| `a portrait of Barack Obama in an office` | `a portrait of a person in an office` |

The counterfactual prompt is a behavioral reference, not the anchor embedding
used by the closed-form editor. It defines what should remain in the generated
scene while leaving the optimizer free to discover the internal anchor that
best realizes that behavior.

For a shared noisy latent and timestep, calculate the frozen teacher target:

\[
\epsilon_{\mathrm{cf}}
=
\operatorname{sg}
\left[
\epsilon_W(x_t,t,h_{\neg t})
\right],
\]

where \(\operatorname{sg}\) denotes stop-gradient. The initial erasure loss is:

\[
\mathcal L_{\mathrm{erase}}
=
\mathbb E_{x_t,t}
\left[
w(t)
\left\|
\epsilon_{W'(a)}(x_t,t,h_t)
-\epsilon_{\mathrm{cf}}
\right\|_2^2
\right].
\]

The timestep weight \(w(t)\) should prevent a narrow range of noise scales
from dominating the loss. The experiment should report the chosen timestep
distribution and weighting rule.

### Less rigid target-direction cancellation

Full counterfactual matching may impose more behavior than necessary. A softer
variant isolates the target-conditioned direction of the original model:

\[
g_t=
\epsilon_W(x_t,t,h_t)
-\epsilon_W(x_t,t,h_{\neg t}).
\]

Let

\[
q_t=
\epsilon_{W'(a)}(x_t,t,h_t)
-\epsilon_W(x_t,t,h_{\neg t}).
\]

Then penalize only the component of the edited residual that remains aligned
with the original target direction:

\[
\mathcal L_{\mathrm{direction}}
=
\mathbb E
\left[
\left\|
\operatorname{Proj}_{g_t}(q_t)
\right\|_2^2
\right].
\]

This objective gives the optimizer freedom in directions unrelated to the
identified target contribution. It should be tested after the full-MSE
objective, since the additional freedom may also permit visual artifacts.

## Preservation objective

For each retained prompt \(p_r\), use the original model as a teacher on the
same noisy latent and timestep:

\[
\mathcal L_{\mathrm{retain}}
=
\mathbb E_{p_r,x_t,t}
\left[
w(t)
\left\|
\epsilon_{W'(a)}(x_t,t,h_r)
-
\operatorname{sg}
\left[
\epsilon_W(x_t,t,h_r)
\right]
\right\|_2^2
\right].
\]

The retain set should include both broad prompts and semantically close
neighbors. Random COCO prompts alone may miss the concepts most likely to be
damaged by a target edit.

An edit-size regularizer supplies an additional trust region:

\[
\mathcal L_{\mathrm{edit}}
=
\frac{1}{L}\sum_{\ell=1}^{L}
\frac{\lVert\Delta W_\ell(a)\rVert_F^2}
     {\lVert W_\ell\rVert_F^2+\varepsilon}.
\]

The initial scalarized objective is:

\[
\mathcal L
=
\mathcal L_{\mathrm{erase}}
+\alpha\mathcal L_{\mathrm{retain}}
+\beta\mathcal L_{\mathrm{edit}}.
\]

Scalar weights make the first prototype simple, but comparisons must be made
at matched erasure strength. A preservation gain obtained only by weakening
erasure is not evidence of a better anchor.

## Constrained formulation

The preferred later formulation treats erasure as a requirement and
preservation as the objective:

\[
\min_a
\quad
\mathcal L_{\mathrm{retain}}(a)
+\beta\mathcal L_{\mathrm{edit}}(a)
\quad
\text{subject to}
\quad
\mathcal L_{\mathrm{erase}}(a)\le\tau.
\]

This reflects the practical goal: once the target is erased sufficiently,
additional target destruction has no value and the remaining optimization
capacity should preserve other behavior. The constraint can be implemented
with an augmented Lagrangian or a dual variable rather than a fixed penalty
weight.

## Degenerate solutions and safeguards

### Identity anchor

If \(a=c\), then \(r=0\), \(D=0\), and no edit occurs. This becomes a trivial
solution if the teacher target is allowed to depend on the learnable anchor.
The teacher must therefore be computed from the frozen original model and an
anchor-independent behavioral target.

### Unbounded residual

Without a bounded residual or edit penalty, optimization can increase the
anchor norm to satisfy an erasure metric by damaging the model. Use the bounded
parameterization, log residual and edit norms, and reject non-finite updates.

### Off-manifold anchor

A freely optimized contextual vector may not be reachable from the frozen text
encoder. This is acceptable for an internal edit parameter but must be stated
clearly. Compare direct contextual optimization with input-token optimization
before making claims about semantic or cross-model transferability.

### Preservation by weak erasure

Low retain loss is trivial when the edit is nearly zero. Report preservation
only at a fixed erasure threshold or as a Pareto frontier across erasure
strengths.

### Hard, anchor-dependent preprocessing

The existing influence filter can select retain examples using an
anchor-dependent preliminary erase weight. Boolean filtering, thresholded SVD
rank, top-k selection, and medoid selection do not provide a useful smooth
gradient. Freeze these choices in the first experiment. Later work can replace
them with temperature-controlled soft weights if joint optimization is useful.

## Differentiable implementation seam

The implementation should introduce a deep module in `src/` that separates
frozen edit-statistic construction from differentiable effective-weight
construction. Its caller-facing interface should remain small:

```python
edit_state = prepare_differentiable_speed_edit(
    base_unet,
    target_embeddings,
    retain_embeddings,
    config,
)

edited_parameters, diagnostics = edit_state.effective_parameters(
    anchor_residuals,
)
```

`prepare_differentiable_speed_edit` should calculate and cache all
anchor-independent quantities, including target covariance, retain projectors,
invariant terms, right-hand operators, target rows, and edited parameter names.
`effective_parameters` should construct graph-connected low-rank updates from
the current residuals without mutating the base U-Net.

The U-Net can initially be evaluated with `torch.func.functional_call` and an
overridden parameter mapping. If this is incompatible with the diffusers U-Net
or consumes too much memory, the alternative is a narrow adapter at the
cross-attention linear seam that evaluates:

```python
output = torch.nn.functional.linear(hidden_states, effective_weight, bias)
```

Direct assignment through `state_dict()`, `load_state_dict()`, `.data`, or a
`torch.no_grad()` edit path must not be used during optimization because those
approaches disconnect the loss from the anchor.

The existing checkpoint-producing `edit_model` path should remain unchanged
for reproducibility. After optimization, detach the best anchor and call the
ordinary SPEED checkpoint path once to create the final artifact.

## Optimization loop

The proposed training loop is:

```text
Precompute frozen SPEED edit statistics
Build target/counterfactual and retain latent banks with the original model
Initialize bounded anchor residual parameters

For each optimization step:
    Construct graph-connected effective cross-attention weights
    Sample target and retain items, timesteps, and noises
    Compute teacher noise predictions without gradients
    Compute edited-model noise predictions with effective weights
    Compute erasure, retention, and edit-size losses
    Backpropagate only into the anchor residual parameters
    Update the optimizer and validate on held-out samples

Restore the best validation anchor
Produce and save a normal detached SPEED checkpoint
```

Teacher predictions can be cached when the latent, noise, timestep, and prompt
banks are fixed. Student examples can be concatenated into one U-Net batch when
their spatial dimensions match. Gradient checkpointing and mixed precision may
be needed for the student U-Net even though all base weights are frozen,
because backpropagation still retains intermediate activations needed to
differentiate the loss with respect to the effective weights.

Matrix decompositions and cached SPEED operators should be calculated in
`float32`. U-Net execution can use `float16` or `bfloat16`, subject to numerical
validation.

## Data construction

The optimization data should contain three distinct groups:

1. **Target-context pairs.** Diverse prompts containing the target, paired
   with counterfactual prompts that retain composition, actions, objects, and
   background while removing only the target identity or style.
2. **Close-neighbor retention prompts.** Semantically related people, objects,
   styles, or subclasses that are most likely to share the target direction.
3. **Broad retention prompts.** General prompts sampled from the existing
   retain benchmark to detect global quality degradation.

Training and validation must use disjoint prompt templates, random seeds,
latents, noises, and preferably timestep samples. A method that optimizes only
one target image or one denoising state is likely to overfit the local score
field.

## Initial experiment

Start with one target for which the current evaluation stack is already
available, such as `Snoopy` for an instance or `Van Gogh` for a style. Use only
`attn2.to_v`, `anchor_mode=legacy`, `aug_num=0`, and frozen retain projectors.

Compare:

1. legacy SPEED with the existing fixed anchor;
2. SPEED with the current independently learned CLIP-guided anchor; and
3. SPEED with the proposed end-to-end differentiable anchor.

Use the same target prompts, retain prompts, seeds, generation scheduler, and
edit hyperparameters for all methods. Tune or sweep each method to obtain
checkpoints at matched target-erasure levels.

The first ablations should cover:

- full counterfactual noise MSE versus target-direction cancellation;
- direct contextual residual versus a residual constrained to the existing
  residual subspace;
- fixed residual norm versus learned bounded magnitude;
- close-neighbor retain prompts versus broad retain prompts only; and
- fixed penalty weights versus the constrained objective.

## Evaluation

Optimization metrics should include:

- training and held-out erasure noise loss;
- training and held-out retain noise loss;
- anchor residual norm and effective edit norm;
- gradient norm and finite-gradient status;
- per-layer update norm; and
- wall-clock time and peak GPU memory.

Final checkpoints should be evaluated through complete generation rather than
only one-step noise losses. Report:

- target CLIP score and task-specific target detector accuracy;
- adversarial or paraphrased target-prompt robustness;
- close-neighbor concept accuracy or CLIP score;
- broad MS-COCO CLIP score and FID;
- qualitative preservation of composition and background in paired prompts;
  and
- the erasure-preservation Pareto frontier.

Noise loss is a training surrogate. It should not be treated as proof that the
concept has been removed from the generative model.

## Tests

Add focused CPU `unittest` coverage before GPU evaluation:

- compare the low-rank effective update with the current dense SPEED formula;
- use `torch.autograd.gradcheck` in double precision on a small synthetic edit;
- verify that the anchor receives finite, non-zero gradients;
- verify that base U-Net and text-encoder parameters receive no gradients;
- verify that `anchor == target` produces the zero legacy update;
- verify residual magnitude bounds;
- verify deterministic results with fixed seeds;
- verify clear failure on singular, shape-mismatched, or non-finite inputs; and
- verify that the final detached checkpoint matches the effective weights used
  for validation within the configured numerical tolerance.

Run the focused tests first, followed by:

```bash
python -m unittest discover -s tests
```

## Success criteria

The hypothesis is supported if the differentiable anchor:

- reaches the same or stronger target erasure as the baselines;
- improves close-neighbor and broad retention at matched erasure strength;
- generalizes to held-out prompts, seeds, latents, and timesteps;
- produces bounded, numerically stable residuals and layer updates; and
- remains computationally practical for Stable Diffusion v1.4.

A reduction in predicted-noise loss alone is insufficient. The method must
improve the final generation-level erasure-preservation frontier.

## Recommended implementation phases

### Phase 1: Synthetic differentiability test

Implement the cached operator and low-rank update for small synthetic matrices.
Verify numerical equivalence with the current dense equation and confirm the
gradient with finite differences or `gradcheck`.

### Phase 2: One-layer U-Net probe

Apply the differentiable update to one `attn2.to_v` layer and optimize a single
bounded residual using a small fixed latent bank. Confirm that the loss changes
and gradients reach only the residual.

### Phase 3: Full value-layer optimization

Enable all selected value-projection layers, add target and retain minibatches,
and select the best anchor using a held-out predicted-noise objective.

### Phase 4: Generation evaluation

Detach the optimized anchor, produce the ordinary SPEED checkpoint, and run the
existing sampling and evaluation workflows. Compare methods at matched erasure
strength.

### Phase 5: Additional flexibility

Only after the basic hypothesis is supported, test soft influence filtering,
subspace-constrained residuals, layer-specific residuals, multi-target anchors,
and continuous input-token parameterization.

## Main risks

The main scientific risk is surrogate mismatch: a one-step predicted-noise
objective may improve locally while failing to improve complete denoising
trajectories. Held-out full-generation evaluation is therefore mandatory.

The main optimization risk is that a free contextual residual exploits
directions that produce artifacts instead of semantic erasure. Residual bounds,
edit regularization, close-neighbor preservation, and generation-level model
selection mitigate this risk.

The main engineering risk is memory use during U-Net backpropagation through
many effective weight updates. The low-rank formulation, frozen base weights,
small minibatches, cached teacher predictions, mixed precision, and gradient
checkpointing should make the first SD v1.4 prototype feasible.
