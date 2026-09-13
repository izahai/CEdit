# Differentiable Closed-Form Anchor Optimization for SPEED

## Status

This document proposes a research direction and an implementation plan. The
method has not yet been implemented or validated. Its purpose is to test
whether an anchor optimized through the actual SPEED edit can produce a better
erasure-preservation trade-off than a manually selected anchor or an anchor
learned independently of the edit.

The former vendored reference tree has been removed after the required
mechanics were ported. The predicted-noise training loop, parameter selection,
and checkpoint conventions now live in this repository's `src/` and trainer
modules; the new SPEED implementation is self-contained.

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
3. minimizes cosine similarity between frozen and edited predicted noise under
   identical target-prompt conditioning; and
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
for editing cross-attention `attn2.to_v` weights. It must preserve the complete
legacy SPEED equation, including the retain projector, `K2` invariant terms,
regularization, and dense target-anchor statistic. The only intended
algorithmic change is to make the anchor learnable through that equation. It
should support one target before extending to multiple targets.

The prototype uses the local SD v1.4 training path and does not generalize
SPEED beyond that model family. Other parameter-training modes are out of
scope; the first experiment remains equivalent to the strict value-projection
subset, so only `attn2.to_v.weight` is overridden.

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

## Local implementation boundaries

The required mechanics are implemented locally behind four small seams:

| Local code | Role | Constraint |
|---|---|---|
| `train_closed_form_backprop.py` | CLI precedence, validation, and SD orchestration | Single-target SD v1.4 prototype |
| `src/closed_form_anchor_training.py` | Frozen denoising-state preparation, cosine objective, and anchor-only optimization | No ESD negative-guidance objective |
| `src/differentiable_legacy_edit.py` | Exact dense SPEED equation, projector geometry, and graph-connected overrides | Override only `attn2.to_v.weight` |
| `src/edit_checkpoint.py` | SPEED-specific partial checkpoint save/load and validation | Do not label artifacts as ESD checkpoints |

The optimized object is the continuous anchor; effective weights are non-leaf
tensors created from it. Use a stateless functional call for the edited
forward, and keep the base U-Net structurally unchanged for teacher inference
and checkpoint reproducibility.

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

## Optional low-rank equivalence

The first implementation should retain the current dense legacy equation and
matrix multiplication order so that its forward output can be compared
directly with `edit_model`. After that equivalence is established, the same
equation can optionally be evaluated without materializing
\(D(a)\in\mathbb R^{d\times d}\). By associativity:

\[
W_\ell D(a)B_\ell
=
\frac{1}{N}\sum_{i=1}^{N}
\left[W_\ell r_i^\top\right]
\left[c_iB_\ell\right].
\]

Each target contributes a rank-one update. The target-dependent row
\(c_iB_\ell\) and all frozen edit statistics can be cached. This equivalent
form may later reduce memory use and shorten the autograd graph, but it is an
optimization rather than part of the minimal-change prototype.

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

## Target-conditioned cosine erasure objective

For every training example, run the frozen and edited models with exactly the
same noisy latent \(x_t\), timestep \(t\), and target-prompt hidden states
\(h_c\):

\[
\epsilon_{\mathrm{base}}
=
\operatorname{sg}\left[\epsilon_W(x_t,t,h_c)\right],
\qquad
\epsilon_{\mathrm{edit}}
=
\epsilon_{W'(a)}(x_t,t,h_c).
\]

Here \(\operatorname{sg}\) denotes stop-gradient. The base prediction is a
fixed reference; the edited prediction remains connected through the
effective SPEED weights to the anchor parameters.

Flatten the non-batch dimensions of each prediction and calculate cosine
similarity per example:

\[
s_i
=
\frac{
\left\langle
\operatorname{vec}(\epsilon_{\mathrm{edit},i}),
\operatorname{vec}(\epsilon_{\mathrm{base},i})
\right\rangle
}{
\max\!\left(
\left\lVert\operatorname{vec}(\epsilon_{\mathrm{edit},i})\right\rVert_2,
\varepsilon
\right)
\max\!\left(
\left\lVert\operatorname{vec}(\epsilon_{\mathrm{base},i})\right\rVert_2,
\varepsilon
\right)
}.
\]

The erasure loss is the mean cosine similarity:

\[
\mathcal L_{\mathrm{erase}}
=
\frac{1}{B}\sum_{i=1}^{B}s_i.
\]

Minimize this loss. An unchanged edit produces similarity near \(1\),
orthogonal predicted-noise directions produce similarity near \(0\), and
opposite directions approach \(-1\). Cosine similarity is used instead of a
raw difference so that variation in predicted-noise magnitude across
timesteps does not dominate optimization.

A direct implementation is:

```python
with torch.no_grad():
    base_prediction = base_unet(
        x_t,
        timestep,
        encoder_hidden_states=target_hidden_states,
        return_dict=False,
    )[0]

edited_prediction = functional_call(
    base_unet,
    effective_parameter_overrides,
    args=(x_t, timestep),
    kwargs={
        "encoder_hidden_states": target_hidden_states,
        "return_dict": False,
    },
)[0]

loss_per_example = torch.nn.functional.cosine_similarity(
    edited_prediction.float().flatten(1),
    base_prediction.float().flatten(1),
    dim=1,
    eps=cosine_eps,
)
erase_loss = loss_per_example.mean()
```

Compute the cosine in `float32`, even when the U-Net runs in a lower-precision
dtype. Do not detach `edited_prediction`, and do not use a batch-flattened
single cosine because that lets high-energy examples dominate the batch. The
base and edited calls must share the identical tensor values for \(x_t\),
\(t\), and \(h_c\); otherwise the loss can measure input differences rather
than the effect of the edit.

## First-experiment loss scope

The first experiment uses exactly one optimization loss:

\[
\mathcal L = \mathcal L_{\mathrm{erase}}.
\]

Do not add a retain loss, edit-size penalty, norm penalty, or constrained-loss
term to this initial run. The purpose of the experiment is to isolate whether
minimizing target-conditioned predicted-noise cosine can learn a useful anchor
through the closed-form edit. Retention and edit norms should still be measured
as diagnostics and evaluated after training, but they must not contribute
gradients in this experiment.

## Future multi-loss experiments

After the cosine-only experiment has been implemented and evaluated, test a
preservation objective. For each retained prompt \(p_r\), the original model
can act as a reference on the same noisy latent and timestep:

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

An edit-size regularizer can supply an additional trust region:

\[
\mathcal L_{\mathrm{edit}}
=
\frac{1}{L}\sum_{\ell=1}^{L}
\frac{\lVert\Delta W_\ell(a)\rVert_F^2}
     {\lVert W_\ell\rVert_F^2+\varepsilon}.
\]

A later scalarized objective can be:

\[
\mathcal L
=
\mathcal L_{\mathrm{erase}}
+\alpha\mathcal L_{\mathrm{retain}}
+\beta\mathcal L_{\mathrm{edit}}.
\]

These terms are not part of the first experiment. When they are introduced,
comparisons must be made at matched erasure strength. A preservation gain
obtained only by weakening erasure is not evidence of a better anchor.

## Future constrained formulation

Another later formulation treats erasure as a requirement and
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
solution if any objective reference is allowed to move with the learnable
anchor. Reference tensors and target construction must therefore be
anchor-independent. In this objective, the frozen target-conditioned base
prediction is the anchor-independent reference.

### Unbounded residual

Without a bounded residual or edit penalty, optimization can increase the
anchor norm to satisfy an erasure metric by damaging the model. Use the bounded
parameterization, log residual and edit norms, and reject non-finite updates.

### Prediction-norm collapse

Cosine similarity removes scale from the objective, but becomes poorly
conditioned if the edited prediction norm approaches zero. Compute it with an
explicit epsilon, reject non-finite losses or gradients, and log the edited to
base prediction-norm ratio per timestep. In the first experiment these checks
are diagnostics only; edit-size and preservation terms are reserved for later
experiments.

### Off-manifold anchor

A freely optimized contextual vector may not be reachable from the frozen text
encoder. This is acceptable for an internal edit parameter but must be stated
clearly. Compare direct contextual optimization with input-token optimization
before making claims about semantic or cross-model transferability.

### Preservation by weak erasure

In later multi-loss experiments, low retain loss is trivial when the edit is
nearly zero. Report preservation only at a fixed erasure threshold or as a
Pareto frontier across erasure strengths.

### Hard, anchor-dependent preprocessing

The existing influence filter can select retain examples using an
anchor-dependent preliminary erase weight. Boolean filtering, thresholded SVD
rank, top-k selection, and medoid selection do not provide a useful smooth
gradient. Freeze these choices in the first experiment. Later work can replace
them with temperature-controlled soft weights if joint optimization is useful.

For the minimal gradient and checkpoint-equivalence milestone, set
`aug_num=0` explicitly. This is an experiment configuration, not an unnoticed
change to the legacy baseline. A later `aug_num=10` experiment should freeze
the initialization projector and materialize the final checkpoint from that
same frozen state; recomputing retain filtering from the optimized anchor can
produce different weights from those used during validation.

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
`effective_parameters` should first construct the graph-connected dense legacy
statistic and apply the current closed-form expression in the same order as
`edit_model`, without mutating the base U-Net. A low-rank adapter can replace
this implementation only after numerical equivalence has been established.

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

Use the reference adapter architecture only at the orchestration boundary. A
minimal local split is:

```text
training configuration
    -> SD prompt/context preparation
    -> shared target-conditioned batch/state preparation
    -> frozen SPEED statistic preparation
    -> differentiable effective-weight builder
    -> frozen base and stateless edited U-Net calls
    -> per-example cosine-similarity loss
    -> anchor-only optimizer
```

The context layer uses fields such as `base_model_id`, `erase_concept`,
`num_inference_steps`, `guidance_scale`, `batch_size`, `resolution`, `device`,
and `torch_dtype`.
SPEED-specific fields such as `retain_path`, `retain_scale`, `threshold`,
`lamb`, `aug_num`, `params`, and projector policy remain separate and keep
their established names.

Parameter discovery should operate on `unet.named_parameters()`, select only
names containing `attn2.to_v` and ending in `.weight`, and fail if none are
found. The override mapping passed to `torch.func.functional_call` contains
only these effective weights; all other parameters and buffers come from the
frozen U-Net. Do not wrap an effective tensor in `torch.nn.Parameter`, because
that makes a new leaf and severs its construction graph.

Base-reference and edited execution must not alternate by mutating module
attributes. Run frozen reference calls normally under `torch.no_grad()`, then
run the edited model through the functional override. This also prevents an
exception between swaps from leaving the U-Net in the wrong state.

The existing checkpoint-producing `edit_model` path should remain unchanged
for reproducibility. In the initial `aug_num=0` experiment, detach the best
anchor and call the ordinary SPEED checkpoint path once to create the final
artifact, then assert that its edited weights match the functional validation
weights. Later frozen-projector modes will need a separate materializer that
consumes the exact cached edit state.

Save two artifacts:

1. an optimization artifact containing the residual parameterization,
   materialized contextual anchor, best-step metrics, projector policy, frozen
   statistic fingerprints, seeds, and all objective/SPEED hyperparameters; and
2. the ordinary edited U-Net state dictionary produced by the legacy SPEED
   path.

The local checkpoint module carries the metadata and a SPEED-specific format
identifier. Do not label the checkpoint `erasing-esd-v2`: an ESD checkpoint
stores independently trained parameters, whereas this artifact stores weights
materialized from a closed-form edit.

## Proposed CLI usage

Add a dedicated entry point rather than expanding `train_erase_null.py` with
optimization-loop concerns:

```bash
CUDA_VISIBLE_DEVICES=0 python train_closed_form_backprop.py \
  --sd_ckpt "CompVis/stable-diffusion-v1-4" \
  --target_concepts "Snoopy" \
  --anchor_concepts "" \
  --retain_path "data/instance.csv" \
  --heads "concept" \
  --params "V" \
  --anchor_mode "legacy" \
  --aug_num 0 \
  --threshold 0.1 \
  --retain_scale 1.0 \
  --residual_scale 1.0 \
  --lamb 0.0 \
  --anchor_steps 200 \
  --anchor_lr 1e-2 \
  --anchor_batch_size 1 \
  --max_residual_norm 1.0 \
  --cosine_eps 1e-8 \
  --num_inference_steps 50 \
  --guidance_scale 3.0 \
  --seed 0 \
  --save_path "logs/closed_form_backprop/snoopy" \
  --file_name "weight"
```

The learning rate above is an illustrative starting point, not a validated
default. The entry point should also accept the repository's existing
`--config` convention, with explicit CLI arguments overriding YAML values:

```bash
CUDA_VISIBLE_DEVICES=0 python train_closed_form_backprop.py \
  --config "configs/closed_form_backprop.yaml"
```

The command above omits `optimization_prompts_path`, so `Snoopy` itself is the
sole optimization prompt. When supplied, the path points to a CSV containing a
required `prompt` column. For a single-target run, every row belongs to
`target_concepts`. A later multi-target extension should require a `concept`
column and validate that each row maps to exactly one configured target.

The established SPEED arguments keep their current meanings. `retain_path` is
still required to construct SPEED's retain covariance and projector even
though retain prompts do not contribute a loss in the first experiment. The
new optimization arguments mean:

| Argument | Meaning |
|---|---|
| `--anchor_steps` | Number of anchor optimizer updates |
| `--anchor_lr` | Learning rate for anchor residual parameters only |
| `--anchor_batch_size` | Number of target-prompt diffusion states per update |
| `--max_residual_norm` | Upper bound in the bounded residual parameterization |
| `--cosine_eps` | Numerical epsilon used by per-example cosine similarity |
| `--optimization_prompts_path` | Optional target-prompt CSV used by the cosine loss |

Do not expose `--retain_weight`, `--edit_weight`, `--norm_weight`, or a generic
`--objective` selector in the first implementation. The only supported
optimization loss is the target-conditioned cosine loss. The parser should
fail early unless `params=V`, `anchor_mode=legacy`, and `aug_num=0`, so a run
cannot silently leave the validated prototype scope.

The output directory should contain two artifacts:

```text
logs/closed_form_backprop/snoopy/
    weight.safetensors
    anchor_optimization.pt
```

`weight.safetensors` contains the materialized edited U-Net weights.
`anchor_optimization.pt` contains the best anchor and residual tensors,
configuration, optimization history, diagnostic metrics, seeds, and frozen
edit-state fingerprints.

## Optimization loop

The proposed training loop is:

```text
Precompute frozen SPEED edit statistics
Encode target prompts and prepare the latent/timestep sampling policy
Initialize bounded anchor residual parameters

For each optimization step:
    Sample target prompts, x_t, and timesteps
    Predict frozen-model noise for the target prompts without gradients
    Construct graph-connected effective cross-attention weights
    Predict edited-model noise for the identical inputs
    Minimize per-example target cosine similarity
    Backpropagate only into the anchor residual parameters
    Update the optimizer and validate on held-out samples

Restore the best validation anchor
Produce and save a normal detached SPEED checkpoint
```

Base predictions can be cached when seeds, latents, timesteps, and prompts
are fixed. Edited examples can be concatenated into one U-Net batch when their
spatial dimensions match. Gradient
checkpointing and mixed precision may be needed for the student U-Net even
though all base weights are frozen, because backpropagation still retains
intermediate activations needed to differentiate the loss with respect to the
effective weights.

Matrix decompositions and cached SPEED operators should be calculated in
`float32`. U-Net execution can use `float16` or `bfloat16`, subject to numerical
validation.

## Data construction

The first optimization dataset contains only target examples:

1. **Target examples.** Diverse prompts containing the erased concept. Each
   example supplies one shared latent, timestep, and prompt embedding to both
   base and edited U-Net calls.

Retain prompts are used for evaluation but not for optimization in the first
experiment. Later multi-loss experiments can add two retention groups:

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
Make all objective and sampling settings explicit, including seeds.

Compare:

1. legacy SPEED with the existing fixed anchor;
2. SPEED with the current independently learned CLIP-guided anchor; and
3. SPEED with the proposed end-to-end differentiable anchor trained with the
   target-conditioned cosine objective.

Use the same target prompts, seeds, generation scheduler, and sampling settings
across methods wherever they apply. Keep method-specific edit hyperparameters
explicit, and tune or sweep each method to obtain checkpoints at matched
target-erasure levels.

Subsequent ablations should cover:

- partial-denoising states versus independently noised latents;
- uniform timestep sampling versus an explicitly stratified schedule;
- target cosine loss alone versus cosine loss with preservation and edit-size
  regularization;
- direct contextual residual versus a residual constrained to the existing
  residual subspace;
- fixed residual norm versus learned bounded magnitude;
- close-neighbor retain prompts versus broad retain prompts only; and
- fixed penalty weights versus the constrained objective.

## Evaluation

Optimization metrics should include:

- training and held-out target-prompt cosine similarity;
- edited/base predicted-noise norm ratio by timestep;
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

- compare the differentiable dense update with the current checkpoint-producing
  legacy SPEED formula;
- verify target cosine values for identical, orthogonal, opposite, and
  positively rescaled synthetic predictions;
- verify finite behavior for zero and near-zero prediction norms;
- verify cosine loss is computed per example before batch averaging;
- verify base and edited calls receive identical latents, timesteps, and target
  hidden states;
- verify parameter selection includes only `attn2.to_v.weight` names;
- optionally compare a later low-rank evaluation with the validated dense
  differentiable update;
- use `torch.autograd.gradcheck` in double precision on a small synthetic edit;
- verify that the anchor receives finite, non-zero gradients;
- verify that base U-Net and text-encoder parameters receive no gradients;
- verify frozen reference calls do not observe functional edited-model
  overrides and that a failed edited call cannot mutate the base U-Net;
- verify that `anchor == target` produces the zero legacy update;
- verify residual magnitude bounds;
- verify deterministic results with fixed seeds;
- verify CLI-over-YAML precedence, optimization-prompt fallback, and rejection
  of settings outside the initial `V`/`legacy`/`aug_num=0` scope;
- verify clear failure on singular, shape-mismatched, or non-finite inputs; and
- verify that the final detached checkpoint matches the effective weights used
  for validation within the configured numerical tolerance.

Run the focused tests first, followed by:

```bash
python -m unittest discover -s tests
```

## Success criteria

The hypothesis is supported if the differentiable anchor:

- lowers held-out target-prompt predicted-noise cosine similarity and reaches
  the same or stronger generation-level target erasure as the baselines;
- improves close-neighbor and broad retention at matched erasure strength;
- generalizes to held-out prompts, seeds, latents, and timesteps;
- produces bounded, numerically stable residuals and layer updates; and
- remains computationally practical for Stable Diffusion v1.4.

A reduction in target-prompt predicted-noise cosine similarity alone is
insufficient. The method must improve the final generation-level
erasure-preservation frontier.

## Recommended implementation phases

### Phase 1: Synthetic differentiability test

Implement the current dense legacy equation for small synthetic matrices,
including `P`, `M`, and `K2`. Verify numerical equivalence with the existing
checkpoint equation and confirm the anchor gradient with finite differences or
`gradcheck`.

### Phase 2: One-layer U-Net probe

Apply the differentiable update to one `attn2.to_v` layer and optimize a single
bounded residual using a small fixed latent bank. Confirm that the loss changes
and gradients reach only the residual.

### Phase 3: Full value-layer optimization

Enable all selected value-projection layers, optimize only the target cosine
loss, and select the best anchor using held-out target examples. Evaluate
retain prompts without using them for gradient updates.

### Phase 4: Generation evaluation

Detach the optimized anchor, produce the ordinary SPEED checkpoint, and run the
existing sampling and evaluation workflows. Compare methods at matched erasure
strength.

### Phase 5: Additional flexibility

Only after the basic hypothesis is supported, test retain and edit-size losses,
soft influence filtering, subspace-constrained residuals, layer-specific
residuals, multi-target anchors, and continuous input-token parameterization.

## Main risks

The main scientific risk is surrogate mismatch: lower one-step target cosine
may fail to erase the concept over a complete denoising trajectory. Held-out
full-generation evaluation is therefore mandatory.

The main optimization risk is that a free contextual residual exploits
directions that produce artifacts instead of semantic erasure. Residual bounds,
diagnostic monitoring, and generation-level model selection are the safeguards
in the cosine-only experiment. Later experiments can add edit regularization
and close-neighbor preservation losses.

The main engineering risk is memory use during U-Net backpropagation through
many effective weight updates. Frozen base weights, small minibatches, cached
base predictions, mixed precision, and gradient checkpointing should make the
first SD v1.4 prototype feasible. The algebraically equivalent low-rank form
remains a later optimization if the validated dense path is too costly.
