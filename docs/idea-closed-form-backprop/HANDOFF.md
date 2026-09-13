# Handoff: Minimal-Change Differentiable Legacy Anchor Optimization

## Objective for the next session

Implement a minimal prototype that learns continuous anchor embeddings by
backpropagating a predicted-noise objective through the existing legacy SPEED
closed-form edit.

Preserve the legacy method's forward equation, including its dense
target-anchor statistic, retain projector `P`, matrix `M`, four-vector null
prompt constraint `K2`, `lamb`, and `retain_scale`. Do not replace the equation
with the no-`K2`, Sherman-Morrison, Woodbury, or low-rank formulations in the
first implementation. Those derivations remain useful for intuition and future
optimization only.

The full research proposal is recorded in:

- `docs/new-idea/differentiable_closed_form_anchor_optimization.md`

## Central idea

The current workflow chooses or learns anchors before editing. The proposed
workflow makes the anchor a parameter of the edit itself:

```text
anchor embedding
    -> legacy target-anchor statistic
    -> exact legacy SPEED effective weights
    -> edited U-Net predicted noise
    -> erasure and preservation losses
    -> gradient back to anchor embedding
```

Freeze the original U-Net and text encoder. Optimize only one anchor embedding
per target, or equivalently one residual per target.

## Exact legacy equation to preserve

For non-nudity targets, stack contextual target and anchor embeddings as:

```text
T: target embeddings       [N, 1, d]
A: learnable anchors       [N, 1, d]
R = scale * (A - T)        [N, 1, d]
```

Construct the same statistics as `build_target_anchor_statistics`:

```python
sum_target_target = torch.stack([
    target.T @ target
    for target in target_embeddings
]).mean(0)

target_anchor_delta = torch.stack([
    residual.T @ target
    for residual, target in zip(
        residuals,
        target_embeddings,
    )
]).mean(0)
```

For each edited layer, preserve the current equation and multiplication order:

```python
M = torch.linalg.inv(
    sum_target_target @ P
    + retain_scale * I
)

delta_weight = (
    layer_weight
    @ target_anchor_delta
    @ P
    @ (
        I
        - M
        @ K2
        @ torch.linalg.inv(
            K2.T @ P @ M @ K2
            + lamb * I2
        )
        @ K2.T
        @ P
    )
    @ M
)

effective_weight = layer_weight + delta_weight
```

Using `torch.linalg.solve` in place of an explicit inverse is acceptable only
if a numerical test confirms equivalence. Do not algebraically remove any term
from the legacy method during the first prototype.

For one target, `target_anchor_delta` is rank at most one. For many targets, it
has rank at most `N`. Keep the dense construction initially even though an
equivalent low-rank evaluation exists.

## What can be precomputed

The following quantities do not depend directly on the learnable anchor and
can be prepared once when retain geometry is frozen:

- target embeddings;
- `sum_target_target`;
- null-prompt hidden states, clustering result, `K2`, and `I2`;
- retain embeddings and retain covariance;
- original layer weights;
- identity matrix `I`;
- retain projector `P` when it is intentionally frozen;
- `M` and the inner `K2` inverse when `P` is frozen.

The learnable path must retain:

```text
A -> residuals -> target_anchor_delta -> delta_weight
  -> effective_weight -> predicted_noise -> loss
```

Do not detach `A`, `residuals`, `target_anchor_delta`, `delta_weight`, or
`effective_weight`.

### Retain projector caveat

The repository default is `aug_num=10`. In that path, preliminary
`erase_weight` affects retain filtering and perturbation, so `P` indirectly
depends on the anchor through hard selections. This path is not smoothly
differentiable.

For the first gradient-path prototype, use one of these explicit policies:

1. Preferred: calculate the normal legacy `P` once from the initialization
   anchor and freeze it throughout anchor optimization.
2. Strict forward refresh: recompute `P` from the current anchor under
   `torch.no_grad()` using deterministic perturbations, while treating `P` as
   stop-gradient.

The first policy is more stable and isolates the question of whether the
anchor can be learned through the legacy closed-form edit. After anchor
optimization, run the normal legacy checkpoint builder once with the detached
best anchor and its configured legacy settings.

Do not silently change `aug_num` or remove filtering. Record the chosen
projector policy in the experiment configuration and artifact.

## Functional U-Net execution

Keep the existing `@torch.no_grad()` `edit_model` function unchanged for
reproducibility and checkpoint generation. Add a separate differentiable
module under `src/`.

The module should expose a small interface:

```python
edit_state = prepare_differentiable_legacy_edit(
    base_unet,
    target_embeddings,
    retain_embeddings,
    config,
)

parameter_overrides, diagnostics = edit_state.effective_parameters(
    anchor_embeddings,
)
```

Use `torch.func.functional_call` to execute the U-Net with graph-connected
effective weights. Build a mapping from `unet.named_parameters()` and override
only the selected `attn2.to_v` weights. Freezing base parameters with
`requires_grad_(False)` does not block gradients through products involving
the learnable anchor.

Do not use these operations in the optimization path:

```python
parameter.data.copy_(effective_weight)
unet.load_state_dict(...)
effective_weight.detach()
```

They disconnect the predicted-noise loss from the anchor.

## Loss decisions

Use the original model as a stop-gradient teacher. For each target-containing
prompt, provide a context-matched counterfactual prompt in which only the
target identity or style is removed.

```text
target:         "Snoopy riding a bicycle"
counterfactual: "a cartoon dog riding a bicycle"
```

The initial erasure objective is predicted-noise MSE between:

- the edited model conditioned on the target prompt; and
- the original model conditioned on the counterfactual prompt.

For retain prompts, minimize predicted-noise MSE between the edited and
original models under the same prompt, latent, noise, and timestep.

Do not minimize target-image diffusion loss against sampled Gaussian noise;
that teaches the model to reconstruct the target. Do not simply maximize it;
arbitrary model damage can satisfy that objective.

A basic scalarized loss is:

```text
loss = erase_loss
     + retain_weight * retain_loss
     + edit_weight * normalized_edit_norm
```

Use per-target erasure losses for many-target optimization. A global mean can
hide hard targets, so later use a soft maximum, hard-target sampling, or one
constraint/dual variable per target. Compare preservation only at matched
erasure strength.

## Anchor parameterization

Start with one learnable continuous contextual anchor per target. A bounded
residual is safer than an unconstrained anchor:

```text
direction_i = raw_direction_i / ||raw_direction_i||
magnitude_i = max_norm * sigmoid(raw_magnitude_i)
residual_i = magnitude_i * direction_i
anchor_i = target_i + residual_i
```

Initialization can use the existing legacy text-anchor residual. Log the norm
and gradient norm for every target. A continuous contextual anchor is an
internal edit parameter and need not decode to a valid discrete prompt.

## Interpretation retained from the discussion

For one target, the legacy edit is algebraically rank one. It can be written as
`delta_W = u @ v`, where:

- `u = W @ (anchor - target).T` is the output residual or value written by the
  edit;
- `v` is the input-side target detector or read vector derived from the target,
  retain geometry, regularization, and, in the exact legacy method, the `K2`
  correction.

For an input embedding `x`, `delta_W @ x = u * (v @ x)`. Anchor learning changes
what is written through `u`; it does not freely learn which inputs trigger the
edit through `v`. This limits how much anchor optimization alone can repair
collateral activation on semantically related concepts.

This factorization is explanatory. Do not use it to replace the dense legacy
formula in the minimal prototype.

## Minimal implementation sequence

1. Extract a pure differentiable implementation of the existing dense legacy
   formula without changing the checkpoint path.
2. Add a synthetic equivalence test against the current legacy result with a
   fixed anchor.
3. Verify a finite, non-zero anchor gradient on synthetic matrices.
4. Connect graph-preserving effective weights to one `attn2.to_v` layer through
   `functional_call`.
5. Verify the base U-Net receives no gradients while the anchor does.
6. Extend to all selected value-projection layers.
7. Add one-target predicted-noise optimization.
8. Extend the anchor tensor and loss accounting to many targets.
9. Detach the best anchor and generate a normal legacy SPEED checkpoint.
10. Run full generation evaluation at matched erasure strength.

## Required tests

- Differentiable dense legacy output equals the current legacy checkpoint
  equation for fixed inputs.
- Single-target and many-target statistic construction matches
  `build_target_anchor_statistics`.
- `torch.autograd.gradcheck` passes on a small double-precision dense equation.
- Each anchor receives a finite, non-zero gradient.
- Original model parameters receive no gradient.
- `anchor == target` produces a zero target-anchor delta and zero update.
- Frozen-projector behavior is deterministic.
- Final detached checkpoint weights match the functional effective weights
  within the configured numerical tolerance.

Run focused tests first, followed by:

```bash
python -m unittest discover -s tests
```

GPU execution and full generation evaluation remain necessary. Improvement in
predicted-noise loss alone is not evidence of successful concept erasure.

## Suggested skills

- `codebase-design`: keep frozen edit-state preparation separate from
  differentiable effective-weight construction.
- `tdd`: establish exact legacy equivalence and anchor gradients before U-Net
  integration.
- `diagnosing-bugs`: use if gradients disappear, `functional_call` is
  incompatible with the diffusers U-Net, or the new dense path differs from the
  current checkpoint equation.

## Workspace caution

Inspect `git status` before editing. Preserve existing user changes and never
modify the read-only `Diffusion-MU-Attack-main/` reference directory.
