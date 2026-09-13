# Handoff: Minimal-Change Differentiable Legacy Anchor Optimization

## Objective for the next session

Implement a minimal prototype that learns continuous anchor embeddings by
minimizing target-conditioned predicted-noise cosine similarity through the
existing legacy SPEED closed-form edit. The required SD training mechanics are
implemented in this repository; keep the prototype independent of the removed
external reference tree and do not introduce the ESD objective.

Preserve the legacy method's forward equation, including its dense
target-anchor statistic, retain projector `P`, matrix `M`, four-vector null
prompt constraint `K2`, `lamb`, and `retain_scale`. Do not replace the equation
with the no-`K2`, Sherman-Morrison, Woodbury, or low-rank formulations in the
first implementation. Those derivations remain useful for intuition and future
optimization only.

The full research proposal is recorded in:

- `docs/idea-closed-form-backprop/differentiable_closed_form_anchor_optimization.md`

## Central idea

The current workflow chooses or learns anchors before editing. The proposed
workflow makes the anchor a parameter of the edit itself:

```text
anchor embedding
    -> legacy target-anchor statistic
    -> exact legacy SPEED effective weights
    -> edited U-Net predicted noise
    -> target cosine loss
    -> gradient back to anchor embedding
```

Freeze the original U-Net and text encoder. Optimize only one anchor embedding
per target, or equivalently one residual per target.

## Local SD training boundary

Use the local SD training and legacy editor modules as the reference for:

- SD pipeline setup and configuration;
- prompt encoding and frozen text/VAE setup;
- sampling \(x_t\) by running a random prefix of the frozen denoising process;
- the SD U-Net calling convention; and
- name-based parameter discovery and metadata-rich checkpoint conventions.

The ESD negative-guidance target is explicitly out of scope. The research
objective instead compares base and edited target-conditioned noise predictions
using cosine similarity.

The new student weights are non-leaf tensors computed from the anchor, so
execute them with `torch.func.functional_call` and leave the base U-Net
untouched. Limit the prototype to SD v1.4 and `attn2.to_v.weight`; other model
families and parameter subsets are out of scope.

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

For the first gradient-path prototype, explicitly use `aug_num=0`. This makes
the retain covariance and `P` anchor-independent while retaining the complete
closed-form equation. It also lets the detached legacy checkpoint builder
reproduce the functional weights exactly.

After that parity milestone, test the repository default `aug_num=10` by
calculating the normal legacy `P` once from the initialization anchor and
freezing it throughout optimization. Supporting that mode requires a
checkpoint materializer that accepts the frozen projector state; rerunning the
current legacy builder with the optimized anchor would recompute filtering and
need not match the weights used for validation.

Do not silently change `aug_num` in baseline comparisons. Record the value and
projector policy in the experiment configuration and artifact. A
stop-gradient projector refresh with deterministic perturbations is a later
ablation, not part of the minimal prototype.

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
only names containing `attn2.to_v` and ending in `.weight`; fail if no names
match. Freezing base parameters with
`requires_grad_(False)` does not block gradients through products involving
the learnable anchor.

Pass the effective tensors directly in the override mapping. Do not wrap them
in `torch.nn.Parameter`, which would create new leaves and break the gradient
path back to the anchor.

Do not use these operations in the optimization path:

```python
parameter.data.copy_(effective_weight)
unet.load_state_dict(...)
effective_weight.detach()
```

They disconnect the predicted-noise loss from the anchor.

After selecting the best anchor, save both the anchor/configuration artifact
and a materialized legacy SPEED U-Net checkpoint. Reuse the local checkpoint
metadata implementation, but use a SPEED-specific format identifier rather
than `erasing-esd-v2`.

## Loss decisions

For the same noisy latent, timestep, and target-prompt hidden states, compute:

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

erase_loss_per_example = F.cosine_similarity(
    edited_prediction.float().flatten(1),
    base_prediction.float().flatten(1),
    dim=1,
    eps=cosine_eps,
)
erase_loss = erase_loss_per_example.mean()
```

Minimize `erase_loss`. Similarity near 1 means the edit has not changed the
target-conditioned noise direction; 0 is orthogonal and -1 is opposite. The
cosine objective avoids raw predicted-noise scale differences across
timesteps. Compute it independently for every batch item and then average.

The base prediction is stop-gradient. The edited prediction must retain its
gradient through the functional effective weights and into the anchor. Do not
detach it. Both calls must receive tensor-identical `x_t`, `timestep`, and
`target_hidden_states`.

Compute cosine similarity in float32 and use an explicit epsilon. Log the
edited/base prediction-norm ratio because cosine is poorly conditioned near a
zero norm and does not itself penalize destructive scale changes.

Low cosine is only a training surrogate. For the first experiment, use it as
the only optimization loss:

```text
loss = erase_loss
```

Do not add retention loss, edit-size regularization, norm penalties, or a
constrained formulation to this initial run. Log retention metrics, edit norms,
prediction-norm ratios, and finite-gradient status without feeding them into
the optimizer. Full generation and retention evaluation are required to detect
general model damage.

Combining cosine loss with preservation or edit-size terms is a later
experiment, after the cosine-only result is established.

Use per-target erasure metrics for many-target optimization. A global mean can
hide hard targets, so later use a soft maximum or hard-target sampling. Compare
preservation only at matched erasure strength.

## CLI interface

Add a dedicated `train_closed_form_backprop.py` entry point. A direct
single-target run should look like:

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

The learning rate is illustrative and must be tuned. Also support
`--config configs/closed_form_backprop.yaml`, with explicit CLI values taking
precedence. The optimization prompt CSV requires a `prompt` column; if omitted,
use each target concept as its sole optimization prompt. `retain_path` remains
required for the SPEED projector, not for a retain loss.

The parser must reject settings outside the initial scope:

```text
params != V
anchor_mode != legacy
aug_num != 0
```

Do not add loss-weight flags or a generic objective selector yet. Save
`weight.safetensors` and `anchor_optimization.pt` under `save_path`.

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
7. Implement and independently test the per-example target-conditioned cosine
   objective using the local frozen SD state-sampling mechanics.
8. Add one-target anchor optimization against the cosine objective.
9. Extend the anchor tensor and loss accounting to many targets.
10. Detach the best anchor and generate a normal legacy SPEED checkpoint.
11. Run full generation evaluation at matched erasure strength.

## Required tests

- Differentiable dense legacy output equals the current legacy checkpoint
  equation for fixed inputs.
- Single-target and many-target statistic construction matches
  `build_target_anchor_statistics`.
- Cosine loss returns 1, 0, and -1 for identical, orthogonal, and opposite
  synthetic predictions and is invariant to positive rescaling.
- Zero and near-zero prediction norms produce finite loss and gradients.
- Cosine is calculated per example before averaging across the batch.
- Base and edited calls receive identical latents, timesteps, and target hidden
  states.
- Parameter selection returns only `attn2.to_v.weight` names and fails on an
  empty match.
- `torch.autograd.gradcheck` passes on a small double-precision dense equation.
- Each anchor receives a finite, non-zero gradient.
- Original model parameters receive no gradient.
- Frozen reference execution sees the unchanged base weights, including after
  a failed functional edited-model call.
- `anchor == target` produces a zero target-anchor delta and zero update.
- Frozen-projector behavior is deterministic.
- CLI values override YAML, a missing optimization prompt file falls back to
  target concepts, and unsupported prototype modes fail early.
- Final detached checkpoint weights match the functional effective weights
  within the configured numerical tolerance.

Run focused tests first, followed by:

```bash
python -m unittest discover -s tests
```

GPU execution and full generation evaluation remain necessary. Improvement in
target-prompt predicted-noise cosine alone is not evidence of successful
concept erasure.

## Suggested skills

- `codebase-design`: keep frozen edit-state preparation separate from
  differentiable effective-weight construction.
- `tdd`: establish exact legacy equivalence and anchor gradients before U-Net
  integration.
- `diagnosing-bugs`: use if gradients disappear, `functional_call` is
  incompatible with the diffusers U-Net, or the new dense path differs from the
  current checkpoint equation.

## Workspace caution

Inspect `git status` before editing. Preserve existing user changes. Treat
the removed external reference tree as unavailable. The production path must
use the repository's local `src/`, trainer, sampler, and checkpoint modules.
