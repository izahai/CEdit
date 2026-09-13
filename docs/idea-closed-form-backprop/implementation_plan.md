# Implementation plan: differentiable closed-form anchor optimization

Status: planned; no training code or experiments implemented by this document.

Implement a separate SD v1.4 trainer that learns bounded contextual anchors
through the exact legacy SPEED edit. Deliver a reproducible single-target
experiment, a checkpoint usable by the existing samplers, and generation-level
evaluation before extending to multiple targets.

This plan follows [the handoff](HANDOFF.md) and
[the research proposal](differentiable_closed_form_anchor_optimization.md).
Repository observations below refer to the current working tree, including
the local `erasing-main/` reference tree.

## 1. Fixed scope

- Optimize only anchor direction and magnitude parameters. Freeze the U-Net,
  text encoder, and VAE; keep the U-Net in evaluation mode during both calls.
- Support `params=V`, `anchor_mode=legacy`, `baseline=SPEED`, and `aug_num=0`.
  Preserve the dense statistic, multiplication order, `P`, `M`, four-column
  `K2`, `lamb`, `threshold`, `retain_scale`, and `residual_scale`.
- Use exactly the mean per-example target-conditioned predicted-noise cosine
  as the optimization loss. Retention and norm measurements are diagnostics.
- Start with one non-nudity target and a legacy text anchor, including the
  empty prompt. Reject the special all-token nudity path in this prototype.
- Leave the existing `train_erase_null.py::edit_model` implementation intact.
  Do not modify or import `erasing-main/` from production code.
- Defer low-rank algebra, alternative objectives, projector refresh,
  `aug_num=10`, SDXL/FLUX, input-token learning, and layer-specific anchors.

## 2. Repository integration decisions

| Existing code | Implementation consequence |
| --- | --- |
| `train_erase_null.py::build_target_anchor_statistics` stacks `[N, 1, d]` embeddings and averages outer products | Use it as an independent parity reference. Add a pure legacy-only tensor implementation in `src/`, without the existing diagnostic SVD in every training graph. |
| `edit_model` selects the last subject token, including index zero for an empty anchor | Match this extraction exactly. Full prompt hidden states used for the noise objective are a separate input. |
| `edit_model` builds `K2` from the null BOS state and three k-means centers of the remaining null states | Preserve this construction and cache the actual result; record its seed and fingerprint. |
| With `aug_num=0`, retain influence filtering and augmentation are bypassed | Build one anchor-independent retain covariance/projector shared by the value layers. Preserve target-text exclusion, retain ordering, averaging, and the strict `S < threshold` rule. |
| `edit_model` accepts anchor strings and encodes them internally | An optimized continuous anchor cannot be passed directly into this API. Use a separate detached materializer over the validated edit state; see checkpoint parity below. |
| The editor writes a partial `.pt` state dictionary; both samplers use `torch.load` | Add a shared checkpoint loader supporting both legacy `.pt` and new `.safetensors`, and use it in both samplers. The new weights remain a partial U-Net state dictionary with original parameter names. |
| `requirements.txt` pins diffusers and transformers but does not directly declare torch or safetensors | Verify the runtime supports `torch.func.functional_call`, then explicitly record compatible direct dependencies and the versions used in the experiment. |

The materializer is a deliberate clarification of the handoff's instruction
to “call the ordinary SPEED checkpoint path”: the current string-only API
cannot represent a learned continuous anchor. Keep the legacy function
unchanged, preserve its equation and checkpoint key contract, and prove parity
against it for text anchors. Do not round a continuous anchor back to a prompt
or monkeypatch text encoding in production to export it.

## 3. Files and interfaces

Paths in this document are relative to the repository root.

| File to add or update | Responsibility |
| --- | --- |
| `src/differentiable_legacy_edit.py` | Pure dense statistics, frozen geometry preparation, name selection, effective parameters, and detached materialization. Torch-only numerical core. |
| `src/closed_form_anchor_training.py` | Bounded residual module, paired cosine objective, SD context/state preparation, anchor-only training, validation, and artifact metadata. Keep SD imports lazy so CPU tests can import the numerical components. |
| `src/edit_checkpoint.py` | Save new partial safetensors checkpoints and load both supported weight formats; validate keys, shapes, and finiteness. |
| `train_closed_form_backprop.py` | `argparse`, YAML precedence, early validation, and orchestration. |
| `configs/closed_form_backprop.yaml` | Explicit single-target configuration, including `aug_num: 0`. |
| `tests/test_differentiable_legacy_edit.py` | Dense formula, geometry, gradients, functional isolation, and materialization. |
| `tests/test_closed_form_anchor_training.py` | Residual bounds, loss, state sampling, optimizer scope, and best-anchor selection. |
| `tests/test_closed_form_backprop_config.py` | CLI/YAML and input-schema validation without model downloads. |
| `tests/test_edit_checkpoint.py` | Legacy/new format loading and invalid checkpoint failures. |
| `sample.py`, `sample2.py` | Replace only the checkpoint-loading operation with the shared loader. |
| `scripts/train_closed_form_backprop.sh` | Explicit GPU selection and quoted paths for the single-target run. |
| `remote_scripts/eval_closed_form_backprop/` | Reproducible training, sampling, evaluation, and experiment README after the local smoke run works. |

Use one consistent edit interface, with anchors rather than residuals at its
public boundary:

```python
edit_state = prepare_differentiable_legacy_edit(
    base_unet,
    target_embeddings,       # [N, 1, d], frozen
    retain_embeddings,       # [R, 1, d], frozen and already filtered
    null_hidden_states,      # [sequence_length, d], frozen; needed for K2
    config,
)
overrides, diagnostics = edit_state.effective_parameters(anchor_embeddings)
weights = edit_state.materialize(anchor_embeddings.detach())
```

The explicit null-hidden-state argument fills a missing input in the source
documents' interface sketch. The trainer owns tokenization and prompt handling;
the edit module owns matrix geometry. Internally allow prepared `K2`/geometry
fixtures for CPU tests without invoking CUDA k-means.

## 4. Milestones and acceptance gates

### Milestone 1 — Establish exact dense forward equivalence

Implement the numerical core before connecting a diffusion model:

```python
residuals = residual_scale * (anchors - targets)
C = torch.stack([t.T @ t for t in targets]).mean(0)
D = torch.stack([r.T @ t for r, t in zip(residuals, targets)]).mean(0)
M = torch.linalg.inv(C @ P + retain_scale * I)
H_inv = torch.linalg.inv(K2.T @ P @ M @ K2 + lamb * I2)
delta_W = W @ D @ P @ (I - M @ K2 @ H_inv @ K2.T @ P) @ M
W_effective = W + delta_W
```

Cache targets, `C`, base weights, identities, `K2`, `P`, `M`, and `H_inv`.
Do not regroup the chained dense update into a low-rank expression or a
precombined right-hand operator in this milestone. Retain the inverse form
first; a later solve implementation needs its own equivalence test.

Compute retain covariance with the legacy chunked reduction and construct `P`
using its SVD and strict threshold rule. Preserve the seeded retain shuffle
for comparison because reduction order can affect floating-point results.
Cache the realized geometry rather than rerunning k-means at export time.

Use float32 for initial real-model geometry/weights and float64 for small
gradient checks. Validate shapes, nonempty inputs, finite scalars/tensors, and
the two inverses. On singular or non-finite geometry, fail with the affected
matrix/configuration identified; do not silently add jitter, use a
pseudoinverse, or change `lamb`.

Acceptance:

- Single-target and multi-target `C`/`D` match the existing statistic helper,
  including non-unit `residual_scale`.
- Dense weights match an independent legacy-equation reference on well
  conditioned synthetic inputs. Start with float64 `rtol=1e-8, atol=1e-10`
  and float32 `rtol=1e-5, atol=1e-6`; record maximum absolute/relative errors.
- `anchor == target` yields zero update; frozen geometry is deterministic.
- `torch.autograd.gradcheck` passes on a small double-precision equation;
  each anchor receives finite nonzero gradients under a nondegenerate probe
  loss. Include singular and shape-mismatch failure cases.

### Milestone 2 — Prove functional execution preserves the gradient

Discover weights using `unet.named_parameters()` and exact name components
ending in `attn2.to_v.weight`. Reject no matches and incompatible input widths.
First exercise a small attention-like test module, then one real value layer
using an internal probe helper. Keep the production CLI restricted to all
selected value layers.

Pass computed non-leaf weights directly to `torch.func.functional_call` with
a partial override mapping. All other parameters and buffers remain the
frozen model's values. Never construct `Parameter(effective_weight)`, copy
through `.data`, detach effective weights, or load a state dictionary in the
optimization path. Detach diagnostic measurements only.

Acceptance:

- A loss on the functional output differentiates back to the anchor.
- Frozen U-Net/text-encoder parameters have no gradients and are absent from
  the optimizer. Only intended value weights appear in overrides.
- Base weights, parameter identities, buffers, and reference predictions remain
  unchanged after successful and deliberately failing edited calls.
- A real SD v1.4 one-layer probe works before enabling all value layers.

### Milestone 3 — Implement bounded anchors and paired noise batches

Parameterize the unscaled contextual residual as:

```python
direction = raw_direction / raw_direction.norm(dim=-1, keepdim=True).clamp_min(eps)
magnitude = max_residual_norm * raw_magnitude.sigmoid()
anchor = target + magnitude * direction
```

Initialize direction from the legacy text-anchor residual. Initialize magnitude
using the inverse sigmoid of its norm divided by the configured bound, clamped
away from zero and one. If that norm exceeds the bound, record the clipping
and retain both original and feasible initialization vectors. If it is zero,
use a seeded small nonzero residual with an interior magnitude. Exact identity
predictions are a stationary point of cosine, so a zero-edit initialization
must not be the training default.

`max_residual_norm` bounds `anchor - target`; the residual entering SPEED is
scaled once by `residual_scale`. Log both norms. A value such as `1.0` is an
initial experimental setting, not a demonstrated appropriate CLIP-space bound.

Port only the needed random-prefix sampling mechanics from
`erasing-main/utils/esd_trainer.py` and `utils/sd_utils.py` into the local
trainer. Use a frozen-base trajectory, no ESD negative-guidance target:

1. Encode and cache full target-prompt and null hidden states under no-grad.
2. Reset the scheduler for each trajectory, sample seeded initial latents,
   and choose prefix index `k` in `[0, num_inference_steps - 1]`.
3. Run exactly the first `k` scheduler steps with the frozen base and configured
   classifier-free guidance. Evaluate at `scheduler.timesteps[k]`; `k=0`
   must represent the initial noisy state.
4. Apply required scheduler input scaling once and use identical resulting
   latents, timestep, and target hidden states in both loss calls. Start with
   the loaded SD v1.4 scheduler and save its class/configuration.
5. Obtain the base conditional prediction under `torch.no_grad()` and the
   edited conditional prediction with autograd enabled. Guidance controls
   trajectory sampling; the loss compares raw target-conditioned predictions.

Use a shared prefix timestep within a minibatch initially, with independent
latent seeds. Keep training and validation generators separate; validation
must not advance the training generator or leave scheduler state behind.

Acceptance:

- Residual bounds and finite gradients hold at ordinary, near-zero, and
  saturated initialization values.
- Cosine is computed in float32, flattened per example, then averaged. Tests
  cover identical/orthogonal/opposite vectors, positive rescaling, unequal
  per-example norms, and zero/near-zero finite behavior.
- A spy model verifies identical inputs to the paired calls; scheduler tests
  cover prefix endpoints, deterministic seeds, scaling, and reset behavior.

### Milestone 4 — Complete the single-target training command

Use `argparse` with YAML defaults followed by explicit CLI overrides. Validate
the merged values, including YAML types, before loading model weights. Require
`target_concepts`, `retain_path`, and `heads`; allow the empty anchor string.
Reject unsupported modes, duplicate/empty targets, mismatched anchor counts,
invalid CSV columns, empty retain sets after exclusion, and non-finite or
invalid ranges. Use positive learning rate, steps, batch size, residual bound,
cosine epsilon, and retain scale; nonnegative `lamb`; positive residual scale
for this optimizer. Reject target lists longer than one until Milestone 7.

Preserve the source CLI names. Add `--optimization_prompts_path`,
`--validation_prompts_path`, `--validation_seed`, `--validation_samples`,
`--validation_interval`, `--resolution`, and `--device` for reproducible input
and validation control. Start at resolution 512, validation interval 10,
16 validation states, and a recorded validation seed distinct from training.

An optimization CSV requires `prompt`. Omission of the path falls back to the
target concept; an explicitly supplied missing or malformed file is an error.
Without a validation CSV, validate on the same prompt with held-out seeds and
timesteps, and label this as state generalization only. Claims about prompt
generalization require a separate held-out prompt set.

The training loop must:

1. Prepare frozen edit geometry and a fixed, detached validation bank.
2. Evaluate and save the feasible initialization as candidate step zero.
3. Sample a training batch, build all effective weights, compute only cosine
   loss, backpropagate, check finite gradients, and update anchor parameters
   with Adam. Do not reuse a previous step's autograd graph.
4. At validation intervals and the final step, evaluate without gradient and
   select the lowest held-out cosine; keep the earlier step on a tie.
5. Snapshot the best anchor/raw parameters by detached clone. Rebuild weights
   from that snapshot for final validation/export, not from the last iterate.

Fail explicitly on non-finite loss, gradients, or parameters and retain the
last valid artifact with failure status. Log zero gradients as a diagnostic;
do not claim every real training step must have a nonzero gradient.

Write `config.yaml` and `metrics.jsonl` with training/validation cosine,
prediction-norm ratios, raw/scaled residual norms, per-target gradient norms,
per-layer update norms, projector rank, wall time, and peak GPU memory.
Measure held-out retain prediction changes during validation only. Do not
backpropagate retention or use it as an unannounced auxiliary objective.

Acceptance: a synthetic trainer test proves only anchor parameters change,
validation is deterministic, and export uses the best snapshot. Then run a
short CUDA smoke experiment before the proposed 200-step Snoopy run. Start
with float32; introduce reduced-precision U-Net execution or checkpointing
only if memory measurements justify it and parity/gradient tests still pass.

### Milestone 5 — Export and verify the exact validated edit

Under no-grad, materialize the best continuous anchor from the same frozen
edit state and original weights. Save CPU contiguous tensors using the original
value-projection names in `<file_name>.safetensors` (default `weight`).

Save `anchor_optimization.pt` containing a versioned SPEED-specific format ID,
best anchors/residuals and raw parameters, target/anchor text, selected names,
best step/metrics, full resolved configuration, input hashes, model identity,
package versions, seeds, projector policy, and frozen statistics needed to
reproduce the edit (`C`, `P`, `K2`, plus target embeddings and fingerprints).
Do not put live graphs or the whole pipeline in the artifact. Resuming optimizer
state is outside the initial artifact contract.

Prove two separate equivalences:

- **Legacy equivalence:** for fixed text anchors and `aug_num=0`, run the
  untouched `edit_model` and the new module with matching inputs and RNG state
  immediately before preparation. Compare every selected weight; inspect
  `K2`, retain covariance, and reduction order if results differ.
- **Continuous-anchor export equivalence:** compare reloaded safetensors
  weights with the selected functional weights, then compare predictions of a
  fresh base U-Net loaded with that partial checkpoint against functional
  predictions on the same fixed batch. Save errors and tolerance results.

Do not relax tolerances just to obtain a pass. Diagnose conditioning, dtype,
randomness, or operation-order differences first. Mixed-precision sampling
needs its own dtype-matched prediction tolerance, recorded separately from
float32 weight parity.

Integrate the shared format loader into `sample.py` and `sample2.py`; preserve
legacy `.pt` behavior and partial-state loading. Validate unexpected keys and
incompatible tensor shapes before application. Always pass `--edit_ckpt`
explicitly in the new workflow, avoiding directory-based latest-file selection.

Acceptance: both checkpoint formats load in CPU tests, every exported weight
passes parity, and an SD sampling smoke run consumes the safetensors output.

### Milestone 6 — Evaluate the single-target research hypothesis

Build a dedicated workflow using the existing few-concept and paper-comparison
workflows as references. Keep dataset paths, GPU assignments, model/config
versions, prompt lists, and seeds explicit. Store outputs under `logs/` or
the workflow output root, never in version control.

Compare the unchanged base model, ordinary legacy text-anchor SPEED at
`aug_num=0`, the bounded initialization at step zero, and the optimized anchor
at `aug_num=0`. The step-zero control isolates learning from initialization
clipping. Label legacy `aug_num=10` separately if added later. TGPRS is an
optional existing-method comparison after the required controls work.

Generate paired target, paraphrase, close-neighbor, and broad-retention images
using identical prompts and seeds across methods. Report target CLIP and
applicable detector accuracy, neighbor preservation, MS-COCO CLIP/FID, visible
artifacts, training cost, and peak memory. Record dataset sizes and metric
configuration; distinguish smoke metrics from the complete benchmark.

After one complete run, perform a small declared sweep over learning rate,
residual bound, and edit strength. Select operating points on validation data,
then compare retention at matched generation-level erasure on held-out data.
Use separate output directories for each configuration. Include qualitative
examples even when cosine improves, because the surrogate can reward damage.

Acceptance: a results report states whether optimization improves generation
erasure/preservation, fails, or is inconclusive. A valid negative result still
completes the implementation; cosine reduction alone does not support the
research hypothesis.

### Milestone 7 — Extend the validated path to multiple targets

After the single-target gates pass, remove the temporary one-target parser
restriction. Broadcast one text anchor or require one anchor per target.
Keep independent bounded residual parameters with shape `[N, 1, d]` and use
the complete mean statistic for every effective edit, including when a batch
contains prompts for only one target.

For multi-target prompt files, require `target_concept` alongside `prompt` and
validate exact membership; omitted files produce one prompt per target.
Sample targets uniformly, then prompts within target. Validate with equal
states per target and report per-target metrics plus their macro mean. Select
the best anchor by macro-mean held-out cosine and report hard targets
separately; defer soft-max objectives and hard-target sampling.

Acceptance: synthetic multi-target gradients and serialization pass; a small
two-target CUDA run completes; every exported weight matches; generation
evaluation reports erasure for each target rather than only a global average.

## 5. Implementation validation order

Run the smallest relevant `unittest` modules after each code milestone, then:

```bash
python -m unittest discover -s tests
```

CPU tests must not download models or require CUDA. Mark real-model checks as
explicit GPU probes/workflow steps. The progression is numerical parity →
functional gradient → SD one-layer probe → full-value smoke training → legacy
and checkpoint parity → complete single-target evaluation → multi-target run.

The planned first full command, once Milestones 1–5 are implemented, is:

```bash
CUDA_VISIBLE_DEVICES=0 python train_closed_form_backprop.py \
  --config configs/closed_form_backprop.yaml \
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

For the smoke run, override `--anchor_steps 2`,
`--validation_interval 1`, and `--validation_samples 2`, and use a separate
`save_path`. Record whether validation was CPU-only, a GPU smoke check, or a
complete generation benchmark whenever reporting implementation progress.
