# ReSAM implementation plan

Status: implementation plan. This follows [the method](ReSAM-idea.md) and
[CLI specification](cli_usage.md).

Implement a single-target trainer that learns scores over a fixed bank of
contextual concept embeddings, forms a sparse top-k anchor, and passes that
anchor to the existing differentiable closed-form SPEED edit. Deliver the
training command, configuration, checkpoint, metadata, and CPU-only tests.
Running a pretrained diffusion model and evaluating image quality are outside
this implementation plan.

## 1. Implementation contract

- Support exactly one target concept, supplied as a CLI string or singleton
  YAML list. Reject multiple targets before model loading.
- Use SPEED value editing with legacy anchor geometry and no augmentation.
  Honor the CLI document's defaults, including retain projection rank 1,
  seed 0, and straight-through estimation enabled. Reject unsupported modes.
- Freeze the teacher U-Net, text encoder, candidate embeddings, target
  embedding, and SPEED geometry. Optimize only one score per candidate.
- Train on predicted-noise mean squared error between teacher and edited U-Net
  outputs on retain prompts. Each pair uses the same noisy latent, timestep,
  and full prompt conditioning. The candidate mixture affects only the
  differentiable SPEED value-weight overrides.
- Optionally include null-prompt preservation via `--use_null_retain_loss`,
  computing \(\mathcal{L} = 0.5 \cdot (\mathcal{L}_{\text{retain}} + \mathcal{L}_{\text{null}})\)
  on the exact same diffusion state.
- No validation-based checkpoint rollback or threshold gating: training runs
  for the configured number of steps and exports the final checkpoint directly.

## 2. Code layout

| Path | Responsibility |
| --- | --- |
| src/resam_training.py | Candidate parsing, score module, sparse mixture, retain loss, null preservation loss, optimization loop, and diagnostics. Keep numerical functions independent of Diffusers. |
| src/diffusion_training_inputs.py | Shared retain-text loading, contextual embedding extraction, prompt encoding, and diffusion-state builders extracted from train_closed_form_backprop.py. Update that entry point to import the shared helpers without changing behavior. |
| train_resam.py | YAML/CLI parsing, validation, model orchestration, logging, artifacts, and checkpoint export. Import Diffusers only inside the runtime path. |
| configs/resam_train.yaml | Documented single-target training configuration with candidate path and optional null retain loss. |
| docs/ReSam/cli_usage.md | Document the CLI parameters, configuration, and defaults. |
| tests/test_resam_training.py and tests/test_resam_config.py | CPU-only tests using small tensors and fake modules. |

Reuse the existing differentiable SPEED state for effective parameters and
materialization, paired predictions and diffusion-state sampling from the
closed-form anchor trainer, and the shared partial safetensors checkpoint
writer/loader. The samplers already load that format.

## 3. Implementation steps

### Step 1: Parse and encode the candidate bank

Accept a CSV with a required concept column and optional id, or a text file
with one concept per line. Preserve file order, trim whitespace, and reject
empty values, duplicates, an empty bank, and k outside [1, bank size]. Hash
the source file for the run artifact.

At runtime, extract each candidate's last subject-token contextual embedding
with the same tokenizer/text-encoder logic used for the target and current
SPEED editor. Store detached embeddings of shape [V, 1, d]; validate
finiteness, width, dtype, and device. Keep the bank frozen.

### Step 2: Implement the sparse score module

Create one trainable score vector [V] and store the embedding bank as a
buffer. Select the highest k scores with a deterministic file-order tie rule.
Apply temperature softmax within the selected set, set all other weights to
zero, and compute one anchor of shape [1, 1, d].

With straight-through estimation enabled, use the documented expression
hard.detach() - soft.detach() + soft: hard is the sparse top-k softmax and
soft is the dense softmax. The forward value remains sparse, while the
backward surrogate reaches every score. Without it, use hard directly.
Validate positive finite temperature and finite scores. Expose detached
selected indices, names, hard weights, anchor-target distance, and selection
changes. Start from deterministic scores and record their initialization.

### Step 3: Wire the retain objective to SPEED

Prepare target and retain embeddings and frozen SPEED geometry once. Filter
the target out of the retain set using the current loader. For each training
state, form the sparse anchor, compute effective value weights, and obtain
teacher and edited predictions through the existing paired-prediction helper.
Calculate per-example mean squared error in float32 and average across the
batch. Backpropagate to scores with Adam. Check loss, gradients, scores,
anchor, and effective weights for non-finite values. Never copy edited
weights into the teacher during training.

When `--use_null_retain_loss` is enabled, compute the predicted-noise MSE on
unconditional/null-prompt conditioning `""` on the exact same diffusion state
and combine as \(\mathcal{L} = 0.5 \cdot (\mathcal{L}_{\text{retain}} + \mathcal{L}_{\text{null}})\).

Define the retain loss as the mean over batch examples of each example's
elementwise mean squared predicted-noise difference. This divides the idea
document's squared L2 norm by the number of prediction elements, keeping
the loss scale consistent across latent sizes. For a fixed resolution, both
forms rank anchors identically, though they scale gradients differently.

Reuse the existing frozen-prefix state sampler for timesteps. At each training
step, uniformly draw a prefix index from 0 through num_inference_steps - 1.
The sampler initializes seeded noise, advances the frozen teacher and scheduler
through that many inference steps, then evaluates both U-Nets at the scheduler
timestep at that index on the same scaled latent. Index 0 evaluates the initial
noise.

### Step 4: CLI, artifacts, and export

Implement YAML defaults with explicit CLI overrides. Validate types and
paths before loading the model. Support the CLI document's model, concept,
candidate, retain, sparse-mixture, optimization, and output options. Keep documented defaults.

Write resolved config.yaml, per-step metrics.jsonl, and a versioned
resam_optimization.pt. Include candidate names and input hash, initial and
final scores, selected candidates and weights, final step and loss,
target/anchor embeddings, geometry metadata, seeds, and run status.
On completion, save a partial U-Net checkpoint as file_name.safetensors with the
existing parameter names. Reload it and check exact tensor equality with
the materialized weights. Assert the base U-Net remains unchanged.

## 4. CPU-only verification

Tests must use synthetic embeddings and small fake U-Nets. They must not
instantiate a Stable Diffusion pipeline, access a model hub, require CUDA,
or download weights.

- Candidate parser: CSV/TXT formats, ordering, blanks, duplicates, missing
  columns, and invalid k.
- Sparse module: forward sparsity, weight normalization, deterministic ties,
  k=1, k=V, and gradients with and without straight-through estimation.
- Training: matched teacher/edited inputs, gradients reaching only scores,
  correct retain MSE, null-prompt preservation loss calculation, finite-value failures, and unchanged base parameters.
- Configuration and export: YAML/CLI precedence, invalid values rejected
  before model setup, artifact fields, and safetensors round trip.
- Regression: existing closed-form trainer helper behavior remains covered
  after extracting shared input functions.

Run focused tests, then repository discovery:

    python -m unittest tests.test_resam_training tests.test_resam_config
    python -m unittest discover -s tests

Completion means the CLI, numerical training path, artifacts, and export are
implemented and all CPU-only tests pass. Model execution and research
evaluation can be planned separately.
