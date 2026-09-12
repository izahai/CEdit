# CLIP-Guided Adversarial Anchor Prompt for SPEED

## Status

This document records the design, implementation plan, and usage of the
experimental implementation now present in this repository. Its CPU-level
logic and integration tests pass, but the method has not yet been validated by
full GPU erasure and retention experiments.

The prompt-optimization mechanism is inspired by the adversarial prompt search
in `Diffusion-MU-Attack-main/`. That directory is a read-only reference. The
adapted implementation lives in this repository's normal source, test, and
configuration directories and does not modify or directly depend on the
reference folder.

## Motivation

SPEED erases a target concept by redirecting its text representation toward an
anchor representation. Existing anchors are normally selected manually, for
example the empty prompt, `person`, or `art`. Such anchors are generic, but they
are not optimized for the particular target concept or for the behavior of the
diffusion model.

The proposed method learns a target-specific anchor prompt before applying the
normal SPEED edit. The learned anchor is optimized so that, when it conditions
the diffusion model on a noised target image, the reconstructed image has low
CLIP similarity with the original target text. The hypothesis is that an anchor
that is strongly separated from the target in the model's image semantics will
produce a more effective target-to-anchor erasure direction.

The method has two stages:

1. learn a CLIP-guided adversarial anchor for each target; and
2. use the learned representation as the anchor in the existing SPEED edit.

## Notation

Let:

- $p_t$ be the original target prompt, such as `Snoopy`;
- $q_1,\ldots,q_k$ be $k$ learnable categorical distributions over the
  tokenizer vocabulary;
- $z_1,\ldots,z_k$ be hard tokens sampled from those distributions with a
  straight-through gradient estimator;
- $p_a=[z_1,\ldots,z_k]+p_t$ be the sampled anchor prompt;
- $E$ be the frozen Stable Diffusion text encoder;
- $\epsilon_\theta$ be the frozen diffusion U-Net;
- $D$ be the frozen VAE decoder;
- $C_I$ and $C_T$ be the frozen CLIP image and text encoders;
- $x_0$ be the latent of an image generated from the target prompt;
- $t$ be a sampled diffusion training timestep; and
- $\epsilon\sim\mathcal N(0,I)$ be sampled Gaussian noise.

The learned parameters are initialized as uniform vocabulary distributions and
projected back onto the probability simplex after each optimizer step. The
target tokens and all model weights remain frozen during anchor learning.

## Stage 1: Learn the adversarial anchor

### 1. Construct the sampled discrete prompt

Prepend $k$ sampled vocabulary tokens to the normal target prompt:

$$
p_a=[z_1,z_2,\ldots,z_k]+p_t,
\qquad z_i\sim\operatorname{Categorical}(q_i).
$$

The prefix is inserted after the start-of-text token and before the first
target token. Each sampled token is represented as a hard one-hot vector in the
forward pass. A straight-through estimator passes clipped gradients back to
the categorical distribution. Each $q_i$ starts uniform over the vocabulary,
and a capped-simplex projection preserves a valid distribution after every
Adam update.

### 2. Generate a target reference image

Use the original, unedited Stable Diffusion model and target prompt $p_t$ to
generate a reference image $I_t$. Encode it with the VAE to obtain the clean
latent $x_0$.

The reference image supplies an image-space realization of the target concept.
It is not the second input to the CLIP loss. Multiple generation seeds or
multiple target prompts can be used to prevent the learned anchor from
overfitting to one image.

### 3. Sample a noisy target latent

At each optimization iteration, sample a timestep $t$ and noise $\epsilon$,
then apply the forward diffusion process:

$$
x_t=
\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

Sampling many random timesteps exposes the anchor to target information at
different noise levels. The timestep distribution and any excluded extreme
timesteps must be recorded in the experiment configuration.

### 4. Predict noise using the learned anchor

Condition the frozen U-Net on the sampled anchor prompt:

$$
h_a=E(p_a),
$$

$$
\hat\epsilon_a=\epsilon_\theta(x_t,t,h_a).
$$

Unlike ordinary diffusion training, the loss is not used to update the U-Net.
Its gradient passes through the U-Net and text encoder into the learnable
categorical prefix distributions.

### 5. Reconstruct the anchor-conditioned clean image

Convert the predicted noise into an estimate of the clean target latent:

$$
\hat x_{0,a}=
\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_a}
     {\sqrt{\bar\alpha_t}}.
$$

Decode that estimate through the VAE:

$$
I_a=D(\hat x_{0,a}).
$$

Here $I_t$ is the fixed target-generated reference image and $I_a$ is the
anchor-conditioned reconstruction. The required CLIP objective is an
image-to-text comparison between $I_a$ and $p_t$; it is not an image-to-image
CLIP comparison between $I_a$ and $I_t$.

### 6. Minimize target CLIP similarity

Define the target-alignment score as cosine similarity in CLIP space:

$$
s_{\mathrm{CLIP}}(I_a,p_t)=
\frac{C_I(I_a)^\top C_T(p_t)}
     {\lVert C_I(I_a)\rVert_2\lVert C_T(p_t)\rVert_2}.
$$

Use the score itself as the loss and minimize it:

$$
\mathcal L_{\mathrm{anchor}}=
s_{\mathrm{CLIP}}(I_a,p_t).
$$

This objective pushes the anchor-conditioned reconstruction away from the
semantic meaning of the original target prompt. Optimization repeats over
random timesteps, noises, and target-image seeds until the anchor converges or
a configured iteration limit is reached.

The raw score, moving-average score, prefix gradient norm, and best checkpoint
should be logged. Anchor selection should use the best validation score across
held-out noise and image seeds instead of only the final training iteration.

## Stage 2: Use the learned anchor in SPEED

During validation, replace every distribution with its argmax token. Retain
the earliest validation candidate with the lowest mean held-out CLIP score.
After optimization, run that discrete prompt through the same frozen text encoder
used by SPEED and extract an anchor representation with the same shape and
token-position convention as the target representation:

$$
e_t=E(p_t),
\qquad
e_a^\star=E(p_a^\star).
$$

For the current non-nudity SPEED path, this normally means selecting the
contextual representation at the final subject-token position. Although that
token still belongs to the original target phrase, its contextual hidden state
is changed by the learned prefix. If a future mode uses the full conditioning
sequence, both target and anchor must use the full sequence consistently.

Replace the manually encoded anchor with the learned representation. In the
legacy target-anchor construction, the residual is

$$
r=e_a^\star-e_t.
$$

The remainder of SPEED is unchanged: construct the target-anchor statistics,
apply retain-aware projection, and perform the normal closed-form edits to the
selected cross-attention weights.

The expected effect is stronger suppression because the target is redirected
toward an anchor chosen specifically to have low image-semantic alignment with
that target. This is a hypothesis to test; a larger semantic separation may
also cause excessive model damage, so retention metrics remain essential.

## Relationship to UnlearnDiffAtk

The reference implementation learns adversarial prompt tokens to make an
already-unlearned diffusion model recover an erased concept. Its differentiable
task loss measures how well the adversarially conditioned model predicts noise
for a noised target image. In that setting, accurate target-image reconstruction
helps attack the unlearned model.

This proposal adapts the mechanism but reverses its purpose:

| Aspect | UnlearnDiffAtk reference | Proposed learned anchor |
|---|---|---|
| Goal | Recover an erased concept | Construct an anchor for erasing a concept |
| Model being optimized through | An already-unlearned model | The original frozen model before SPEED editing |
| Prompt parameters | Adversarial inserted tokens | Learnable prefix anchor tokens |
| Main objective | Noise-prediction or attack-task loss | Low CLIP similarity to the target text |
| Result | An attack prompt | An anchor representation consumed by SPEED |

## Anchor artifact and integration

The primary artifact contains the learned vocabulary distributions, the exact
best-validated argmax token IDs, and the derived contextual anchor
representation. Exact token IDs are retained because decoding and retokenizing
a displayed prefix is not guaranteed to round-trip. Each saved artifact should
contain at least:

- target prompt and its token IDs;
- learned vocabulary-distribution tensor and argmax prefix token IDs;
- derived anchor hidden state used by SPEED;
- number and placement of prefix tokens;
- base model and CLIP model identifiers;
- optimization hyperparameters;
- random seeds and timestep-sampling configuration; and
- best training and validation CLIP scores.

The existing `--anchor_concepts` string interface cannot safely represent this
artifact because decoded tokens may not retokenize identically and the model
compatibility metadata would be lost. Integration therefore uses a separate
learned-anchor source that loads and validates exact token IDs before building
SPEED's target-anchor statistics.

## Initial validation plan

The first experiment should use one instance target, such as `Snoopy`, and
compare:

1. original Stable Diffusion without erasure;
2. legacy SPEED with the empty anchor;
3. SPEED with the best-validated learned discrete anchor.

Anchor learning should be evaluated before editing with held-out target images,
noise samples, and timesteps. The edited checkpoints should then be evaluated
for both target erasure and retention. At minimum, record:

- CLIP similarity of $I_a$ to the target text during anchor search;
- CLIP similarity and target detection rate after erasure;
- qualitative generations for the erased target;
- retained-concept CLIP score or accuracy;
- MS-COCO CLIP score and FID when running the full evaluation; and
- the norm and direction of $e_a^\star-e_t$ relative to legacy anchors.

The central claim is supported only if the learned anchor improves target
erasure over the legacy anchor without an unacceptable loss of retained
concepts or general generation quality.

## Open experimental decisions

The following choices should be controlled through configuration and ablated:

- number $k$ of learned prefix tokens;
- alternative continuous soft-token parameterizations;
- one target reference image versus multiple images and prompt templates;
- timestep sampling distribution;
- whether to include classifier-free guidance in the differentiable path;
- one-step clean-latent reconstruction versus differentiating through several
  denoising steps;
- raw cosine minimization versus a bounded margin objective;
- optional regularization on prefix norm or distance from vocabulary
  embeddings; and
- convergence and best-checkpoint selection criteria.

The first version uses uniform categorical prefix distributions,
straight-through sampled vocabulary tokens, best-validated argmax selection,
one-step clean-latent reconstruction, uniformly sampled non-extreme timesteps,
a frozen base Stable Diffusion v1.4 model, and the raw CLIP cosine objective.

## Usage API

The implementation exposes anchor learning and SPEED editing as two
separate commands. Anchor search is iterative and expensive, whereas SPEED
editing is fast. Saving the anchor as an intermediate artifact makes the search
reproducible and allows the same learned anchor to be reused in multiple SPEED
experiments.

### Learn an anchor

For one target:

```bash
CUDA_VISIBLE_DEVICES=0 python learn_anchor.py \
    --target_concepts "Snoopy" \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --num_prefix_tokens 4 \
    --num_reference_images 4 \
    --num_validation_images 1 \
    --iterations 1000 \
    --learning_rate 0.01 \
    --weight_decay 0.1 \
    --timestep_min 50 \
    --timestep_max 950 \
    --validation_samples 16 \
    --validation_interval 50 \
    --seed 0 \
    --save_root "logs/learned_anchors/snoopy"
```

For multiple targets, learn one anchor per target and store them in one
artifact:

```bash
CUDA_VISIBLE_DEVICES=0 python learn_anchor.py \
    --target_concepts "Snoopy,Mickey,Spongebob" \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --num_prefix_tokens 4 \
    --num_reference_images 4 \
    --num_validation_images 1 \
    --iterations 1000 \
    --learning_rate 0.01 \
    --weight_decay 0.1 \
    --timestep_min 50 \
    --timestep_max 950 \
    --validation_samples 16 \
    --validation_interval 50 \
    --seed 0 \
    --save_root "logs/learned_anchors/cartoon_characters"
```

The proposed anchor-learning arguments are:

| Argument | Meaning | Initial default |
|---|---|---:|
| `--target_concepts` | Comma-separated targets, following the existing SPEED interface | Required |
| `--sd_ckpt` | Frozen Stable Diffusion checkpoint used for reference generation and optimization | `CompVis/stable-diffusion-v1-4` |
| `--num_prefix_tokens` | Number $k$ of categorical prefix-token distributions | `4` |
| `--num_reference_images` | Target-generated reference images per concept | `4` |
| `--num_validation_images` | Separately seeded validation reference images per concept | `1` |
| `--iterations` | Optimization iterations per target | `1000` |
| `--learning_rate` | Categorical-distribution learning rate | `0.01` |
| `--timestep_min` | Lowest sampled diffusion training timestep, inclusive | `50` |
| `--timestep_max` | Highest sampled diffusion training timestep, inclusive | `950` |
| `--validation_samples` | Fixed held-out noise/timestep samples used to select the best argmax anchor | `16` |
| `--validation_interval` | Iterations between deterministic argmax validation passes | `50` |
| `--weight_decay` | Adam weight decay applied before simplex projection | `0.1` |
| `--clip_model` | CLIP checkpoint used by the image-to-text objective | `openai/clip-vit-large-patch14` |
| `--reference_inference_steps` | Denoising steps used to generate each reference image | `50` |
| `--reference_guidance_scale` | Classifier-free guidance used only for reference generation | `7.5` |
| `--seed` | Seed for initialization, reference generation, noise, and timestep sampling | `0` |
| `--device` | Torch device used for optimization | `cuda` |
| `--dtype` | Model dtype (`float16` or `float32`) | `float16` |
| `--save_root` | Output directory for the learned-anchor artifact and diagnostics | Required |

`timestep_min` must not exceed `timestep_max`, both must be valid training
timesteps for the scheduler, and the target prompt plus prefix must fit within
the tokenizer's maximum sequence length.

### Learn an anchor from YAML

The new entry point supports the same YAML-plus-CLI pattern as
`train_erase_null.py`. For example:

```yaml
# configs/learned_anchor/snoopy.yaml
sd_ckpt: CompVis/stable-diffusion-v1-4
target_concepts:
  - Snoopy
num_prefix_tokens: 4
num_reference_images: 4
num_validation_images: 1
iterations: 1000
learning_rate: 0.01
weight_decay: 0.1
timestep_min: 50
timestep_max: 950
validation_samples: 16
validation_interval: 50
reference_inference_steps: 50
reference_guidance_scale: 7.5
seed: 0
device: cuda
dtype: float16
save_root: logs/learned_anchors/snoopy
```

```bash
CUDA_VISIBLE_DEVICES=0 python learn_anchor.py \
    --config "configs/learned_anchor/snoopy.yaml"
```

Explicit CLI arguments override values loaded from YAML.

### Apply the learned anchor with SPEED

Anchor origin and residual construction are separate decisions. Use
`--anchor_source` to select how anchor embeddings are obtained while preserving
the existing meaning of `--anchor_mode`:

```bash
CUDA_VISIBLE_DEVICES=0 python train_erase_null.py \
    --target_concepts "Snoopy" \
    --anchor_source learned \
    --learned_anchor_path "logs/learned_anchors/snoopy" \
    --anchor_mode legacy \
    --retain_path "data/instance.csv" \
    --heads concept \
    --save_path "logs/checkpoints/snoopy_learned_anchor" \
    --file_name "weight"
```

The source options should be:

| `--anchor_source` | Behavior |
|---|---|
| `text` | Encode `--anchor_concepts` exactly as the current implementation does |
| `learned` | Load and validate best-argmax anchor representations from `--learned_anchor_path` |

`text` remains the default so every existing command continues to work without
modification. The first implementation supports `learned` only with
`--anchor_mode legacy`, non-nudity targets, and `--erase_style` disabled.
Additional representations and residual modes can be enabled after the basic
method has been validated.

The parser should reject ambiguous or incomplete combinations:

- `--anchor_source text` requires `--anchor_concepts`;
- `--anchor_source learned` requires `--learned_anchor_path`;
- `--anchor_source learned` cannot be combined with `--anchor_concepts`; and
- the learned artifact must contain exactly one compatible entry for every
  requested target concept.

An equivalent SPEED YAML configuration is:

```yaml
sd_ckpt: CompVis/stable-diffusion-v1-4
target_concepts:
  - Snoopy
anchor_source: learned
learned_anchor_path: logs/learned_anchors/snoopy
anchor_mode: legacy
retain_path: data/instance.csv
heads: concept
save_path: logs/checkpoints/snoopy_learned_anchor
file_name: weight
```

### Artifact layout

Use a directory rather than one opaque pickle file:

```text
logs/learned_anchors/snoopy/
├── manifest.json
├── embeddings.safetensors
├── metrics.jsonl
└── previews/
```

The files have distinct responsibilities:

- `manifest.json` records target ordering, model identifiers, representation
  conventions, hyperparameters, random seeds, and best validation scores;
- `embeddings.safetensors` stores vocabulary distributions, exact argmax token
  IDs, and contextual anchor representations consumed by SPEED;
- `metrics.jsonl` records per-iteration training and validation diagnostics;
  and
- `previews/` contains optional reference and anchor-reconstruction images for
  qualitative inspection.

Recommended tensor keys are namespaced by a stable target identifier:

```text
targets.<target_id>.token_distributions
targets.<target_id>.prefix_token_ids
targets.<target_id>.anchor_hidden_state
targets.<target_id>.target_hidden_state
```

The manifest maps each `target_id` to the exact target string, avoiding unsafe
or ambiguous tensor keys derived directly from arbitrary prompt text.

Before editing, the loader must validate that:

1. every requested target is present exactly once;
2. target spelling and ordering are unambiguous;
3. tensor dimensions match the active text encoder;
4. tokenizer, text encoder, and base-model identifiers match;
5. all loaded tensor values are finite; and
6. the saved token-position convention matches the representation expected by
   SPEED.

### Internal Python API

Keep the reusable implementation in `src/` and make both entry points thin CLI
wrappers. A minimal interface is:

```python
from src.learned_anchor import (
    LearnedAnchorConfig,
    learn_anchors,
    load_learned_anchors,
)

config = LearnedAnchorConfig(
    num_prefix_tokens=4,
    num_reference_images=4,
    iterations=1000,
    learning_rate=0.01,
    timestep_min=50,
    timestep_max=950,
    validation_samples=16,
    seed=0,
)

bundle = learn_anchors(
    pipeline=pipeline,
    clip_model=clip_model,
    clip_processor=clip_processor,
    target_concepts=["Snoopy"],
    config=config,
)
bundle.save("logs/learned_anchors/snoopy")

anchor_embeddings = load_learned_anchors(
    "logs/learned_anchors/snoopy",
    target_concepts=["Snoopy"],
    pipeline=pipeline,
)
```

`learn_anchors` owns prompt construction, target-image caching, timestep
sampling, optimization, validation, and best-anchor selection. Callers do not
need to coordinate individual diffusion components. The loader owns artifact
validation and returns anchor tensors already ordered to match
`target_concepts`.

## Implementation plan and status

The implementation is organized into independently testable layers:

1. **Prompt optimization core — implemented.** `src/learned_anchor.py`
   provides categorical initialization, straight-through hard token sampling,
   simplex projection, prefixed CLIP-text encoding, noising, one-step clean
   latent reconstruction, differentiable CLIP preprocessing, validation, and
   best-candidate selection. Stable Diffusion, its scheduler components, and
   CLIP remain frozen.
2. **Artifact boundary — implemented.** The same module writes an atomic,
   versioned artifact and validates target order, model/tokenizer metadata,
   tensor shapes, finite values, simplex constraints, and saved argmax IDs when
   loading. The discrete prompt is re-encoded and checked against the saved
   contextual hidden state before SPEED receives it.
3. **Learning command — implemented.** `learn_anchor.py` provides YAML plus CLI
   configuration, validates the first-version constraints, loads SD v1.4 and
   CLIP ViT-L/14, learns one independent anchor per ordered target, and writes
   metrics and previews. `configs/learned_anchor/snoopy.yaml` is the initial
   experiment configuration.
4. **SPEED integration — implemented.** `train_erase_null.py` accepts
   `--anchor_source learned` and resolves the loaded contextual hidden state at
   the same seam used by text anchors. Existing text-anchor behavior remains
   the default. `configs/train_learned_anchor.yaml` demonstrates the edit
   stage.
5. **CPU/unit verification — implemented.** Tests cover simplex invariants,
   straight-through gradients, prompt placement and truncation, custom text
   encoding parity, exact $x_0$ reconstruction, differentiable CLIP image
   preprocessing, artifact round trips and rejection paths, CLI configuration,
   and SPEED anchor resolution.
6. **GPU experimental validation — pending.** Run the Snoopy anchor search,
   inspect its held-out CLIP trajectory and previews, produce the learned-anchor
   SPEED checkpoint, then compare erasure and retention against the unchanged
   model and the legacy null-anchor baseline. This is required before treating
   the method as empirically validated.
