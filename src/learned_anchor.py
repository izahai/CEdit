"""Learn and load CLIP-guided adversarial anchors for SPEED.

The categorical prompt optimization adapts the simplex projection and
straight-through token sampling used by UnlearnDiffAtk (MIT License,
Copyright (c) 2023 OPTML Group).  This module is an independent implementation
and never imports from the read-only reference directory.
"""

from __future__ import annotations

import json
import inspect
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler
from tqdm import tqdm


ARTIFACT_SCHEMA_VERSION = 1
ANCHOR_REPRESENTATION = "last_target_token"
DEFAULT_SD_CKPT = "CompVis/stable-diffusion-v1-4"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


@dataclass(frozen=True)
class LearnedAnchorConfig:
    """Configuration for one learned-anchor run."""

    sd_ckpt: str = DEFAULT_SD_CKPT
    clip_model: str = DEFAULT_CLIP_MODEL
    num_prefix_tokens: int = 4
    num_reference_images: int = 4
    num_validation_images: int = 1
    iterations: int = 1000
    learning_rate: float = 1e-2
    weight_decay: float = 0.1
    timestep_min: int = 50
    timestep_max: int = 950
    validation_samples: int = 16
    validation_interval: int = 50
    reference_inference_steps: int = 50
    reference_guidance_scale: float = 7.5
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float16"

    def validate(self, num_train_timesteps: Optional[int] = None) -> None:
        positive_ints = {
            "num_prefix_tokens": self.num_prefix_tokens,
            "num_reference_images": self.num_reference_images,
            "num_validation_images": self.num_validation_images,
            "iterations": self.iterations,
            "validation_samples": self.validation_samples,
            "validation_interval": self.validation_interval,
            "reference_inference_steps": self.reference_inference_steps,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        finite_values = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "reference_guidance_scale": self.reference_guidance_scale,
        }
        for name, value in finite_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.reference_guidance_scale < 0:
            raise ValueError("reference_guidance_scale must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name, value in {
            "timestep_min": self.timestep_min,
            "timestep_max": self.timestep_max,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.timestep_min < 0 or self.timestep_max < self.timestep_min:
            raise ValueError(
                "timestep_min and timestep_max must define a non-negative "
                "inclusive range"
            )
        if (
            num_train_timesteps is not None
            and self.timestep_max >= num_train_timesteps
        ):
            raise ValueError(
                f"timestep_max {self.timestep_max} is outside the scheduler's "
                f"{num_train_timesteps} training timesteps"
            )
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if self.device.startswith("cpu") and self.dtype == "float16":
            raise ValueError("float16 anchor learning is not supported on CPU")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid device {self.device!r}") from error


@dataclass(frozen=True)
class PromptLayout:
    """Fixed-length token layout for a prefixed target prompt."""

    input_ids: torch.Tensor
    target_token_ids: Tuple[int, ...]
    target_token_index: int
    target_was_truncated: bool


@dataclass
class LearnedAnchorBundle:
    """Serializable result returned by :func:`learn_anchors`."""

    manifest: Dict[str, Any]
    tensors: Dict[str, torch.Tensor]
    metrics: List[Dict[str, Any]]
    previews: Dict[str, Image.Image]

    def save(self, save_root: os.PathLike[str] | str) -> None:
        """Atomically publish the tensor, metric, and manifest files."""

        from safetensors.torch import save_file

        root = Path(save_root)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        tensors_path = root / "embeddings.safetensors"
        metrics_path = root / "metrics.jsonl"
        if manifest_path.exists() or tensors_path.exists():
            raise FileExistsError(
                f"A learned-anchor artifact already exists at {root}"
            )

        tensor_tmp = root / ".embeddings.safetensors.tmp"
        metrics_tmp = root / ".metrics.jsonl.tmp"
        manifest_tmp = root / ".manifest.json.tmp"
        serializable_tensors = {
            key: value.detach().contiguous().cpu()
            for key, value in self.tensors.items()
        }
        save_file(serializable_tensors, str(tensor_tmp))
        with metrics_tmp.open("w", encoding="utf-8") as metrics_file:
            for record in self.metrics:
                metrics_file.write(json.dumps(record, allow_nan=False) + "\n")

        previews_root = root / "previews"
        for relative_path, image in self.previews.items():
            output_path = previews_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

        manifest = dict(self.manifest)
        manifest["status"] = "complete"
        with manifest_tmp.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False, allow_nan=False)
            manifest_file.write("\n")

        os.replace(tensor_tmp, tensors_path)
        os.replace(metrics_tmp, metrics_path)
        os.replace(manifest_tmp, manifest_path)


class StraightThroughCategorical(torch.autograd.Function):
    """Sample hard tokens while sending a clipped identity gradient backward."""

    @staticmethod
    def forward(ctx, probabilities, generator=None):
        shape = probabilities.shape
        sampled = torch.multinomial(
            probabilities.reshape(-1, shape[-1]),
            1,
            generator=generator,
        ).reshape(shape[:-1])
        return F.one_hot(sampled, num_classes=shape[-1]).to(probabilities.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output), None


def project_rows_to_simplex(
    values: torch.Tensor,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> torch.Tensor:
    """Project the final dimension onto the capped probability simplex."""

    if values.ndim < 2:
        raise ValueError("Simplex projection expects at least two dimensions")
    flat = values.reshape(-1, values.shape[-1])
    projected = []
    for row in flat:
        clipped = row.clamp(0, 1)
        if abs(clipped.sum().item() - 1.0) <= tolerance:
            result = clipped
        else:
            lower = (row - 1).min().item()
            upper = row.max().item()
            for _ in range(max_iterations):
                midpoint = (lower + upper) / 2
                total = (row - midpoint).clamp(0, 1).sum().item()
                if abs(total - 1.0) <= tolerance:
                    lower = upper = midpoint
                    break
                if total > 1:
                    lower = midpoint
                else:
                    upper = midpoint
            result = (row - (lower + upper) / 2).clamp(0, 1)
        result = result / result.sum().clamp_min(torch.finfo(result.dtype).eps)
        projected.append(result)
    return torch.stack(projected).reshape_as(values)


def initialize_token_distributions(
    num_prefix_tokens: int,
    vocab_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the reference method's uniform categorical initialization."""

    if num_prefix_tokens <= 0 or vocab_size <= 0:
        raise ValueError("num_prefix_tokens and vocab_size must be positive")
    return torch.full(
        (num_prefix_tokens, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )


def normalize_target_concepts(value: str | Sequence[str]) -> List[str]:
    """Normalize target concepts without importing an entry-point module."""

    if isinstance(value, str):
        concepts = value.split(",")
    elif isinstance(value, Sequence):
        concepts = list(value)
    else:
        raise ValueError("target_concepts must be a comma-separated string or list")
    if not concepts or not all(isinstance(item, str) for item in concepts):
        raise ValueError("target_concepts must contain strings")
    concepts = [item.strip() for item in concepts]
    if any(not item for item in concepts):
        raise ValueError("target_concepts must not contain empty values")
    if len(set(concepts)) != len(concepts):
        raise ValueError("target_concepts must not contain duplicates")
    if any(item.casefold() == "nudity" for item in concepts):
        raise ValueError("The first learned-anchor implementation does not support nudity")
    return concepts


def _tokenizer_value(encoded: Any, name: str) -> Any:
    if isinstance(encoded, Mapping):
        return encoded[name]
    return getattr(encoded, name)


def _content_token_ids(tokenizer, target: str) -> List[int]:
    encoded = tokenizer(
        target,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids = _tokenizer_value(encoded, "input_ids")
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_prompt_layout(
    tokenizer,
    target: str,
    prefix_token_ids: Sequence[int],
) -> PromptLayout:
    """Build `[BOS, prefix, target, EOS, padding]` at the model length."""

    max_length = int(tokenizer.model_max_length)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if None in {bos_token_id, eos_token_id, pad_token_id}:
        raise ValueError("Tokenizer must define BOS, EOS, and padding token IDs")
    max_target_tokens = max_length - len(prefix_token_ids) - 2
    if max_target_tokens < 1:
        raise ValueError(
            "The prefix leaves no room for a target token and EOS token"
        )
    content_ids = _content_token_ids(tokenizer, target)
    if not content_ids:
        raise ValueError(f"Target prompt {target!r} produces no content tokens")
    truncated = len(content_ids) > max_target_tokens
    content_ids = content_ids[:max_target_tokens]
    prefix_ids = [int(token_id) for token_id in prefix_token_ids]
    vocab_size = len(tokenizer)
    if any(token_id < 0 or token_id >= vocab_size for token_id in prefix_ids):
        raise ValueError("A prefix token ID is outside the tokenizer vocabulary")
    ids = [bos_token_id] + prefix_ids + content_ids + [eos_token_id]
    ids.extend([pad_token_id] * (max_length - len(ids)))
    target_index = len(prefix_ids) + len(content_ids)
    return PromptLayout(
        input_ids=torch.tensor([ids], dtype=torch.long),
        target_token_ids=tuple(content_ids),
        target_token_index=target_index,
        target_was_truncated=truncated,
    )


def _make_causal_mask(
    input_shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    batch_size, sequence_length = input_shape
    mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    positions = torch.arange(sequence_length, device=device)
    mask.masked_fill_(positions < (positions + 1).view(sequence_length, 1), 0)
    return mask[None, None].expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )


def encode_soft_prompt(
    text_encoder,
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Run CLIP text encoding from caller-supplied token embeddings."""

    text_model = text_encoder.text_model
    hidden_states = text_model.embeddings(
        input_ids=None,
        position_ids=None,
        inputs_embeds=input_embeddings,
    )
    encoder_parameters = inspect.signature(text_model.encoder.forward).parameters
    if "causal_attention_mask" in encoder_parameters:
        causal_mask = _make_causal_mask(
            input_ids.shape,
            hidden_states.dtype,
            hidden_states.device,
        )
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    else:
        from transformers.masking_utils import create_causal_mask

        causal_mask = create_causal_mask(
            config=text_model.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
        )
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=causal_mask,
            is_causal=True,
            return_dict=True,
        )
    return text_model.final_layer_norm(encoder_outputs[0])


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def _module_dtype(module) -> torch.dtype:
    return next(module.parameters()).dtype


def _sampled_prompt_hidden_states(
    text_encoder,
    tokenizer,
    target: str,
    token_distributions: torch.Tensor,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, PromptLayout]:
    num_prefix_tokens, vocab_size = token_distributions.shape
    placeholder_ids = [int(tokenizer.pad_token_id)] * num_prefix_tokens
    layout = build_prompt_layout(tokenizer, target, placeholder_ids)
    input_ids = layout.input_ids.to(_module_device(text_encoder))
    token_embedding = text_encoder.text_model.embeddings.token_embedding
    base_embeddings = token_embedding(input_ids)
    one_hot = StraightThroughCategorical.apply(token_distributions, generator)
    prefix_embeddings = one_hot.to(token_embedding.weight.dtype) @ token_embedding.weight
    input_embeddings = torch.cat(
        [
            base_embeddings[:, :1],
            prefix_embeddings.unsqueeze(0),
            base_embeddings[:, 1 + num_prefix_tokens :],
        ],
        dim=1,
    )
    return encode_soft_prompt(text_encoder, input_ids, input_embeddings), layout


def _hard_prompt_hidden_states(
    text_encoder,
    tokenizer,
    target: str,
    prefix_token_ids: Sequence[int],
) -> Tuple[torch.Tensor, PromptLayout]:
    layout = build_prompt_layout(tokenizer, target, prefix_token_ids)
    input_ids = layout.input_ids.to(_module_device(text_encoder))
    hidden_states = text_encoder(input_ids=input_ids).last_hidden_state
    return hidden_states, layout


def _target_subject_hidden(text_encoder, tokenizer, target: str) -> torch.Tensor:
    device = _module_device(text_encoder)
    inputs = tokenizer(
        target,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = _tokenizer_value(inputs, "input_ids").to(device)
    attention_mask = _tokenizer_value(inputs, "attention_mask").to(device)
    hidden_states = text_encoder(input_ids=input_ids).last_hidden_state
    subject_index = int(attention_mask[0].sum().item()) - 2
    return hidden_states[0, subject_index].unsqueeze(0)


def _pil_to_latent(pipeline, image: Image.Image) -> torch.Tensor:
    vae = pipeline.vae
    image_tensor = pipeline.image_processor.preprocess(image)
    image_tensor = image_tensor.to(_module_device(vae), dtype=_module_dtype(vae))
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.mean
        latent = latent * vae.config.scaling_factor
    return latent.detach()


def _generate_reference_images(
    pipeline,
    target: str,
    seeds: Iterable[int],
    config: LearnedAnchorConfig,
) -> Tuple[List[Image.Image], List[torch.Tensor]]:
    images = []
    latents = []
    device = _module_device(pipeline.unet)
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            output = pipeline(
                prompt=target,
                height=512,
                width=512,
                num_inference_steps=config.reference_inference_steps,
                guidance_scale=config.reference_guidance_scale,
                generator=generator,
            )
        image = output.images[0]
        images.append(image)
        latents.append(_pil_to_latent(pipeline, image))
    return images, latents


def differentiable_clip_preprocess(
    images: torch.Tensor,
    clip_image_processor,
) -> torch.Tensor:
    """Apply the square-image CLIP transform without breaking gradients."""

    crop_size = clip_image_processor.crop_size
    if isinstance(crop_size, Mapping):
        height = int(crop_size.get("height", crop_size.get("shortest_edge")))
        width = int(crop_size.get("width", crop_size.get("shortest_edge")))
    else:
        height = width = int(crop_size)
    resized = F.interpolate(
        images.float(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    mean = torch.tensor(
        clip_image_processor.image_mean,
        device=resized.device,
        dtype=resized.dtype,
    ).view(1, -1, 1, 1)
    std = torch.tensor(
        clip_image_processor.image_std,
        device=resized.device,
        dtype=resized.dtype,
    ).view(1, -1, 1, 1)
    return (resized - mean) / std


def reconstruct_clean_latent(
    noised_latent: torch.Tensor,
    predicted_noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Recover an epsilon-prediction scheduler's clean-latent estimate."""

    output_dtype = noised_latent.dtype
    calculation_dtype = (
        torch.float32
        if noised_latent.dtype in {torch.float16, torch.bfloat16}
        else noised_latent.dtype
    )
    noised_latent = noised_latent.to(calculation_dtype)
    predicted_noise = predicted_noise.to(calculation_dtype)
    alpha = alphas_cumprod.to(
        device=noised_latent.device,
        dtype=calculation_dtype,
    )[timesteps]
    alpha = alpha.flatten()
    while alpha.ndim < noised_latent.ndim:
        alpha = alpha.unsqueeze(-1)
    clean_latent = (
        noised_latent - (1 - alpha).sqrt() * predicted_noise
    ) / alpha.sqrt().clamp_min(torch.finfo(noised_latent.dtype).eps)
    return clean_latent.to(output_dtype)


def _decode_image_tensor(vae, clean_latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(clean_latent / vae.config.scaling_factor).sample
    return (decoded / 2 + 0.5).clamp(0, 1)


def _clip_target_feature(clip_model, clip_processor, target: str) -> torch.Tensor:
    device = _module_device(clip_model)
    encoded = clip_processor(
        text=[target],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        feature = clip_model.get_text_features(**encoded).float()
    return F.normalize(feature, dim=-1)


def _clip_similarity(
    clip_model,
    clip_image_processor,
    image: torch.Tensor,
    target_feature: torch.Tensor,
) -> torch.Tensor:
    pixel_values = differentiable_clip_preprocess(image, clip_image_processor)
    pixel_values = pixel_values.to(dtype=_module_dtype(clip_model))
    image_feature = clip_model.get_image_features(pixel_values=pixel_values).float()
    image_feature = F.normalize(image_feature, dim=-1)
    return (image_feature * target_feature).sum(dim=-1).mean()


def _predict_reconstruction(
    pipeline,
    noise_scheduler,
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    timestep_tensor = torch.tensor(
        [timestep],
        device=clean_latent.device,
        dtype=torch.long,
    )
    noised_latent = noise_scheduler.add_noise(
        clean_latent,
        noise,
        timestep_tensor,
    )
    predicted_noise = pipeline.unet(
        noised_latent,
        timestep_tensor,
        encoder_hidden_states=encoder_hidden_states,
    ).sample
    clean_estimate = reconstruct_clean_latent(
        noised_latent,
        predicted_noise,
        noise_scheduler.alphas_cumprod,
        timestep_tensor,
    )
    return _decode_image_tensor(pipeline.vae, clean_estimate)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    pixels = (
        image[0]
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(pixels)


def _validation_bank(
    latents: Sequence[torch.Tensor],
    config: LearnedAnchorConfig,
    generator: torch.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    bank = []
    device = latents[0].device
    for sample_index in range(config.validation_samples):
        latent = latents[sample_index % len(latents)]
        timestep = int(
            torch.randint(
                config.timestep_min,
                config.timestep_max + 1,
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
        noise = torch.randn(
            latent.shape,
            generator=generator,
            device=device,
            dtype=latent.dtype,
        )
        bank.append((latent, noise, timestep))
    return bank


@torch.no_grad()
def _validate_prefix(
    pipeline,
    tokenizer,
    clip_model,
    clip_image_processor,
    noise_scheduler,
    target: str,
    prefix_token_ids: torch.Tensor,
    target_feature: torch.Tensor,
    validation_bank: Sequence[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[float, Image.Image]:
    hidden_states, _ = _hard_prompt_hidden_states(
        pipeline.text_encoder,
        tokenizer,
        target,
        prefix_token_ids.tolist(),
    )
    scores = []
    preview = None
    for clean_latent, noise, timestep in validation_bank:
        image = _predict_reconstruction(
            pipeline,
            noise_scheduler,
            clean_latent,
            noise,
            timestep,
            hidden_states,
        )
        scores.append(
            float(
                _clip_similarity(
                    clip_model,
                    clip_image_processor,
                    image,
                    target_feature,
                ).item()
            )
        )
        if preview is None:
            preview = _tensor_to_pil(image)
    return sum(scores) / len(scores), preview


def _freeze_module(module) -> None:
    module.eval()
    module.requires_grad_(False)


def _target_tensor_prefix(target_id: str) -> str:
    return f"targets.{target_id}"


def learn_anchors(
    pipeline,
    clip_model,
    clip_processor,
    target_concepts: str | Sequence[str],
    config: LearnedAnchorConfig,
) -> LearnedAnchorBundle:
    """Learn independent adversarial prefix anchors for ordered targets."""

    targets = normalize_target_concepts(target_concepts)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    if noise_scheduler.config.prediction_type != "epsilon":
        raise ValueError("Learned anchors currently require epsilon prediction")
    config.validate(noise_scheduler.config.num_train_timesteps)
    tokenizer = pipeline.tokenizer
    if config.num_prefix_tokens > tokenizer.model_max_length - 3:
        raise ValueError("num_prefix_tokens leaves no room for a target token")
    if len(tokenizer) != pipeline.text_encoder.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary and text-encoder embeddings disagree")

    for module in [pipeline.text_encoder, pipeline.unet, pipeline.vae, clip_model]:
        _freeze_module(module)

    device = _module_device(pipeline.unet)
    configured_device = torch.device(config.device)
    device_mismatch = device.type != configured_device.type or (
        configured_device.index is not None
        and device.index != configured_device.index
    )
    if device_mismatch:
        raise ValueError(
            f"Pipeline is on {device}, but config.device is {config.device}"
        )
    expected_dtype = (
        torch.float16 if config.dtype == "float16" else torch.float32
    )
    for name, module in [
        ("text encoder", pipeline.text_encoder),
        ("U-Net", pipeline.unet),
        ("VAE", pipeline.vae),
        ("CLIP model", clip_model),
    ]:
        if _module_dtype(module) != expected_dtype:
            raise ValueError(
                f"The {name} uses {_module_dtype(module)}, but config.dtype "
                f"is {config.dtype}"
            )
    tensors: Dict[str, torch.Tensor] = {}
    metrics: List[Dict[str, Any]] = []
    previews: Dict[str, Image.Image] = {}
    manifest_targets = []
    vocab_size = len(tokenizer)

    for target_index, target in enumerate(targets):
        target_id = f"target_{target_index:04d}"
        seed_base = config.seed + target_index * 1_000_000
        train_seeds = [seed_base + index for index in range(config.num_reference_images)]
        validation_seeds = [
            seed_base + 100_000 + index
            for index in range(config.num_validation_images)
        ]
        train_images, train_latents = _generate_reference_images(
            pipeline,
            target,
            train_seeds,
            config,
        )
        validation_images, validation_latents = _generate_reference_images(
            pipeline,
            target,
            validation_seeds,
            config,
        )
        for image_index, image in enumerate(train_images):
            previews[f"{target_id}/train_reference_{image_index:03d}.png"] = image
        for image_index, image in enumerate(validation_images):
            previews[f"{target_id}/validation_reference_{image_index:03d}.png"] = image

        optimization_generator = torch.Generator(device=device).manual_seed(
            seed_base + 200_000
        )
        validation_generator = torch.Generator(device=device).manual_seed(
            seed_base + 300_000
        )
        validation_bank = _validation_bank(
            validation_latents,
            config,
            validation_generator,
        )
        target_feature = _clip_target_feature(clip_model, clip_processor, target)
        token_distributions = torch.nn.Parameter(
            initialize_token_distributions(
                config.num_prefix_tokens,
                vocab_size,
                device,
            )
        )
        optimizer = torch.optim.Adam(
            [token_distributions],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and _module_dtype(pipeline.unet) == torch.float16
        )
        best_score = math.inf
        best_step = None
        best_ids = None
        best_distributions = None
        best_preview = None
        recent_training_scores = deque(maxlen=20)

        progress = tqdm(
            range(1, config.iterations + 1),
            desc=f"Learn anchor: {target}",
        )
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            reference_index = int(
                torch.randint(
                    0,
                    len(train_latents),
                    (1,),
                    generator=optimization_generator,
                    device=device,
                ).item()
            )
            clean_latent = train_latents[reference_index]
            timestep = int(
                torch.randint(
                    config.timestep_min,
                    config.timestep_max + 1,
                    (1,),
                    generator=optimization_generator,
                    device=device,
                ).item()
            )
            noise = torch.randn(
                clean_latent.shape,
                generator=optimization_generator,
                device=device,
                dtype=clean_latent.dtype,
            )
            hidden_states, _ = _sampled_prompt_hidden_states(
                pipeline.text_encoder,
                tokenizer,
                target,
                token_distributions,
                optimization_generator,
            )
            reconstructed = _predict_reconstruction(
                pipeline,
                noise_scheduler,
                clean_latent,
                noise,
                timestep,
                hidden_states,
            )
            loss = _clip_similarity(
                clip_model,
                clip_processor.image_processor,
                reconstructed,
                target_feature,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite CLIP loss for target {target!r} at step {step}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if token_distributions.grad is None:
                raise RuntimeError("CLIP loss did not reach the token distributions")
            gradient_norm = float(token_distributions.grad.norm().item())
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                token_distributions.copy_(
                    project_rows_to_simplex(token_distributions)
                )
            training_score = float(loss.detach().item())
            recent_training_scores.append(training_score)
            metrics.append(
                {
                    "type": "train",
                    "target_id": target_id,
                    "step": step,
                    "reference_index": reference_index,
                    "timestep": timestep,
                    "clip_similarity": training_score,
                    "clip_similarity_moving_average": (
                        sum(recent_training_scores) / len(recent_training_scores)
                    ),
                    "gradient_norm": gradient_norm,
                }
            )
            progress.set_postfix(clip=f"{training_score:.4f}")

            if step % config.validation_interval == 0 or step == config.iterations:
                candidate_ids = token_distributions.detach().argmax(dim=-1)
                validation_score, validation_preview = _validate_prefix(
                    pipeline,
                    tokenizer,
                    clip_model,
                    clip_processor.image_processor,
                    noise_scheduler,
                    target,
                    candidate_ids,
                    target_feature,
                    validation_bank,
                )
                metrics.append(
                    {
                        "type": "validation",
                        "target_id": target_id,
                        "step": step,
                        "clip_similarity": validation_score,
                        "prefix_token_ids": candidate_ids.tolist(),
                    }
                )
                if validation_score < best_score:
                    best_score = validation_score
                    best_step = step
                    best_ids = candidate_ids.detach().cpu().clone()
                    best_distributions = token_distributions.detach().cpu().clone()
                    best_preview = validation_preview

        if best_ids is None or best_distributions is None:
            raise RuntimeError(f"No validation candidate was produced for {target!r}")
        hard_hidden, hard_layout = _hard_prompt_hidden_states(
            pipeline.text_encoder,
            tokenizer,
            target,
            best_ids.tolist(),
        )
        anchor_hidden = hard_hidden[0, hard_layout.target_token_index].unsqueeze(0)
        target_hidden = _target_subject_hidden(
            pipeline.text_encoder,
            tokenizer,
            target,
        )
        tensor_prefix = _target_tensor_prefix(target_id)
        tensors[f"{tensor_prefix}.token_distributions"] = best_distributions
        tensors[f"{tensor_prefix}.prefix_token_ids"] = best_ids
        tensors[f"{tensor_prefix}.anchor_hidden_state"] = anchor_hidden.float().cpu()
        tensors[f"{tensor_prefix}.target_hidden_state"] = target_hidden.float().cpu()
        previews[f"{target_id}/best_validation_reconstruction.png"] = best_preview
        manifest_targets.append(
            {
                "target_id": target_id,
                "target": target,
                "best_step": best_step,
                "best_validation_clip_similarity": best_score,
                "prefix_token_ids": best_ids.tolist(),
                "decoded_prefix": tokenizer.decode(best_ids.tolist()),
                "target_token_index": hard_layout.target_token_index,
                "target_was_truncated": hard_layout.target_was_truncated,
                "training_reference_seeds": train_seeds,
                "validation_reference_seeds": validation_seeds,
                "optimization_seed": seed_base + 200_000,
                "validation_seed": seed_base + 300_000,
            }
        )

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "sd_ckpt": config.sd_ckpt,
        "clip_model": config.clip_model,
        "anchor_representation": ANCHOR_REPRESENTATION,
        "anchor_parameterization": "categorical_simplex_straight_through",
        "selection": "best_validated_argmax",
        "scheduler_prediction_type": noise_scheduler.config.prediction_type,
        "target_order": targets,
        "tokenizer": {
            "vocab_size": vocab_size,
            "model_max_length": int(tokenizer.model_max_length),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "text_hidden_size": int(pipeline.text_encoder.config.hidden_size),
        "config": asdict(config),
        "targets": manifest_targets,
    }
    return LearnedAnchorBundle(
        manifest=manifest,
        tensors=tensors,
        metrics=metrics,
        previews=previews,
    )


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Learned-anchor tensor {name!r} contains non-finite values")


@torch.no_grad()
def load_learned_anchors(
    artifact_path: os.PathLike[str] | str,
    target_concepts: str | Sequence[str],
    pipeline,
    expected_sd_ckpt: Optional[str] = None,
) -> List[torch.Tensor]:
    """Load, validate, and re-encode learned discrete anchor prompts."""

    from safetensors.torch import load_file

    targets = normalize_target_concepts(target_concepts)
    root = Path(artifact_path)
    manifest_path = root / "manifest.json"
    tensors_path = root / "embeddings.safetensors"
    if not manifest_path.is_file() or not tensors_path.is_file():
        raise ValueError(f"Incomplete learned-anchor artifact: {root}")
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("status") != "complete":
        raise ValueError("Learned-anchor artifact is not complete")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported learned-anchor schema {manifest.get('schema_version')!r}"
        )
    if manifest.get("anchor_representation") != ANCHOR_REPRESENTATION:
        raise ValueError("Learned-anchor representation is incompatible with SPEED")
    if manifest.get("scheduler_prediction_type") != "epsilon":
        raise ValueError("Learned-anchor artifact does not use epsilon prediction")
    if manifest.get("target_order") != targets:
        raise ValueError(
            "Learned-anchor target order does not exactly match target_concepts"
        )
    if expected_sd_ckpt is not None and manifest.get("sd_ckpt") != expected_sd_ckpt:
        raise ValueError(
            f"Learned anchor was built for {manifest.get('sd_ckpt')!r}, "
            f"not {expected_sd_ckpt!r}"
        )
    tokenizer = pipeline.tokenizer
    tokenizer_metadata = manifest.get("tokenizer", {})
    expected_tokenizer = {
        "vocab_size": len(tokenizer),
        "model_max_length": int(tokenizer.model_max_length),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer_metadata != expected_tokenizer:
        raise ValueError("Learned-anchor tokenizer metadata does not match the model")
    if manifest.get("text_hidden_size") != pipeline.text_encoder.config.hidden_size:
        raise ValueError("Learned-anchor hidden size does not match the text encoder")
    entries = manifest.get("targets")
    if not isinstance(entries, list) or len(entries) != len(targets):
        raise ValueError("Learned-anchor manifest has an invalid target list")
    if [entry.get("target") for entry in entries] != targets:
        raise ValueError("Learned-anchor manifest target entries are inconsistent")
    config_metadata = manifest.get("config")
    if not isinstance(config_metadata, Mapping):
        raise ValueError("Learned-anchor manifest has invalid configuration metadata")
    expected_prefix_count = config_metadata.get("num_prefix_tokens")
    if (
        isinstance(expected_prefix_count, bool)
        or not isinstance(expected_prefix_count, int)
        or expected_prefix_count <= 0
    ):
        raise ValueError("Learned-anchor manifest has an invalid prefix-token count")

    target_ids = [entry.get("target_id") for entry in entries]
    if (
        any(not isinstance(target_id, str) or not target_id for target_id in target_ids)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise ValueError("Learned-anchor manifest has invalid target identifiers")

    try:
        tensors = load_file(str(tensors_path), device="cpu")
    except Exception as error:
        raise ValueError(f"Could not read learned-anchor tensors: {error}") from error
    resolved = []
    vocab_size = len(tokenizer)
    hidden_size = pipeline.text_encoder.config.hidden_size
    comparison_tolerance = (
        (1e-3, 1e-3)
        if _module_dtype(pipeline.text_encoder) == torch.float16
        else (1e-5, 1e-4)
    )
    for entry, target in zip(entries, targets):
        target_id = entry.get("target_id")
        tensor_prefix = _target_tensor_prefix(target_id)
        required_keys = {
            "distributions": f"{tensor_prefix}.token_distributions",
            "ids": f"{tensor_prefix}.prefix_token_ids",
            "anchor": f"{tensor_prefix}.anchor_hidden_state",
            "target": f"{tensor_prefix}.target_hidden_state",
        }
        missing = [key for key in required_keys.values() if key not in tensors]
        if missing:
            raise ValueError(f"Learned-anchor tensors are missing: {missing}")
        distributions = tensors[required_keys["distributions"]].float()
        prefix_ids = tensors[required_keys["ids"]].long()
        saved_anchor = tensors[required_keys["anchor"]].float()
        saved_target = tensors[required_keys["target"]].float()
        for name, tensor in [
            (required_keys["distributions"], distributions),
            (required_keys["anchor"], saved_anchor),
            (required_keys["target"], saved_target),
        ]:
            _require_finite(name, tensor)
        if distributions.shape != (expected_prefix_count, vocab_size):
            raise ValueError(f"Invalid token distribution shape for {target!r}")
        if prefix_ids.shape != (expected_prefix_count,):
            raise ValueError(f"Invalid prefix token shape for {target!r}")
        if saved_anchor.shape != (1, hidden_size) or saved_target.shape != (
            1,
            hidden_size,
        ):
            raise ValueError(f"Invalid contextual hidden-state shape for {target!r}")
        if (distributions < 0).any() or not torch.allclose(
            distributions.sum(dim=-1),
            torch.ones(expected_prefix_count),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(f"Invalid simplex distributions for {target!r}")
        if not torch.equal(distributions.argmax(dim=-1), prefix_ids):
            raise ValueError(f"Prefix IDs are not the distribution argmax for {target!r}")
        if prefix_ids.tolist() != entry.get("prefix_token_ids"):
            raise ValueError(f"Manifest prefix IDs disagree for {target!r}")

        hard_hidden, layout = _hard_prompt_hidden_states(
            pipeline.text_encoder,
            tokenizer,
            target,
            prefix_ids.tolist(),
        )
        recomputed_anchor = hard_hidden[0, layout.target_token_index].unsqueeze(0)
        recomputed_target = _target_subject_hidden(
            pipeline.text_encoder,
            tokenizer,
            target,
        )
        atol, rtol = comparison_tolerance
        if not torch.allclose(
            recomputed_anchor.float().cpu(),
            saved_anchor,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(f"Recomputed learned anchor differs for {target!r}")
        if not torch.allclose(
            recomputed_target.float().cpu(),
            saved_target,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(f"Recomputed target embedding differs for {target!r}")
        resolved.append(recomputed_anchor)
    return resolved
