"""Learn a continuous anchor through the exact legacy SPEED value edit."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from src.closed_form_anchor_training import (
    AnchorTrainingConfig,
    BoundedAnchor,
    optimize_anchor,
    paired_predictions,
    sample_prefix_diffusion_state,
)
from src.differentiable_legacy_edit import (
    SPEED_CHECKPOINT_FORMAT,
    DifferentiableLegacyEditConfig,
    prepare_differentiable_legacy_edit,
)
from src.edit_checkpoint import load_edit_checkpoint, save_speed_checkpoint
from src.utils import seed_everything

ANCHOR_ARTIFACT_FORMAT = "speed-anchor-optimization-v2"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a bounded anchor through differentiable legacy SPEED",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--target_concepts", type=str, default=None)
    parser.add_argument("--anchor_concepts", type=str, default=None)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="SPEED")
    parser.add_argument("--params", type=str, default="V")
    parser.add_argument("--anchor_mode", type=str, default="legacy")
    parser.add_argument("--aug_num", type=int, default=0)
    parser.add_argument("--erase_style", action="store_true", default=False)

    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--retain_scale", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.0)

    parser.add_argument("--anchor_steps", type=int, default=200)
    parser.add_argument("--anchor_lr", type=float, default=1e-2)
    parser.add_argument("--anchor_batch_size", type=int, default=1)
    parser.add_argument(
        "--max_anchor_norm",
        type=str,
        default="target",
        help="Use each target embedding norm as the fixed anchor-norm cap",
    )
    parser.add_argument("--cosine_eps", type=float, default=1e-8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--validation_seed", type=int, default=10000)
    parser.add_argument("--validation_samples", type=int, default=16)
    parser.add_argument("--validation_interval", type=int, default=10)
    parser.add_argument("--optimization_prompts_path", type=str, default=None)
    parser.add_argument("--validation_prompts_path", type=str, default=None)
    return parser


def load_yaml_config(
    config_path: str | os.PathLike[str],
    parser: argparse.ArgumentParser,
) -> dict:
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        parser.error("YAML config must contain a mapping")
    valid_keys = {
        action.dest for action in parser._actions if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        parser.error("Unknown YAML config key(s): " + ", ".join(unknown))
    return {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in config.items()
    }


def parse_args(argv=None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_argument_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        parser.set_defaults(**load_yaml_config(config_args.config, parser))
    args = parser.parse_args(argv)
    return parser, args


def normalize_concepts(value, name: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, str):
        concepts = value.split(",")
    elif isinstance(value, list):
        concepts = value
    else:
        raise TypeError(f"{name} must be a comma-separated string or YAML list")
    if not concepts or not all(isinstance(value, str) for value in concepts):
        raise ValueError(f"{name} must contain strings")
    normalized = [value.strip() for value in concepts]
    if not allow_empty and any(not value for value in normalized):
        raise ValueError(f"{name} cannot contain empty concepts")
    return normalized


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    required = {
        "target_concepts": args.target_concepts,
        "anchor_concepts": args.anchor_concepts,
        "retain_path": args.retain_path,
        "heads": args.heads,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("Missing required option(s): " + ", ".join(missing))
    try:
        targets = normalize_concepts(args.target_concepts, "target_concepts")
        anchors = normalize_concepts(
            args.anchor_concepts,
            "anchor_concepts",
            allow_empty=True,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if len(targets) != 1:
        parser.error("The prototype supports exactly one target concept")
    if len(set(targets)) != len(targets):
        parser.error("target_concepts contains duplicates")
    if len(anchors) == 1:
        anchors *= len(targets)
    if len(anchors) != len(targets):
        parser.error("anchor_concepts must contain one anchor per target")
    if not isinstance(args.erase_style, bool):
        parser.error("--erase_style must be a boolean")
    if not isinstance(args.heads, str) or not args.heads.strip():
        parser.error("--heads must be a non-empty string")
    if not isinstance(args.device, str) or not args.device.strip():
        parser.error("--device must be a non-empty string")
    if not isinstance(args.sd_ckpt, str) or not args.sd_ckpt.strip():
        parser.error("--sd_ckpt must be a non-empty string")
    if not isinstance(args.file_name, str) or not args.file_name or "/" in args.file_name:
        parser.error("--file_name must be a non-empty basename")
    if args.save_path is not None and not isinstance(args.save_path, str):
        parser.error("--save_path must be a string")

    expected = {
        "params": (args.params, "V"),
        "anchor_mode": (args.anchor_mode, "legacy"),
        "baseline": (args.baseline, "SPEED"),
        "aug_num": (args.aug_num, 0),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            parser.error(f"--{name} must be {wanted!r} for this prototype")
    if args.max_anchor_norm != "target":
        parser.error("--max_anchor_norm must be 'target'")
    positive = {
        "retain_scale": args.retain_scale,
        "residual_scale": args.residual_scale,
        "anchor_lr": args.anchor_lr,
        "cosine_eps": args.cosine_eps,
        "guidance_scale": args.guidance_scale,
    }
    for name, value in positive.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            parser.error(f"--{name} must be positive and finite")
    if (
        isinstance(args.threshold, bool)
        or not isinstance(args.threshold, (int, float))
        or not math.isfinite(args.threshold)
    ):
        parser.error("--threshold must be finite")
    if (
        isinstance(args.lamb, bool)
        or not isinstance(args.lamb, (int, float))
        or not math.isfinite(args.lamb)
        or args.lamb < 0
    ):
        parser.error("--lamb must be nonnegative and finite")
    integer_positive = {
        "anchor_steps": args.anchor_steps,
        "anchor_batch_size": args.anchor_batch_size,
        "num_inference_steps": args.num_inference_steps,
        "resolution": args.resolution,
        "validation_samples": args.validation_samples,
        "validation_interval": args.validation_interval,
    }
    for name, value in integer_positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            parser.error(f"--{name} must be a positive integer")
    for name in ("seed", "validation_seed"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            parser.error(f"--{name} must be a nonnegative integer")
    if args.validation_seed == args.seed:
        parser.error("--validation_seed must differ from --seed")
    if args.resolution % 8:
        parser.error("--resolution must be divisible by 8")
    if not hasattr(torch, "func") or not hasattr(torch.func, "functional_call"):
        parser.error("The installed PyTorch does not provide torch.func.functional_call")
    return targets, anchors


def load_prompt_csv(path: str | os.PathLike[str]) -> list[str]:
    prompts: list[str] = []
    with open(path, newline="", encoding="utf-8") as prompt_file:
        reader = csv.DictReader(prompt_file)
        if not reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError(f"Prompt CSV must contain a 'prompt' column: {path}")
        for row_number, row in enumerate(reader, start=2):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"Prompt CSV contains a blank prompt at row {row_number}: {path}")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"Prompt CSV contains no prompts: {path}")
    return prompts


def resolve_prompts(path: str | None, fallback: Sequence[str]) -> list[str]:
    return load_prompt_csv(path) if path is not None else list(fallback)


def load_retain_texts(
    path: str | os.PathLike[str],
    heads: str,
    targets: Sequence[str],
) -> list[str]:
    if not str(path).endswith(".csv"):
        raise ValueError("retain_path must point to a CSV file")
    requested_heads = [head.strip() for head in heads.split(",") if head.strip()]
    if not requested_heads:
        raise ValueError("heads must contain at least one CSV column")
    values: list[str] = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as retain_file:
        reader = csv.DictReader(retain_file)
        missing = [head for head in requested_heads if head not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Retain CSV is missing column(s): " + ", ".join(missing))
        for row_number, row in enumerate(reader, start=2):
            for head in requested_heads:
                text = (row.get(head) or "").strip()
                if not text:
                    raise ValueError(
                        f"Retain CSV contains a blank {head!r} value at row {row_number}"
                    )
                if text not in seen:
                    seen.add(text)
                    values.append(text)

    filtered = [
        text
        for text in values
        if not any(
            re.search(r"\b" + re.escape(target.lower()) + r"\b", text.lower())
            for target in targets
        )
    ]
    if not filtered:
        raise ValueError("Retain set is empty after target exclusion")
    return filtered


@torch.no_grad()
def encode_prompts(pipe, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
    tokens = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state


@torch.no_grad()
def encode_last_subject_embeddings(
    pipe,
    prompts: Sequence[str],
    device: torch.device,
    chunk_size: int = 128,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(prompts), chunk_size):
        values = list(prompts[start : start + chunk_size])
        tokens = pipe.tokenizer(
            values,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state
        subject_indices = (tokens.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(hidden.shape[0], device=device)
        embeddings.append(hidden[batch_indices, subject_indices].unsqueeze(1))
    return torch.cat(embeddings)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def build_state(
    pipe,
    hidden_cache: dict[str, torch.Tensor],
    prompts: Sequence[str],
    null_hidden: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed: int,
    prefix_index: int,
):
    hidden = torch.cat([hidden_cache[prompt] for prompt in prompts])
    return sample_prefix_diffusion_state(
        pipe,
        hidden,
        null_hidden,
        prompt=" | ".join(prompts),
        seed=seed,
        prefix_index=prefix_index,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        generator=_make_generator(pipe.unet.device, seed),
    )


def build_validation_bank(
    pipe,
    hidden_cache: dict[str, torch.Tensor],
    prompts: Sequence[str],
    null_hidden: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed_offset: int = 0,
) -> list:
    states = []
    rng = random.Random(args.validation_seed + seed_offset)
    for index in range(args.validation_samples):
        prompt = prompts[index % len(prompts)]
        seed = args.validation_seed + seed_offset + index
        prefix_index = rng.randrange(args.num_inference_steps)
        states.append(
            build_state(
                pipe,
                hidden_cache,
                [prompt],
                null_hidden,
                args,
                seed=seed,
                prefix_index=prefix_index,
            )
        )
    return states


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__, "cuda": str(torch.version.cuda)}
    for package in ("diffusers", "transformers", "safetensors", "kmeans-pytorch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _cpu_tensors(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in values.items()
    }


@torch.no_grad()
def verify_materialized_prediction(
    base_unet: torch.nn.Module,
    functional_weights: dict[str, torch.Tensor],
    reloaded_weights: dict[str, torch.Tensor],
    state,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, float | bool]:
    """Compare functional and freshly loaded partial-checkpoint predictions."""

    _, functional_prediction = paired_predictions(base_unet, functional_weights, state)
    fresh_unet = copy.deepcopy(base_unet).to(device="cpu")
    fresh_unet.load_state_dict(reloaded_weights, strict=False)
    fresh_unet.to(device=state.model_input.device).eval()
    kwargs = dict(state.unet_kwargs)
    kwargs.update(
        encoder_hidden_states=state.target_hidden_states,
        return_dict=False,
    )
    loaded_prediction = fresh_unet(state.model_input, state.timestep, **kwargs)[0]
    absolute_error = (loaded_prediction.float() - functional_prediction.float()).abs()
    relative_error = absolute_error / functional_prediction.float().abs().clamp_min(atol)
    matches = torch.allclose(
        loaded_prediction.float(),
        functional_prediction.float(),
        rtol=rtol,
        atol=atol,
    )
    diagnostics = {
        "prediction_max_abs_error": float(absolute_error.max().item()),
        "prediction_max_relative_error": float(relative_error.max().item()),
        "prediction_allclose": bool(matches),
        "prediction_rtol": rtol,
        "prediction_atol": atol,
    }
    if not matches:
        raise RuntimeError(
            "Fresh checkpoint prediction differs from the functional edit: "
            f"{diagnostics}"
        )
    return diagnostics


def save_anchor_artifact(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    targets: Sequence[str],
    anchors: Sequence[str],
    anchor_model: BoundedAnchor,
    edit_state,
    history: Sequence[dict],
    best_step: int | None,
    best_validation_cosine: float | None,
    runtime_metadata: dict[str, object] | None = None,
    export_parity: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "format": ANCHOR_ARTIFACT_FORMAT,
        "anchor_parameterization": "trainable_anchor_direction_and_norm_target_cap",
        "status": status,
        "error": error,
        "target_concepts": list(targets),
        "anchor_concepts": list(anchors),
        "best_step": best_step,
        "best_validation_cosine": best_validation_cosine,
        "resolved_config": vars(args),
        "runtime_metadata": runtime_metadata or {},
        "export_parity": export_parity or {},
        "projector_policy": "aug_num_0_frozen",
        "package_versions": _package_versions(),
        "geometry": edit_state.geometry_metadata(),
        "history": list(history),
        "anchor_state": _cpu_tensors(anchor_model.state_dict()),
        "best_anchor_embeddings": anchor_model().detach().cpu().contiguous(),
        "initial_text_anchor_embeddings": (
            anchor_model.original_initial_anchor.detach().cpu().contiguous()
        ),
        "feasible_initial_anchor_embeddings": (
            anchor_model.feasible_initial_anchor.detach().cpu().contiguous()
        ),
        "max_anchor_norm": anchor_model.max_anchor_norm.detach().cpu().contiguous(),
        "best_residuals": anchor_model.residuals().detach().cpu().contiguous(),
        "target_embeddings": edit_state.target_embeddings.detach().cpu().contiguous(),
        "sum_target_target": edit_state.sum_target_target.detach().cpu().contiguous(),
        "retain_projector": edit_state.retain_projector.detach().cpu().contiguous(),
        "k2": edit_state.k2.detach().cpu().contiguous(),
        "input_hashes": {
            "retain_path": _sha256(args.retain_path),
            "optimization_prompts_path": (
                _sha256(args.optimization_prompts_path)
                if args.optimization_prompts_path
                else None
            ),
            "validation_prompts_path": (
                _sha256(args.validation_prompts_path)
                if args.validation_prompts_path
                else None
            ),
        },
    }
    torch.save(payload, path)


def main(argv=None) -> None:
    parser, args = parse_args(argv)
    targets, anchor_texts = validate_args(parser, args)
    try:
        retain_texts = load_retain_texts(args.retain_path, args.heads, targets)
        optimization_prompts = resolve_prompts(args.optimization_prompts_path, targets)
        validation_prompts = resolve_prompts(
            args.validation_prompts_path,
            optimization_prompts,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    save_path = Path(
        args.save_path or f"logs/closed_form_backprop/{targets[0].lower().replace(' ', '_')}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    resolved_config_path = save_path / "config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(vars(args), config_file, sort_keys=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    seed_everything(args.seed, deterministic=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as error:
        raise RuntimeError("diffusers is required for anchor training") from error

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.unet.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    runtime_metadata = {
        "base_model_id": args.sd_ckpt,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "unet_class": pipe.unet.__class__.__name__,
    }

    target_embedding_prompts = (
        [f"{targets[0]} style"] if args.erase_style else list(targets)
    )
    with torch.no_grad():
        target_embeddings = encode_last_subject_embeddings(
            pipe, target_embedding_prompts, device
        )
        initial_anchor_embeddings = encode_last_subject_embeddings(
            pipe, anchor_texts, device
        )
        retain_embeddings = encode_last_subject_embeddings(pipe, retain_texts, device)
        null_hidden = encode_prompts(pipe, [""], device)
        all_prompts = list(
            dict.fromkeys(
                optimization_prompts + validation_prompts + retain_texts[: args.validation_samples]
            )
        )
        hidden_values = encode_prompts(pipe, all_prompts, device)
        hidden_cache = {
            prompt: hidden_values[index : index + 1]
            for index, prompt in enumerate(all_prompts)
        }

    retain_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    retain_permutation = torch.randperm(
        retain_embeddings.shape[0], generator=retain_generator
    )
    edit_config = DifferentiableLegacyEditConfig(
        residual_scale=args.residual_scale,
        retain_scale=args.retain_scale,
        threshold=args.threshold,
        lamb=args.lamb,
        seed=args.seed,
    )
    edit_state = prepare_differentiable_legacy_edit(
        pipe.unet,
        target_embeddings,
        retain_embeddings,
        null_hidden,
        edit_config,
        retain_permutation=retain_permutation,
    )
    anchor_model = BoundedAnchor(
        target_embeddings,
        initial_anchor_embeddings,
        seed=args.seed,
    ).to(device)

    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_states = build_validation_bank(
        pipe,
        hidden_cache,
        validation_prompts,
        null_hidden,
        args,
    )
    retain_validation_prompts = retain_texts[: args.validation_samples]
    retain_validation_states = build_validation_bank(
        pipe,
        hidden_cache,
        retain_validation_prompts,
        null_hidden,
        args,
        seed_offset=1_000_000,
    )

    training_rng = random.Random(args.seed)

    def training_state_factory():
        prompts = [
            training_rng.choice(optimization_prompts)
            for _ in range(args.anchor_batch_size)
        ]
        state_seed = training_rng.randrange(2**31)
        prefix_index = training_rng.randrange(args.num_inference_steps)
        return build_state(
            pipe,
            hidden_cache,
            prompts,
            null_hidden,
            args,
            seed=state_seed,
            prefix_index=prefix_index,
        )

    metrics_path = save_path / "metrics.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")
    started_at = time.monotonic()
    last_valid_raw_state = {
        name: value.detach().clone()
        for name, value in anchor_model.state_dict().items()
        if name in {"raw_direction", "raw_magnitude"}
    }

    def write_metrics(record: dict) -> None:
        nonlocal last_valid_raw_state
        value = dict(record)
        value["wall_time_seconds"] = time.monotonic() - started_at
        value["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        metrics_file.write(json.dumps(value, sort_keys=True) + "\n")
        metrics_file.flush()
        print(json.dumps(value, sort_keys=True), flush=True)
        history.append(value)
        last_valid_raw_state = {
            name: value.detach().clone()
            for name, value in anchor_model.state_dict().items()
            if name in {"raw_direction", "raw_magnitude"}
        }

    history = []
    result = None
    export_parity: dict[str, object] = {}
    try:
        result = optimize_anchor(
            pipe.unet,
            edit_state,
            anchor_model,
            training_state_factory,
            validation_states,
            AnchorTrainingConfig(
                steps=args.anchor_steps,
                learning_rate=args.anchor_lr,
                validation_interval=args.validation_interval,
                cosine_eps=args.cosine_eps,
            ),
            retain_validation_states=retain_validation_states,
            metrics_callback=write_metrics,
        )
        materialized = edit_state.materialize(anchor_model())
        for name, base_weight in edit_state.base_weights.items():
            current = dict(pipe.unet.named_parameters())[name]
            if not torch.equal(current.detach(), base_weight):
                raise RuntimeError(f"Base U-Net parameter was mutated: {name}")

        checkpoint_path = save_path / f"{args.file_name}.safetensors"
        save_speed_checkpoint(
            materialized,
            checkpoint_path,
            metadata={
                "base_model_id": args.sd_ckpt,
                "component": "unet",
                "target_concept": targets[0],
                "best_step": result.best_step,
            },
        )
        reloaded, metadata = load_edit_checkpoint(
            checkpoint_path,
            reference_state=pipe.unet.state_dict(),
        )
        max_abs_error = 0.0
        for name, expected in materialized.items():
            max_abs_error = max(
                max_abs_error,
                float((reloaded[name] - expected.cpu()).abs().max().item()),
            )
        if metadata.get("format") != SPEED_CHECKPOINT_FORMAT or max_abs_error != 0.0:
            raise RuntimeError("Reloaded checkpoint did not exactly match materialized weights")

        functional_weights, _ = edit_state.effective_parameters(anchor_model())
        export_parity = verify_materialized_prediction(
            pipe.unet,
            functional_weights,
            reloaded,
            validation_states[0],
        )
        export_parity["weight_max_abs_error"] = max_abs_error

        save_anchor_artifact(
            save_path / "anchor_optimization.pt",
            status="complete",
            args=args,
            targets=targets,
            anchors=anchor_texts,
            anchor_model=anchor_model,
            edit_state=edit_state,
            history=history,
            best_step=result.best_step,
            best_validation_cosine=result.best_validation_cosine,
            runtime_metadata=runtime_metadata,
            export_parity=export_parity,
        )
        print(
            f"Saved {checkpoint_path} and {save_path / 'anchor_optimization.pt'} "
            f"(best step {result.best_step}, validation cosine "
            f"{result.best_validation_cosine:.6f})"
        )
    except Exception as error:
        anchor_model.load_state_dict(last_valid_raw_state, strict=False)
        save_anchor_artifact(
            save_path / "anchor_optimization.pt",
            status="failed",
            args=args,
            targets=targets,
            anchors=anchor_texts,
            anchor_model=anchor_model,
            edit_state=edit_state,
            history=result.history if result is not None else history,
            best_step=result.best_step if result is not None else None,
            best_validation_cosine=(
                result.best_validation_cosine if result is not None else None
            ),
            runtime_metadata=runtime_metadata,
            export_parity=export_parity,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
