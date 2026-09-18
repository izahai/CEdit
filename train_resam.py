"""Train ReSAM: Retain-Guided Sparse Anchor Mixture for Stable Diffusion v1.4."""

from __future__ import annotations

import argparse
import copy
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
from tqdm import tqdm

from src.diffusion_training_inputs import (
    _make_generator,
    build_state,
    cpu_tensor_dict,
    encode_last_subject_embeddings,
    encode_prompts,
    get_package_versions,
    load_candidate_concepts,
    load_retain_texts,
    resolve_prompts,
    sha256_file,
)
from src.differentiable_legacy_edit import (
    SPEED_CHECKPOINT_FORMAT,
    DifferentiableLegacyEditConfig,
    prepare_differentiable_legacy_edit,
)
from src.edit_checkpoint import load_edit_checkpoint, save_speed_checkpoint
from src.resam_training import (
    ReSAMTrainingConfig,
    ReSAMTrainingResult,
    SparseAnchorMixture,
    optimize_resam,
)
from src.utils import seed_everything

RESAM_ARTIFACT_FORMAT = "resam-optimization-v1"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a sparse candidate anchor mixture for concept erasure",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # Concept specification
    parser.add_argument("--target_concepts", type=str, default=None)
    parser.add_argument("--candidate_concepts", type=str, default=None)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default="concept")
    parser.add_argument("--erase_style", action="store_true", default=False)

    # ReSAM Sparse Mixture Parameters
    parser.add_argument("--resam_k", type=int, default=2)
    parser.add_argument("--resam_temperature", type=float, default=1.0)
    parser.add_argument(
        "--resam_ste",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use straight-through estimator for dense backward gradient through top-k",
    )
    parser.add_argument("--resam_steps", type=int, default=200)
    parser.add_argument("--resam_lr", type=float, default=0.05)
    parser.add_argument(
        "--use_null_retain_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include null-prompt predicted-noise MSE alongside retain prompts",
    )

    # SPEED closed-form options (locked to legacy value editing)
    parser.add_argument("--baseline", type=str, default="SPEED")
    parser.add_argument("--params", type=str, default="V")
    parser.add_argument("--anchor_mode", type=str, default="legacy")
    parser.add_argument("--aug_num", type=int, default=0)
    parser.add_argument("--retain_projection_rank", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.031670106575)
    parser.add_argument("--retain_scale", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument(
        "--use_k2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include K2 null-prompt invariant constraint",
    )

    # Sampling controls
    parser.add_argument(
        "--num_train_inference_steps",
        type=int,
        default=None,
        help="Number of discrete inference steps in the training diffusion trajectory (defaults to num_inference_steps)",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--resolution", type=int, default=512)

    return parser


def load_yaml_config(
    config_path: str | os.PathLike[str],
    parser: argparse.ArgumentParser,
) -> dict:
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        parser.error("YAML config must contain a mapping")

    # Map aliases if present
    alias_map = {
        "steps": "resam_steps",
        "lr": "resam_lr",
        "k": "resam_k",
        "temperature": "resam_temperature",
        "train_inference_steps": "num_train_inference_steps",
    }
    mapped_config = {}
    for key, value in config.items():
        actual_key = alias_map.get(key, key)
        mapped_config[actual_key] = value

    valid_keys = {
        action.dest for action in parser._actions if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(mapped_config) - valid_keys)
    if unknown:
        parser.error("Unknown YAML config key(s): " + ", ".join(unknown))
    return {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in mapped_config.items()
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


def normalize_target(value) -> str:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, list):
        if not all(isinstance(x, str) for x in value):
            raise TypeError("target_concepts elements must be strings")
        parts = [x.strip() for x in value if x.strip()]
    else:
        raise TypeError("target_concepts must be a string or list of strings")

    if not parts:
        raise ValueError("target_concepts cannot be empty")
    if len(parts) != 1:
        raise ValueError("ReSAM supports exactly one target concept")
    return parts[0]


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[str, list[str], str]:
    required = {
        "target_concepts": args.target_concepts,
        "candidate_concepts": args.candidate_concepts,
        "retain_path": args.retain_path,
        "heads": args.heads,
    }
    missing = [name for name, val in required.items() if val is None]
    if missing:
        parser.error("Missing required option(s): " + ", ".join(missing))

    try:
        target = normalize_target(args.target_concepts)
    except (TypeError, ValueError) as err:
        parser.error(str(err))

    try:
        candidates, candidate_file_hash = load_candidate_concepts(args.candidate_concepts)
    except (FileNotFoundError, OSError, ValueError) as err:
        parser.error(str(err))

    target_pattern = r"\b" + re.escape(target.lower()) + r"\b"
    candidates = [
        c for c in candidates
        if not re.search(target_pattern, c.lower())
    ]
    if not candidates:
        parser.error("Candidate bank is empty after target exclusion")

    v_count = len(candidates)
    if not isinstance(args.resam_k, int) or isinstance(args.resam_k, bool) or not 1 <= args.resam_k <= v_count:
        parser.error(f"--resam_k must be an integer between 1 and bank size ({v_count}), got {args.resam_k}")

    if (
        isinstance(args.resam_temperature, bool)
        or not isinstance(args.resam_temperature, (int, float))
        or not math.isfinite(args.resam_temperature)
        or args.resam_temperature <= 0
    ):
        parser.error("--resam_temperature must be a positive, finite float")

    if not isinstance(args.use_null_retain_loss, bool):
        parser.error("--use_null_retain_loss must be a boolean")

    if (
        isinstance(args.resam_lr, bool)
        or not isinstance(args.resam_lr, (int, float))
        or not math.isfinite(args.resam_lr)
        or args.resam_lr <= 0
    ):
        parser.error("--resam_lr must be a positive, finite float")

    if not isinstance(args.resam_steps, int) or isinstance(args.resam_steps, bool) or args.resam_steps <= 0:
        parser.error("--resam_steps must be a positive integer")

    # Locked SPEED modes
    expected_modes = {
        "params": (args.params, "V"),
        "baseline": (args.baseline, "SPEED"),
        "anchor_mode": (args.anchor_mode, "legacy"),
        "aug_num": (args.aug_num, 0),
    }
    for name, (actual, wanted) in expected_modes.items():
        if actual != wanted:
            parser.error(f"--{name} must be {wanted!r} for ReSAM")

    if not isinstance(args.retain_projection_rank, int) or isinstance(args.retain_projection_rank, bool) or args.retain_projection_rank <= 0:
        parser.error("--retain_projection_rank must be a positive integer")

    if (
        isinstance(args.threshold, bool)
        or not isinstance(args.threshold, (int, float))
        or not math.isfinite(args.threshold)
    ):
        parser.error("--threshold must be finite")

    if not isinstance(args.seed, int) or isinstance(args.seed, bool) or args.seed < 0:
        parser.error("--seed must be a nonnegative integer")

    if not isinstance(args.file_name, str) or not args.file_name or "/" in args.file_name:
        parser.error("--file_name must be a non-empty basename")

    if not isinstance(args.heads, str) or not args.heads.strip():
        parser.error("--heads must be a non-empty string")

    if args.num_train_inference_steps is None:
        args.num_train_inference_steps = args.num_inference_steps

    if (
        not isinstance(args.num_train_inference_steps, int)
        or isinstance(args.num_train_inference_steps, bool)
        or args.num_train_inference_steps <= 0
    ):
        parser.error("--num_train_inference_steps must be a positive integer")

    if (
        not isinstance(args.num_inference_steps, int)
        or isinstance(args.num_inference_steps, bool)
        or args.num_inference_steps <= 0
    ):
        parser.error("--num_inference_steps must be a positive integer")

    return target, candidates, candidate_file_hash


def verify_materialized_prediction(
    base_unet: torch.nn.Module,
    functional_weights: dict[str, torch.Tensor],
    reloaded_weights: dict[str, torch.Tensor],
    state,
) -> dict[str, float]:
    """Verify that reloaded safetensors weights match functional overrides on a fixed state."""
    kwargs = dict(state.unet_kwargs)
    kwargs.update(
        encoder_hidden_states=state.target_hidden_states,
        return_dict=False,
    )
    with torch.no_grad():
        func_out = torch.func.functional_call(
            base_unet,
            functional_weights,
            args=(state.model_input, state.timestep),
            kwargs=kwargs,
            strict=False,
        )
        if isinstance(func_out, tuple):
            func_pred = func_out[0]
        elif hasattr(func_out, "sample"):
            func_pred = func_out.sample
        else:
            func_pred = func_out

        reloaded_named = {
            name: reloaded_weights[name].to(
                device=base_unet.device, dtype=base_unet.dtype
            )
            for name in functional_weights
            if name in reloaded_weights
        }
        reloaded_out = torch.func.functional_call(
            base_unet,
            reloaded_named,
            args=(state.model_input, state.timestep),
            kwargs=kwargs,
            strict=False,
        )
        if isinstance(reloaded_out, tuple):
            reloaded_pred = reloaded_out[0]
        elif hasattr(reloaded_out, "sample"):
            reloaded_pred = reloaded_out.sample
        else:
            reloaded_pred = reloaded_out

        max_pred_err = float((reloaded_pred - func_pred).abs().max().item())

    return {"prediction_max_abs_error": max_pred_err}


def save_resam_artifact(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    target: str,
    candidate_names: list[str],
    candidate_file_hash: str,
    candidate_embeddings: torch.Tensor,
    initial_scores: torch.Tensor,
    mixture: SparseAnchorMixture,
    edit_state: DifferentiableLegacyEditState,
    history: list[dict[str, object]],
    final_step: int,
    final_loss: float,
    runtime_metadata: dict[str, object],
    export_parity: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "format": RESAM_ARTIFACT_FORMAT,
        "status": status,
        "error": error,
        "args": vars(args),
        "target_concept": target,
        "candidate_concepts": candidate_names,
        "candidate_file_hash": candidate_file_hash,
        "final_step": final_step,
        "final_loss": final_loss,
        "history": history,
        "initial_scores": initial_scores.cpu(),
        "final_scores": mixture.raw_scores.detach().cpu(),
        "candidate_embeddings": candidate_embeddings.detach().cpu(),
        "target_embeddings": edit_state.target_embeddings.detach().cpu(),
        "anchor_embeddings": mixture().detach().cpu(),
        "residuals": mixture.residuals().detach().cpu(),
        "sum_target_target": edit_state.sum_target_target.detach().cpu(),
        "retain_projector": edit_state.retain_projector.detach().cpu(),
        "k2": edit_state.k2.detach().cpu() if edit_state.k2 is not None else None,
        "runtime_metadata": runtime_metadata,
        "export_parity": export_parity,
        "input_hashes": {
            "retain_path": sha256_file(args.retain_path),
            "candidate_concepts": candidate_file_hash,
        },
    }
    torch.save(payload, path)


def main(argv=None) -> None:
    parser, args = parse_args(argv)
    target, candidates, candidate_file_hash = validate_args(parser, args)

    try:
        retain_texts = load_retain_texts(args.retain_path, args.heads, [target])
        retain_prompts = list(retain_texts)
    except (OSError, ValueError) as err:
        parser.error(str(err))

    save_path = Path(
        args.save_path or f"logs/resam/{target.lower().replace(' ', '_')}"
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

    # Runtime-only diffusers import
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as err:
        raise RuntimeError("diffusers is required for ReSAM training") from err

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.unet.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()

    runtime_metadata = {
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "package_versions": get_package_versions(),
        "sd_ckpt": args.sd_ckpt,
        "scheduler_config": dict(pipe.scheduler.config),
        "unet_class": pipe.unet.__class__.__name__,
    }

    target_embedding_prompts = [f"{target} style"] if args.erase_style else [target]
    with torch.no_grad():
        target_embeddings = encode_last_subject_embeddings(
            pipe, target_embedding_prompts, device
        )
        candidate_embeddings = encode_last_subject_embeddings(
            pipe, candidates, device
        )
        retain_embeddings = encode_last_subject_embeddings(pipe, retain_texts, device)
        null_hidden = encode_prompts(pipe, [""], device)

        all_prompts = list(dict.fromkeys(retain_prompts[:500] + candidates))
        hidden_values = encode_prompts(pipe, all_prompts, device)
        hidden_cache = {
            prompt: hidden_values[idx : idx + 1]
            for idx, prompt in enumerate(all_prompts)
        }

    retain_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    retain_permutation = torch.randperm(
        retain_embeddings.shape[0], generator=retain_generator
    )
    edit_config = DifferentiableLegacyEditConfig(
        residual_scale=args.residual_scale,
        retain_scale=args.retain_scale,
        retain_projection_rank=args.retain_projection_rank,
        threshold=args.threshold,
        lamb=args.lamb,
        seed=args.seed,
        use_k2=args.use_k2,
    )
    edit_state = prepare_differentiable_legacy_edit(
        pipe.unet,
        target_embeddings,
        retain_embeddings,
        null_hidden if args.use_k2 else None,
        edit_config,
        retain_permutation=retain_permutation,
    )

    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mixture = SparseAnchorMixture(
        candidate_embeddings,
        target_embeddings,
        k=args.resam_k,
        temperature=args.resam_temperature,
        use_ste=args.resam_ste,
        candidate_names=candidates,
    ).to(device)
    initial_scores = mixture.raw_scores.detach().clone()

    training_rng = random.Random(args.seed)

    def training_state_factory() -> DiffusionState:
        prompt = training_rng.choice(retain_prompts[:500])
        state_seed = training_rng.randrange(2**31)
        prefix_index = training_rng.randrange(args.num_train_inference_steps)
        return build_state(
            pipe,
            hidden_cache,
            [prompt],
            null_hidden,
            args,
            seed=state_seed,
            prefix_index=prefix_index,
        )

    metrics_path = save_path / "metrics.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")
    started_at = time.monotonic()
    history: list[dict[str, object]] = []

    pbar = tqdm(total=args.resam_steps, desc="ReSAM Training")

    def write_metrics(record: dict) -> None:
        rec = dict(record)
        rec["wall_time_seconds"] = time.monotonic() - started_at
        rec["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        metrics_file.write(json.dumps(rec, sort_keys=True) + "\n")
        metrics_file.flush()
        # print(json.dumps(rec, sort_keys=True), flush=True)

        chosen = ", ".join(record.get("selected_names", []))
        loss = record.get("train_loss", 0.0)
        pbar.set_postfix_str(f"loss: {loss:.4f}, chosen: [{chosen}]")
        pbar.update(1)

    config = ReSAMTrainingConfig(
        steps=args.resam_steps,
        learning_rate=args.resam_lr,
        use_null_retain_loss=args.use_null_retain_loss,
    )

    try:
        result = optimize_resam(
            pipe.unet,
            edit_state,
            mixture,
            training_state_factory,
            config,
            metrics_callback=write_metrics,
        )
        pbar.close()

        materialized = edit_state.materialize(mixture())
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
                "target_concept": target,
                "final_step": result.final_step,
                "final_loss": result.final_loss,
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

        sample_state = training_state_factory()
        functional_weights, _ = edit_state.effective_parameters(mixture())
        export_parity = verify_materialized_prediction(
            pipe.unet,
            functional_weights,
            reloaded,
            sample_state,
        )
        export_parity["weight_max_abs_error"] = max_abs_error

        save_resam_artifact(
            save_path / "resam_optimization.pt",
            status="complete",
            args=args,
            target=target,
            candidate_names=candidates,
            candidate_file_hash=candidate_file_hash,
            candidate_embeddings=candidate_embeddings,
            initial_scores=initial_scores,
            mixture=mixture,
            edit_state=edit_state,
            history=result.history,
            final_step=result.final_step,
            final_loss=result.final_loss,
            runtime_metadata=runtime_metadata,
            export_parity=export_parity,
        )
        print(
            f"Saved {checkpoint_path} and {save_path / 'resam_optimization.pt'} "
            f"(final step {result.final_step}, final loss {result.final_loss:.6f})"
        )

    except Exception as err:
        save_resam_artifact(
            save_path / "resam_optimization.pt",
            status="failed",
            args=args,
            target=target,
            candidate_names=candidates,
            candidate_file_hash=candidate_file_hash,
            candidate_embeddings=candidate_embeddings,
            initial_scores=initial_scores,
            mixture=mixture,
            edit_state=edit_state,
            history=history,
            final_step=args.resam_steps,
            final_loss=0.0,
            runtime_metadata=runtime_metadata,
            error=f"{type(err).__name__}: {err}",
        )
        raise
    finally:
        pbar.close()
        metrics_file.close()


if __name__ == "__main__":
    main()
