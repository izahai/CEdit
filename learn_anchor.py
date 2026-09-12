"""CLI for learning CLIP-guided adversarial anchors."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

from src.learned_anchor import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_SD_CKPT,
    LearnedAnchorConfig,
    learn_anchors,
    normalize_target_concepts,
)
from src.utils import seed_everything


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Learn CLIP-guided adversarial anchor prompts for SPEED",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file. Explicit CLI arguments override YAML values.",
    )
    parser.add_argument("--target_concepts", type=str, default=None)
    parser.add_argument("--sd_ckpt", type=str, default=DEFAULT_SD_CKPT)
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--num_prefix_tokens", type=int, default=4)
    parser.add_argument("--num_reference_images", type=int, default=4)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--timestep_min", type=int, default=50)
    parser.add_argument("--timestep_max", type=int, default=950)
    parser.add_argument("--validation_samples", type=int, default=16)
    parser.add_argument("--validation_interval", type=int, default=50)
    parser.add_argument("--reference_inference_steps", type=int, default=50)
    parser.add_argument("--reference_guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--save_root", type=str, default=None)
    return parser


def load_yaml_config(config_path, parser):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        parser.error(
            f"YAML config must contain a mapping, got {type(config).__name__}"
        )
    valid_keys = {
        action.dest
        for action in parser._actions
        if action.dest not in {"config", "help"}
    }
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        parser.error(f"Unknown YAML config key(s): {', '.join(unknown_keys)}")
    return {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in config.items()
    }


def parse_args(argv=None):
    parser = build_argument_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        parser.set_defaults(**load_yaml_config(config_args.config, parser))
    args = parser.parse_args(argv)
    if args.target_concepts is None:
        parser.error("target_concepts is required in the YAML config or CLI")
    if args.save_root is None:
        parser.error("save_root is required in the YAML config or CLI")
    try:
        targets = normalize_target_concepts(args.target_concepts)
        config = LearnedAnchorConfig(
            sd_ckpt=args.sd_ckpt,
            clip_model=args.clip_model,
            num_prefix_tokens=args.num_prefix_tokens,
            num_reference_images=args.num_reference_images,
            num_validation_images=args.num_validation_images,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            timestep_min=args.timestep_min,
            timestep_max=args.timestep_max,
            validation_samples=args.validation_samples,
            validation_interval=args.validation_interval,
            reference_inference_steps=args.reference_inference_steps,
            reference_guidance_scale=args.reference_guidance_scale,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
        )
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    artifact_root = Path(args.save_root)
    if (artifact_root / "manifest.json").exists() or (
        artifact_root / "embeddings.safetensors"
    ).exists():
        parser.error(f"A learned-anchor artifact already exists at {artifact_root}")
    return parser, args, targets, config


def main(argv=None):
    parser, args, targets, config = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device {args.device!r} is not available")
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    seed_everything(args.seed, deterministic=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        safety_checker=None,
        torch_dtype=torch_dtype,
    ).to(device)
    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        torch_dtype=torch_dtype,
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    bundle = learn_anchors(
        pipeline=pipeline,
        clip_model=clip_model,
        clip_processor=clip_processor,
        target_concepts=targets,
        config=config,
    )
    bundle.save(args.save_root)
    print(f"Saved learned anchors to {Path(args.save_root).resolve()}")
    for entry in bundle.manifest["targets"]:
        print(
            f"{entry['target']!r}: prefix={entry['decoded_prefix']!r} | "
            f"validation CLIP={entry['best_validation_clip_similarity']:.6f}"
        )


if __name__ == "__main__":
    main()
