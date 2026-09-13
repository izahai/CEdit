import argparse
import sys

import torch

sys.path.append(".")
from utils.esd_trainer import ESDConfig, run_esd_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TrainESD for FLUX",
        description="Fine-tune a FLUX transformer to erase concepts.",
    )
    parser.add_argument(
        "--basemodel_id",
        help="HF model id for any FLUX-compatible diffusers pipeline",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument("--erase_concept", help="concept to erase", type=str, required=True)
    parser.add_argument("--erase_from", help="target concept to erase from", type=str, default=None)
    parser.add_argument("--num_inference_steps", help="number of denoising steps", type=int, default=28)
    parser.add_argument("--guidance_scale", help="guidance scale for direct transformer calls", type=float, default=1)
    parser.add_argument(
        "--inference_guidance_scale",
        help="guidance scale used while sampling xt",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--max_sequence_length",
        help="max sequence length for FLUX text encoders. Defaults to 77; increase for longer prompts.",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--train_method",
        help="train method (esd-x or esd-x-strict). Legacy aliases still work.",
        type=str,
        required=True,
    )
    parser.add_argument("--iterations", help="number of optimization steps", type=int, default=1400)
    parser.add_argument(
        "--resolution",
        help="training resolution. Defaults to 512 to keep memory manageable.",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--lr",
        help="learning rate. If omitted, a family-specific default is used.",
        type=float,
        default=None,
    )
    parser.add_argument("--batchsize", help="batch size for prompt embeddings", type=int, default=1)
    parser.add_argument("--negative_guidance", help="negative guidance value", type=float, default=1)
    parser.add_argument("--save_path", help="directory to save checkpoints", type=str, default="esd-models/flux/")
    parser.add_argument("--device", help="device to train on", type=str, default="cuda:0")
    parser.add_argument(
        "--gradient_checkpointing",
        help="enable gradient checkpointing on the trainable component",
        action="store_true",
    )
    parser.add_argument(
        "--allow_tf32",
        help="allow TF32 matmuls on supported CUDA hardware",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ESDConfig(
        family="flux",
        base_model_id=args.basemodel_id,
        erase_concept=args.erase_concept,
        erase_from=args.erase_from,
        train_method=args.train_method,
        iterations=args.iterations,
        lr=args.lr,
        negative_guidance=args.negative_guidance,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batchsize,
        resolution=args.resolution,
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        inference_guidance_scale=args.inference_guidance_scale,
        max_sequence_length=args.max_sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
        allow_tf32=args.allow_tf32,
    )
    checkpoint_path = run_esd_training(config)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
