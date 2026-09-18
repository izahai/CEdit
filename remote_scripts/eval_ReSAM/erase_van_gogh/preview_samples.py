"""Generate 3-way preview comparison images (Original SD 1.4 vs SPEED vs Ours) for Van Gogh erasure.

Produces:
1. 10 Van Gogh prompts comparing: Original SD 1.4 | SPEED | Ours (ReSAM)
2. 10 randomly chosen retain artist prompts (from data/style.csv) comparing: Original SD 1.4 | SPEED | Ours (ReSAM)

All comparisons use identical random noise latents. Only the 3-way concatenated images are saved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from src.edit_checkpoint import apply_edit_checkpoint

VAN_GOGH_PROMPTS = [
    "a painting of a starry night over a quiet town in Van Gogh style",
    "a vibrant painting of sunflowers in a clay vase in Van Gogh style",
    "a landscape of rolling hills with dramatic swirling brushwork in Van Gogh style",
    "self-portrait in Van Gogh style with intense gaze and vivid colors",
    "a wheat field under a turbulent twilight sky in the style of Van Gogh",
    "a warm outdoor cafe terrace at night with bright yellow lanterns in Van Gogh style",
    "olive trees with twisting branches under a glowing sun in Van Gogh style",
    "an old rustic stone church under a dramatic swirling sky in Van Gogh style",
    "an iris flower garden in full bloom in the style of Van Gogh",
    "fishing boats docked on a lively riverbank in Van Gogh style",
]

RETAIN_PROMPT_TEMPLATES = [
    "a painting of a scenic coastal harbor in {} style",
    "a portrait of an elegant woman in {} style",
    "a vibrant landscape with rolling hills in {} style",
    "a still life painting of fruit and flowers in {} style",
    "a bustling city street scene in {} style",
    "a peaceful countryside pathway in {} style",
    "a dramatic mountain landscape at sunset in {} style",
    "a tranquil garden with blooming flowers in {} style",
    "a serene river flowing through a forest in {} style",
    "a marketplace full of lively figures in {} style",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3-way preview comparisons (Original | SPEED | Ours) for Van Gogh erasure",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base Stable Diffusion checkpoint",
    )
    parser.add_argument(
        "--speed_ckpt",
        type=str,
        default="logs/speed/van_gogh/weight.pt",
        help="Path to trained SPEED checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--ours_ckpt",
        "--edit_ckpt",
        dest="ours_ckpt",
        type=str,
        default="logs/resam/van_gogh/weight.safetensors",
        help="Path to trained Ours (ReSAM) checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--style_csv",
        type=str,
        default="data/style.csv",
        help="Path to style.csv containing candidate/retain artists",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/resam/van_gogh/preview_samples",
        help="Directory to save concatenated preview images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for artist selection and generation")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps per image")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    return parser.parse_args()


def load_retain_artists(style_csv_path: str | Path, target: str, count: int = 10, seed: int = 42) -> list[str]:
    with open(style_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        target_pattern = r"\b" + re.escape(target.lower()) + r"\b"
        all_artists = [
            row["concept"].strip()
            for row in reader
            if row.get("concept") and not re.search(target_pattern, row["concept"].strip().lower())
        ]
    rng = random.Random(seed)
    return rng.sample(all_artists, min(count, len(all_artists)))


def combine_three_way(
    orig_img: Image.Image,
    speed_img: Image.Image,
    ours_img: Image.Image,
    label_left: str = "Original SD 1.4",
    label_mid: str = "SPEED",
    label_right: str = "Ours",
) -> Image.Image:
    w, h = orig_img.size
    header_h = 40
    combined = Image.new("RGB", (w * 3, h + header_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(combined)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Paste 3 images
    combined.paste(orig_img, (0, header_h))
    combined.paste(speed_img, (w, header_h))
    combined.paste(ours_img, (w * 2, header_h))

    # Helper to draw centered label in each column
    def draw_centered_text(center_x: int, text: str):
        if hasattr(draw, "textbbox") and font is not None:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
        else:
            tw = len(text) * 6
        draw.text((center_x - tw // 2, 12), text, fill=(255, 255, 255), font=font)

    draw_centered_text(w // 2, label_left)
    draw_centered_text(w + w // 2, label_mid)
    draw_centered_text(w * 2 + w // 2, label_right)

    # Column divider lines
    draw.line([(w, 0), (w, h + header_h)], fill=(80, 80, 80), width=2)
    draw.line([(w * 2, 0), (w * 2, h + header_h)], fill=(80, 80, 80), width=2)
    return combined


def sanitize_filename(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    return cleaned[:max_len]


def main():
    args = parse_args()
    save_path = Path(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Import diffusers here so CPU unit tests importing this module don't fail
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    print(f"Loading pipeline from {args.sd_ckpt} on {device} ({dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    base_unet = pipe.unet

    print(f"Loading SPEED weights from {args.speed_ckpt}...")
    speed_unet = copy.deepcopy(base_unet)
    apply_edit_checkpoint(speed_unet, args.speed_ckpt, device="cpu")

    print(f"Loading Ours (ReSAM) weights from {args.ours_ckpt}...")
    ours_unet = copy.deepcopy(base_unet)
    apply_edit_checkpoint(ours_unet, args.ours_ckpt, device="cpu")

    # Select retain artists
    retain_artists = load_retain_artists(args.style_csv, target="Van Gogh", count=10, seed=args.seed)
    print(f"Randomly selected 10 retain artists: {', '.join(retain_artists)}")

    # Prepare directories: only concatenated images are saved directly inside van_gogh/ and retain/
    vg_dir = save_path / "van_gogh"
    ret_dir = save_path / "retain"
    vg_dir.mkdir(parents=True, exist_ok=True)
    ret_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 1. Generate Van Gogh preview images (10 prompts)
    print("\n--- Generating 10 Van Gogh Target 3-Way Comparisons (Original | SPEED | Ours) ---")
    for idx, prompt in enumerate(VAN_GOGH_PROMPTS, start=1):
        latent_seed = args.seed + idx * 100

        print(f"[{idx}/10] Van Gogh: {prompt}")
        # Original
        pipe.unet = base_unet
        orig_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # SPEED
        pipe.unet = speed_unet
        speed_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # Ours
        pipe.unet = ours_unet
        ours_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        combined = combine_three_way(
            orig_img,
            speed_img,
            ours_img,
            label_left="Original SD 1.4",
            label_mid="SPEED",
            label_right="Ours",
        )

        fname = f"{idx:02d}_{sanitize_filename(prompt)}"
        out_file = vg_dir / f"{fname}.jpg"
        combined.save(out_file, quality=95)

        records.append({
            "type": "target",
            "concept": "Van Gogh",
            "prompt": prompt,
            "seed": latent_seed,
            "filename": f"{fname}.jpg",
        })

    # 2. Generate Retain preview images (10 artists)
    print("\n--- Generating 10 Retain Artists 3-Way Comparisons (Original | SPEED | Ours) ---")
    for idx, (artist, template) in enumerate(zip(retain_artists, RETAIN_PROMPT_TEMPLATES), start=1):
        prompt = template.format(artist)
        latent_seed = args.seed + 1000 + idx * 100

        print(f"[{idx}/10] Retain ({artist}): {prompt}")
        # Original
        pipe.unet = base_unet
        orig_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # SPEED
        pipe.unet = speed_unet
        speed_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        # Ours
        pipe.unet = ours_unet
        ours_img = pipe(
            prompt=prompt,
            generator=torch.Generator(device=device).manual_seed(latent_seed),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

        combined = combine_three_way(
            orig_img,
            speed_img,
            ours_img,
            label_left="Original SD 1.4",
            label_mid="SPEED",
            label_right="Ours",
        )

        fname = f"{idx:02d}_{sanitize_filename(artist)}_{sanitize_filename(prompt)}"
        out_file = ret_dir / f"{fname}.jpg"
        combined.save(out_file, quality=95)

        records.append({
            "type": "retain",
            "concept": artist,
            "prompt": prompt,
            "seed": latent_seed,
            "filename": f"{fname}.jpg",
        })

    # Save summary metadata
    with open(save_path / "preview_manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nCompleted 3-way preview sampling! All concatenated images saved under: {save_path.resolve()}")
    print(f"Van Gogh 3-way comparisons: {vg_dir.resolve()}")
    print(f"Retain 3-way comparisons:   {ret_dir.resolve()}")


if __name__ == "__main__":
    main()
