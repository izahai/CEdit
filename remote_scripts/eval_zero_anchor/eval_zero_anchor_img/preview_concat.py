"""Generate 3-way concatenated comparison images for Van Gogh erasure.

Compares:
1. Original Stable Diffusion v1.4 (unmodified)
2. SPEED with "art" anchor
3. SPEED with "<zero>" anchor

Supports:
- 10-prompt preview evaluation mode (legacy)
- 100-prompt benchmark evaluation mode (5 canonical templates x 20 seeds for Van Gogh,
  and 100 distinct retain artists from data/style_100_retain_eval.csv paired with cyclic templates)

Concatenation:
- Horizontal stitch in exact order: [Original | Art Anchor | Zero Anchor]
- Raw image concatenation (512x1536 pixels, no text banners or borders)
- Only the concatenated images are saved (no single images, no extra disk usage).
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random
import re
import sys

current = Path(__file__).resolve()
REPO_ROOT = None
for parent in current.parents:
    if (parent / "train_erase_null.py").exists() or (parent / "src").exists():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    REPO_ROOT = current.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import torch

from src.edit_checkpoint import apply_edit_checkpoint
from src.template import painting_templates

CANONICAL_STYLE_TEMPLATES = [
    "An image in the style of {artist}.",
    "A portrait in the style of {artist}.",
    "A landscape in the style of {artist}.",
    "A city street in the style of {artist}.",
    "A still life in the style of {artist}.",
]


def slugify(text: str, max_len: int = 50) -> str:
    """Create a safe filesystem filename slug from prompt text."""
    slug = re.sub(r"[^\w\s]", "", text).strip().replace(" ", "_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_len]


def concat_three_images(
    img_orig: Image.Image,
    img_art: Image.Image,
    img_zero: Image.Image,
) -> Image.Image:
    """Horizontally stitch 3 images in order: [Original | Art Anchor | Zero Anchor]."""
    w1, h1 = img_orig.size
    w2, h2 = img_art.size
    w3, h3 = img_zero.size
    if not (h1 == h2 == h3):
        raise ValueError(f"Image heights do not match: {h1}, {h2}, {h3}")
    total_w = w1 + w2 + w3
    combined = Image.new("RGB", (total_w, h1))
    combined.paste(img_orig, (0, 0))
    combined.paste(img_art, (w1, 0))
    combined.paste(img_zero, (w1 + w2, 0))
    return combined


def load_random_artists(
    style_csv_path: str | Path,
    target: str = "Van Gogh",
    count: int = 10,
    seed: int = 42,
) -> list[str]:
    """Sample candidate artists from style.csv excluding the target concept."""
    artists = []
    with open(style_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row.get("concept", "").strip()
            if concept and concept.lower() != target.lower():
                artists.append(concept)
    if len(artists) < count:
        raise ValueError(f"Expected at least {count} artists in {style_csv_path}, found {len(artists)}")
    rng = random.Random(seed)
    return rng.sample(artists, count)


def build_van_gogh_prompts(count: int = 10) -> list[str]:
    """Return the first `count` generic painting templates formatted for Van Gogh."""
    return [template.format("Van Gogh") for template in painting_templates[:count]]


def build_random_artist_prompts(artists: list[str]) -> list[tuple[str, str]]:
    """Pair each artist with the corresponding painting template. Returns list of (artist, prompt)."""
    return [
        (artist, template.format(artist))
        for artist, template in zip(artists, painting_templates[:len(artists)])
    ]


def build_van_gogh_100_prompts(
    count: int = 100,
    base_seed: int = 42,
    templates: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Build (prompt, seed) pairs for Van Gogh.

    Uses 5 canonical templates cycled across `count` with deterministic seeds.
    For count=100 with 5 templates, this produces 20 seeds per template.
    """
    if templates is None:
        templates = CANONICAL_STYLE_TEMPLATES
    prompts = []
    num_templates = len(templates)
    for idx in range(count):
        template = templates[idx % num_templates]
        prompt = template.format(artist="Van Gogh")
        seed = base_seed + idx * 1000
        prompts.append((prompt, seed))
    return prompts


def build_retain_100_prompts(
    style_csv_path: str | Path,
    count: int = 100,
    base_seed: int = 42,
    templates: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Build (artist, prompt, seed) tuples for retain styles.

    Reads artists from style_csv_path, pairs each with cycling canonical templates,
    and assigns deterministic seeds.
    """
    if templates is None:
        templates = CANONICAL_STYLE_TEMPLATES
    artists = []
    with open(style_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row.get("concept", "").strip()
            if concept and concept.lower() != "van gogh":
                artists.append(concept)
    if len(artists) < count:
        raise ValueError(f"Expected at least {count} retain artists in {style_csv_path}, found {len(artists)}")
    artists = artists[:count]
    num_templates = len(templates)
    items = []
    for idx, artist in enumerate(artists):
        template = templates[idx % num_templates]
        prompt = template.format(artist=artist)
        seed = base_seed + (count + idx) * 1000
        items.append((artist, prompt, seed))
    return items


def default_style_csv() -> str:
    eval_csv = REPO_ROOT / "data" / "style_100_retain_eval.csv"
    if eval_csv.exists():
        return str(eval_csv)
    return str(REPO_ROOT / "data" / "style.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3-way concatenated comparison images for Van Gogh erasure (Original | Art | Zero)",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base Stable Diffusion checkpoint path or HuggingFace ID",
    )
    parser.add_argument(
        "--art_ckpt",
        type=str,
        default="remote_scripts/eval_zero_anchor/outputs/checkpoints/art/weight.pt",
        help="Path to SPEED checkpoint trained with 'art' anchor (.pt)",
    )
    parser.add_argument(
        "--zero_ckpt",
        type=str,
        default="remote_scripts/eval_zero_anchor/outputs/checkpoints/zero/weight.pt",
        help="Path to SPEED checkpoint trained with '<zero>' anchor (.pt)",
    )
    parser.add_argument(
        "--style_csv",
        type=str,
        default=default_style_csv(),
        help="Path to style CSV containing retain candidate artists",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="remote_scripts/eval_zero_anchor/outputs/concat_images_100",
        help="Directory to save 3-way concatenated comparison images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to generate per category (target erase and retain)",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="canonical_5",
        choices=["canonical_5", "legacy_painting"],
        help="Prompt style mode: 'canonical_5' (5 canonical templates) or 'legacy_painting'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducible sampling",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of diffusion inference steps per image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if target image file already exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def run_pipeline(args):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    output_dir = Path(args.output_dir)
    vg_dir = output_dir / "van_gogh"
    random_dir = output_dir / "random_artists"
    vg_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading base pipeline from {args.sd_ckpt} on {device} ({dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Base UNet for original model
    base_unet = pipe.unet

    print(f"Loading SPEED 'art' anchor weights from {args.art_ckpt}...")
    art_unet = copy.deepcopy(base_unet)
    apply_edit_checkpoint(art_unet, args.art_ckpt, device="cpu")

    print(f"Loading SPEED '<zero>' anchor weights from {args.zero_ckpt}...")
    zero_unet = copy.deepcopy(base_unet)
    apply_edit_checkpoint(zero_unet, args.zero_ckpt, device="cpu")

    # Build prompt and seed items
    if args.prompt_mode == "canonical_5":
        vg_items = build_van_gogh_100_prompts(args.count, base_seed=args.seed)
        retain_items = build_retain_100_prompts(args.style_csv, args.count, base_seed=args.seed)
    else:
        legacy_prompts = build_van_gogh_prompts(args.count)
        vg_items = [(p, args.seed + idx * 1000) for idx, p in enumerate(legacy_prompts, start=1)]
        sampled_artists = load_random_artists(args.style_csv, target="Van Gogh", count=args.count, seed=args.seed)
        legacy_artist_items = build_random_artist_prompts(sampled_artists)
        retain_items = [
            (artist, prompt, args.seed + (args.count + idx) * 1000)
            for idx, (artist, prompt) in enumerate(legacy_artist_items, start=1)
        ]

    pad_digits = 3 if args.count >= 100 else 2

    # Helper function to generate and save concatenated image for a prompt
    def generate_and_save_concat(prompt: str, latent_seed: int, save_path: Path):
        if args.skip_existing and save_path.exists():
            print(f"       -> Skipped (already exists): {save_path.name}")
            return

        # Model 1: Original SD 1.4
        pipe.unet = base_unet
        generator = torch.Generator(device=device).manual_seed(latent_seed)
        img_orig = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        # Model 2: SPEED (art anchor)
        pipe.unet = art_unet
        generator = torch.Generator(device=device).manual_seed(latent_seed)
        img_art = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        # Model 3: SPEED (zero anchor)
        pipe.unet = zero_unet
        generator = torch.Generator(device=device).manual_seed(latent_seed)
        img_zero = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        # 3-way horizontal concatenation: [Original | Art | Zero]
        concat_img = concat_three_images(img_orig, img_art, img_zero)
        concat_img.save(save_path)

    # 1. Generate Van Gogh comparison images
    print(f"\n--- Generating {len(vg_items)} Van Gogh 3-Way Comparisons [Original | Art | Zero] ({args.steps} steps) ---")
    for idx, (prompt, latent_seed) in enumerate(vg_items, start=1):
        slug = slugify(prompt, max_len=45)
        out_file = vg_dir / f"{idx:0{pad_digits}d}_van_gogh_{slug}.png"
        print(f"[{idx:0{pad_digits}d}/{len(vg_items)}] Van Gogh (seed={latent_seed}): {prompt}")
        generate_and_save_concat(prompt, latent_seed, out_file)
        print(f"       -> Saved: {out_file.name}")

    # 2. Generate Retain Artist comparison images
    print(f"\n--- Generating {len(retain_items)} Retain Artist 3-Way Comparisons [Original | Art | Zero] ({args.steps} steps) ---")
    for idx, (artist, prompt, latent_seed) in enumerate(retain_items, start=1):
        artist_slug = slugify(artist, max_len=20)
        prompt_slug = slugify(prompt, max_len=35)
        out_file = random_dir / f"{idx:0{pad_digits}d}_{artist_slug}_{prompt_slug}.png"
        print(f"[{idx:0{pad_digits}d}/{len(retain_items)}] Retain '{artist}' (seed={latent_seed}): {prompt}")
        generate_and_save_concat(prompt, latent_seed, out_file)
        print(f"       -> Saved: {out_file.name}")

    print("\n" + "=" * 60)
    print(f"All {len(vg_items) + len(retain_items)} 3-way concatenated comparisons complete!")
    print(f"Van Gogh images:     {vg_dir} ({len(vg_items)} images)")
    print(f"Retain artist images:{random_dir} ({len(retain_items)} images)")
    print("Image format: [Original | SPEED (art) | SPEED (<zero>)] (512x1536)")
    print("=" * 60)


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
