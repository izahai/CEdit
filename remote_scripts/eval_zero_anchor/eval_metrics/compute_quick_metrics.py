"""Fast CLIP evaluation suite for Van Gogh style erasure comparing Art Anchor vs Zero Anchor.

Operates directly on the generated 3-way concatenated comparison images (1536x512)
without requiring additional diffusion generation.

Supports both:
- 10-prompt preview evaluation mode (legacy)
- 100-prompt benchmark evaluation mode (canonical_5 templates, 100 retain artists)

Evaluates:
1. Target Erasure (Van Gogh prompts):
   - Concept CLIP score: sim(image, "Van Gogh") [Lower is better]
   - Prompt CLIP score: sim(image, full_prompt) [Lower is better]
   - Relative Erasure Rate (% reduction vs Original)
2. Retain Preservation (Random / Benchmark Retain Artists):
   - Concept CLIP score: sim(image, artist) [Higher is better, closer to original]
   - Prompt CLIP score: sim(image, full_prompt) [Higher is better, closer to original]
   - Preservation Ratio (% preserved vs Original)
3. Image Fidelity to Original:
   - Image-to-image CLIP similarity: sim(edited_image, original_image) [Higher is better]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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

try:
    from remote_scripts.eval_zero_anchor.preview_concat import (
        CANONICAL_STYLE_TEMPLATES,
        build_random_artist_prompts,
        build_retain_100_prompts,
        build_van_gogh_100_prompts,
        build_van_gogh_prompts,
        load_random_artists,
    )
except (ImportError, ModuleNotFoundError):
    from remote_scripts.eval_zero_anchor.eval_zero_anchor_img.preview_concat import (
        CANONICAL_STYLE_TEMPLATES,
        build_random_artist_prompts,
        build_retain_100_prompts,
        build_van_gogh_100_prompts,
        build_van_gogh_prompts,
        load_random_artists,
    )
from src.template import painting_templates


def detect_default_image_dir() -> str:
    candidates = [
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "outputs" / "concat_images_100",
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "eval_zero_anchor_img" / "outputs" / "concat_images_100",
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "outputs" / "concat_images",
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "eval_zero_anchor_img" / "outputs" / "concat_images",
    ]
    # Check for downloaded result directories like results_YYYYMMDD_HHMMSS
    for parent in [
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "eval_zero_anchor_img" / "outputs",
        REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "outputs",
    ]:
        if parent.exists():
            res_dirs = sorted([d for d in parent.iterdir() if d.is_dir() and d.name.startswith("results_")], reverse=True)
            candidates.extend(res_dirs)

    for cand in candidates:
        if (cand / "van_gogh").exists() and (cand / "random_artists").exists():
            return str(cand)
    return str(REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "outputs" / "concat_images_100")


def detect_default_output_dir(image_dir_str: str | None = None) -> str:
    img_dir = image_dir_str or detect_default_image_dir()
    is_100 = "100" in img_dir
    suffix = "eval_metrics_100" if is_100 else "eval_metrics"

    cand_img = REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "eval_zero_anchor_img" / "outputs"
    if cand_img.exists():
        return str(cand_img / suffix)
    return str(REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "outputs" / suffix)


def default_style_csv() -> str:
    eval_csv = REPO_ROOT / "data" / "style_100_retain_eval.csv"
    if eval_csv.exists():
        return str(eval_csv)
    return str(REPO_ROOT / "data" / "style.csv")


def slice_three_way_image(
    concat_img: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Slice a 3-way horizontally concatenated image into (Original, Art, Zero)."""
    w, h = concat_img.size
    if w % 3 != 0:
        raise ValueError(f"Image width ({w}) is not divisible by 3")
    slice_w = w // 3
    img_orig = concat_img.crop((0, 0, slice_w, h))
    img_art = concat_img.crop((slice_w, 0, 2 * slice_w, h))
    img_zero = concat_img.crop((2 * slice_w, 0, w, h))
    return img_orig, img_art, img_zero


def parse_args():
    detected_img_dir = detect_default_image_dir()
    parser = argparse.ArgumentParser(
        description="Quickly compute CLIP evaluation metrics comparing Art Anchor vs Zero Anchor from 3-way preview images",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=detected_img_dir,
        help="Path to directory containing van_gogh/ and random_artists/ concatenated images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=detect_default_output_dir(detected_img_dir),
        help="Directory to save JSON and CSV metric results",
    )
    parser.add_argument(
        "--style_csv",
        type=str,
        default=default_style_csv(),
        help="Path to style CSV for artist prompt reconstruction",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="auto",
        choices=["auto", "canonical_5", "legacy_painting"],
        help="Prompt reconstruction mode: 'auto' (detect from count/names), 'canonical_5', or 'legacy_painting'",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model ID for CLIP (default matches SD 1.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for random artist selection / deterministic prompt generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda', 'cpu', or 'mps')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for CLIP feature extraction",
    )
    return parser.parse_args()


class QuickClipEvaluator:
    def __init__(self, model_id: str, device: str):
        self.device = device
        print(f"Loading CLIP model '{model_id}' on {device}...")
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def cosine_similarity(feats1: torch.Tensor, feats2: torch.Tensor) -> torch.Tensor:
        return (feats1 * feats2).sum(dim=-1)


def evaluate_metrics(
    evaluator: QuickClipEvaluator,
    image_dir: Path,
    style_csv: Path,
    seed: int = 42,
    prompt_mode: str = "auto",
) -> tuple[dict, list[dict]]:
    vg_dir = image_dir / "van_gogh"
    random_dir = image_dir / "random_artists"

    if not vg_dir.exists():
        raise FileNotFoundError(f"Van Gogh images directory not found: {vg_dir}")
    if not random_dir.exists():
        raise FileNotFoundError(f"Random artists directory not found: {random_dir}")

    vg_files = sorted([f for f in vg_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    random_files = sorted([f for f in random_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])

    if not vg_files:
        raise ValueError(f"No image files found in {vg_dir}")
    if not random_files:
        raise ValueError(f"No image files found in {random_dir}")

    num_vg = len(vg_files)
    num_rnd = len(random_files)

    # Determine prompt mode
    mode = prompt_mode
    if mode == "auto":
        if num_vg >= 20 or (vg_files and any("style_of" in f.name or "in_the_style" in f.name for f in vg_files)):
            mode = "canonical_5"
        else:
            mode = "legacy_painting"

    print(f"Detected {num_vg} Van Gogh images, {num_rnd} Retain images (prompt_mode='{mode}').")

    # Reconstruct prompts
    if mode == "canonical_5":
        vg_items = build_van_gogh_100_prompts(num_vg, base_seed=seed)
        vg_prompts = [p for p, _ in vg_items]
        retain_items = build_retain_100_prompts(style_csv, num_rnd, base_seed=seed)
        artist_items = [(artist, prompt) for artist, prompt, _ in retain_items]
    else:
        vg_prompts = build_van_gogh_prompts(num_vg)
        sampled_artists = load_random_artists(style_csv, target="Van Gogh", count=num_rnd, seed=seed)
        artist_items = build_random_artist_prompts(sampled_artists)

    per_image_rows = []

    # -------------------------------------------------------------
    # 1. Target Evaluation (Van Gogh)
    # -------------------------------------------------------------
    print(f"\n[1/2] Evaluating Target Concept Erasure (Van Gogh, {num_vg} images)...")
    vg_concept_scores = {"orig": [], "art": [], "zero": []}
    vg_prompt_scores = {"orig": [], "art": [], "zero": []}

    vg_text_emb = evaluator.get_text_features(["Van Gogh"])[0]

    for idx, img_path in enumerate(vg_files):
        prompt = vg_prompts[idx % len(vg_prompts)]
        prompt_emb = evaluator.get_text_features([prompt])[0]

        with Image.open(img_path) as full_img:
            img_orig, img_art, img_zero = slice_three_way_image(full_img)

        img_embs = evaluator.get_image_features([img_orig, img_art, img_zero])
        emb_orig, emb_art, emb_zero = img_embs[0], img_embs[1], img_embs[2]

        c_orig = evaluator.cosine_similarity(emb_orig, vg_text_emb).item()
        c_art = evaluator.cosine_similarity(emb_art, vg_text_emb).item()
        c_zero = evaluator.cosine_similarity(emb_zero, vg_text_emb).item()

        p_orig = evaluator.cosine_similarity(emb_orig, prompt_emb).item()
        p_art = evaluator.cosine_similarity(emb_art, prompt_emb).item()
        p_zero = evaluator.cosine_similarity(emb_zero, prompt_emb).item()

        vg_concept_scores["orig"].append(c_orig)
        vg_concept_scores["art"].append(c_art)
        vg_concept_scores["zero"].append(c_zero)

        vg_prompt_scores["orig"].append(p_orig)
        vg_prompt_scores["art"].append(p_art)
        vg_prompt_scores["zero"].append(p_zero)

        per_image_rows.append({
            "category": "target",
            "index": idx + 1,
            "artist": "Van Gogh",
            "prompt": prompt,
            "file": img_path.name,
            "orig_concept_clip": c_orig,
            "art_concept_clip": c_art,
            "zero_concept_clip": c_zero,
            "orig_prompt_clip": p_orig,
            "art_prompt_clip": p_art,
            "zero_prompt_clip": p_zero,
            "art_image_fidelity_to_orig": evaluator.cosine_similarity(emb_art, emb_orig).item(),
            "zero_image_fidelity_to_orig": evaluator.cosine_similarity(emb_zero, emb_orig).item(),
        })

    # -------------------------------------------------------------
    # 2. Retain Evaluation (Artists)
    # -------------------------------------------------------------
    print(f"[2/2] Evaluating Retain Specificity & Fidelity ({num_rnd} images)...")
    ret_concept_scores = {"orig": [], "art": [], "zero": []}
    ret_prompt_scores = {"orig": [], "art": [], "zero": []}
    ret_image_fidelity = {"art": [], "zero": []}

    for idx, img_path in enumerate(random_files):
        artist, prompt = artist_items[idx % len(artist_items)]
        artist_emb = evaluator.get_text_features([artist])[0]
        prompt_emb = evaluator.get_text_features([prompt])[0]

        with Image.open(img_path) as full_img:
            img_orig, img_art, img_zero = slice_three_way_image(full_img)

        img_embs = evaluator.get_image_features([img_orig, img_art, img_zero])
        emb_orig, emb_art, emb_zero = img_embs[0], img_embs[1], img_embs[2]

        c_orig = evaluator.cosine_similarity(emb_orig, artist_emb).item()
        c_art = evaluator.cosine_similarity(emb_art, artist_emb).item()
        c_zero = evaluator.cosine_similarity(emb_zero, artist_emb).item()

        p_orig = evaluator.cosine_similarity(emb_orig, prompt_emb).item()
        p_art = evaluator.cosine_similarity(emb_art, prompt_emb).item()
        p_zero = evaluator.cosine_similarity(emb_zero, prompt_emb).item()

        f_art = evaluator.cosine_similarity(emb_art, emb_orig).item()
        f_zero = evaluator.cosine_similarity(emb_zero, emb_orig).item()

        ret_concept_scores["orig"].append(c_orig)
        ret_concept_scores["art"].append(c_art)
        ret_concept_scores["zero"].append(c_zero)

        ret_prompt_scores["orig"].append(p_orig)
        ret_prompt_scores["art"].append(p_art)
        ret_prompt_scores["zero"].append(p_zero)

        ret_image_fidelity["art"].append(f_art)
        ret_image_fidelity["zero"].append(f_zero)

        per_image_rows.append({
            "category": "retain",
            "index": idx + 1,
            "artist": artist,
            "prompt": prompt,
            "file": img_path.name,
            "orig_concept_clip": c_orig,
            "art_concept_clip": c_art,
            "zero_concept_clip": c_zero,
            "orig_prompt_clip": p_orig,
            "art_prompt_clip": p_art,
            "zero_prompt_clip": p_zero,
            "art_image_fidelity_to_orig": f_art,
            "zero_image_fidelity_to_orig": f_zero,
        })

    # -------------------------------------------------------------
    # Summary Aggregation
    # -------------------------------------------------------------
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "num_target_images": num_vg,
        "num_retain_images": num_rnd,
        "prompt_mode": mode,
        "target_erasure": {
            "concept_clip": {
                "original": _mean(vg_concept_scores["orig"]),
                "art_anchor": _mean(vg_concept_scores["art"]),
                "zero_anchor": _mean(vg_concept_scores["zero"]),
                "delta_zero_minus_art": _mean(vg_concept_scores["zero"]) - _mean(vg_concept_scores["art"]),
                "zero_erasure_rate_pct": (
                    (_mean(vg_concept_scores["orig"]) - _mean(vg_concept_scores["zero"]))
                    / max(_mean(vg_concept_scores["orig"]), 1e-8) * 100
                ),
                "art_erasure_rate_pct": (
                    (_mean(vg_concept_scores["orig"]) - _mean(vg_concept_scores["art"]))
                    / max(_mean(vg_concept_scores["orig"]), 1e-8) * 100
                ),
            },
            "prompt_clip": {
                "original": _mean(vg_prompt_scores["orig"]),
                "art_anchor": _mean(vg_prompt_scores["art"]),
                "zero_anchor": _mean(vg_prompt_scores["zero"]),
                "delta_zero_minus_art": _mean(vg_prompt_scores["zero"]) - _mean(vg_prompt_scores["art"]),
            },
        },
        "retain_preservation": {
            "artist_clip": {
                "original": _mean(ret_concept_scores["orig"]),
                "art_anchor": _mean(ret_concept_scores["art"]),
                "zero_anchor": _mean(ret_concept_scores["zero"]),
                "delta_zero_minus_art": _mean(ret_concept_scores["zero"]) - _mean(ret_concept_scores["art"]),
                "zero_preservation_ratio_pct": (
                    _mean(ret_concept_scores["zero"]) / max(_mean(ret_concept_scores["orig"]), 1e-8) * 100
                ),
                "art_preservation_ratio_pct": (
                    _mean(ret_concept_scores["art"]) / max(_mean(ret_concept_scores["orig"]), 1e-8) * 100
                ),
            },
            "prompt_clip": {
                "original": _mean(ret_prompt_scores["orig"]),
                "art_anchor": _mean(ret_prompt_scores["art"]),
                "zero_anchor": _mean(ret_prompt_scores["zero"]),
                "delta_zero_minus_art": _mean(ret_prompt_scores["zero"]) - _mean(ret_prompt_scores["art"]),
            },
            "image_fidelity_to_orig": {
                "art_anchor": _mean(ret_image_fidelity["art"]),
                "zero_anchor": _mean(ret_image_fidelity["zero"]),
                "delta_zero_minus_art": _mean(ret_image_fidelity["zero"]) - _mean(ret_image_fidelity["art"]),
            },
        },
    }

    return summary, per_image_rows


def format_summary_table(summary: dict) -> str:
    lines = []
    lines.append("=" * 88)
    num_t = summary.get("num_target_images", 10)
    num_r = summary.get("num_retain_images", 10)
    mode = summary.get("prompt_mode", "canonical_5")
    lines.append(f" Van Gogh Erasure Comparison: Art Anchor vs Zero Anchor ({num_t} Target, {num_r} Retain, mode={mode})")
    lines.append("=" * 88)
    lines.append(f"{'Category':<16} {'Metric':<28} {'Original':<10} {'Art Anchor':<12} {'Zero Anchor':<13} {'Delta (Zero-Art)'}")
    lines.append("-" * 88)

    # Target
    t_c = summary["target_erasure"]["concept_clip"]
    d_tc = t_c["delta_zero_minus_art"]
    winner_tc = "Zero (Better)" if d_tc < 0 else "Art (Better)"
    lines.append(f"{'Target Erasure':<16} {'Concept CLIP Score (↓)':<28} {t_c['original']:<10.4f} {t_c['art_anchor']:<12.4f} {t_c['zero_anchor']:<13.4f} {d_tc:+.4f} [{winner_tc}]")

    t_p = summary["target_erasure"]["prompt_clip"]
    d_tp = t_p["delta_zero_minus_art"]
    winner_tp = "Zero (Better)" if d_tp < 0 else "Art (Better)"
    lines.append(f"{'':<16} {'Prompt CLIP Score (↓)':<28} {t_p['original']:<10.4f} {t_p['art_anchor']:<12.4f} {t_p['zero_anchor']:<13.4f} {d_tp:+.4f} [{winner_tp}]")

    lines.append("-" * 88)

    # Retain
    r_c = summary["retain_preservation"]["artist_clip"]
    d_rc = r_c["delta_zero_minus_art"]
    winner_rc = "Zero (Better)" if d_rc > 0 else "Art (Better)"
    lines.append(f"{'Retain Style':<16} {'Artist CLIP Score (↑)':<28} {r_c['original']:<10.4f} {r_c['art_anchor']:<12.4f} {r_c['zero_anchor']:<13.4f} {d_rc:+.4f} [{winner_rc}]")

    r_p = summary["retain_preservation"]["prompt_clip"]
    d_rp = r_p["delta_zero_minus_art"]
    winner_rp = "Zero (Better)" if d_rp > 0 else "Art (Better)"
    lines.append(f"{'':<16} {'Prompt CLIP Score (↑)':<28} {r_p['original']:<10.4f} {r_p['art_anchor']:<12.4f} {r_p['zero_anchor']:<13.4f} {d_rp:+.4f} [{winner_rp}]")

    r_f = summary["retain_preservation"]["image_fidelity_to_orig"]
    d_rf = r_f["delta_zero_minus_art"]
    winner_rf = "Zero (Better)" if d_rf > 0 else "Art (Better)"
    lines.append(f"{'':<16} {'Image Fidelity to Orig (↑)':<28} {'1.0000':<10} {r_f['art_anchor']:<12.4f} {r_f['zero_anchor']:<13.4f} {d_rf:+.4f} [{winner_rf}]")

    lines.append("=" * 88)
    lines.append("Notes: (↓) = Lower is better (stronger concept erasure).")
    lines.append("       (↑) = Higher is better (greater preservation of retain style / original content).")
    lines.append("=" * 88)
    return "\n".join(lines)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style_csv = Path(args.style_csv)

    evaluator = QuickClipEvaluator(args.clip_model, args.device)
    summary, rows = evaluate_metrics(
        evaluator=evaluator,
        image_dir=image_dir,
        style_csv=style_csv,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
    )

    # Format and print table
    table_str = format_summary_table(summary)
    print("\n" + table_str)

    # Save summary JSON
    json_path = output_dir / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary: {json_path}")

    # Save table text
    txt_path = output_dir / "summary_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"Saved text table:      {txt_path}")

    # Save per-image CSV
    csv_path = output_dir / "per_image_metrics.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-image CSV:   {csv_path}")


if __name__ == "__main__":
    main()
