#!/usr/bin/env python3
"""Compute paper-style MS-COCO CLIP score and FID for each checkpoint."""

import argparse
import csv
from pathlib import Path


def load_prompt_records(csv_path, num_prompts):
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))[:num_prompts]
    if len(rows) != num_prompts:
        raise ValueError(
            f"Expected {num_prompts} MS-COCO prompts, found {len(rows)}"
        )

    records = []
    for row in rows:
        image_id = int(row["image_id"])
        records.append({
            "filename": f"COCO_val2014_{image_id:012}.png",
            "prompt": row["text"],
        })
    filenames = [record["filename"] for record in records]
    if len(set(filenames)) != len(filenames):
        raise ValueError("The first MS-COCO prompts contain duplicate image IDs")
    return records


def validate_image_directory(image_dir, records):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    expected = {record["filename"] for record in records}
    actual = {path.name for path in image_dir.glob("*.png")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {len(missing)} (first: {missing[0]})")
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} (first: {unexpected[0]})"
            )
        raise ValueError(f"Invalid image set in {image_dir}: {', '.join(details)}")


def image_dir_for_run(image_root, method, benchmark):
    image_root = Path(image_root)
    if method == "original":
        return image_root / "original" / "coco" / "original"
    return image_root / method / benchmark / "coco" / "edit"


def clip_score(model, processor, image_dir, records, batch_size, device):
    import torch
    from PIL import Image

    total = 0.0
    count = 0
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        images = []
        for record in batch:
            with Image.open(Path(image_dir) / record["filename"]) as image:
                images.append(image.convert("RGB").copy())
        prompts = [record["prompt"] for record in batch]
        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        image_inputs = {
            key: value.to(device) for key, value in image_inputs.items()
        }
        text_inputs = {
            key: value.to(device) for key, value in text_inputs.items()
        }
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
            similarities = (image_features * text_features).sum(dim=-1)
        total += similarities.sum().item()
        count += len(batch)
    return 100.0 * total / count


def fid_score(original_dir, edited_dir, use_cuda):
    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=str(edited_dir),
        input2=str(original_dir),
        cuda=use_cuda,
        fid=True,
        isc=False,
        kid=False,
        prc=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def write_rows(output_csv, rows):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    fieldnames = [
        "benchmark",
        "model",
        "mscoco_n_images",
        "mscoco_clip_score",
        "mscoco_fid_vs_original",
    ]
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscoco-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--clip-model", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if "original" not in args.methods:
        raise SystemExit("The method list must include original as the FID reference")
    if args.num_prompts <= 0 or args.batch_size <= 0:
        raise SystemExit("num-prompts and batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    records = load_prompt_records(args.mscoco_csv, args.num_prompts)
    run_directories = {}
    for method in args.methods:
        benchmarks = args.benchmarks if method != "original" else ["shared"]
        for benchmark in benchmarks:
            image_dir = image_dir_for_run(args.image_root, method, benchmark)
            validate_image_directory(image_dir, records)
            run_directories[(method, benchmark)] = image_dir

    model = CLIPModel.from_pretrained(args.clip_model).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    original_dir = run_directories[("original", "shared")]
    original_clip_score = clip_score(
        model, processor, original_dir, records, args.batch_size, args.device
    )

    rows = []
    for method in args.methods:
        for benchmark in args.benchmarks:
            if method == "original":
                score = original_clip_score
                fid = 0.0
            else:
                image_dir = run_directories[(method, benchmark)]
                score = clip_score(
                    model,
                    processor,
                    image_dir,
                    records,
                    args.batch_size,
                    args.device,
                )
                fid = fid_score(original_dir, image_dir, args.device == "cuda")
            row = {
                "benchmark": benchmark,
                "model": method,
                "mscoco_n_images": args.num_prompts,
                "mscoco_clip_score": score,
                "mscoco_fid_vs_original": fid,
            }
            rows.append(row)
            print(
                f"{method} {benchmark}: CLIP={score:.6f}, FID={fid:.6f}",
                flush=True,
            )
    write_rows(args.output_csv, rows)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
