#!/usr/bin/env python3
"""Concatenate generated image versions that share a prompt and seed."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHOD_LABELS = {
    "original": "Original",
    "legacy": "Legacy SPEED",
    "target_global_pairwise_residual_subspace": "TGPRS-SPEED",
}
METHOD_ORDER = tuple(METHOD_LABELS)
SOURCE_ID_PATTERN = re.compile(r"_(\d+)\.png$")


def parse_downloaded_name(path):
    parts = path.name.split("__", 4)
    if len(parts) != 5:
        raise ValueError(f"Unexpected downloaded image name: {path.name}")
    benchmark, method, split, mode, source_name = parts
    match = SOURCE_ID_PATTERN.search(source_name)
    if not match:
        raise ValueError(f"Cannot read source ID from: {source_name}")
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown model in {path.name}: {method}")
    return {
        "path": path,
        "benchmark": benchmark,
        "method": method,
        "split": split,
        "mode": mode,
        "source_id": int(match.group(1)),
    }


def benchmark_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return {
            int(row["id"]): row
            for row in csv.DictReader(csv_file)
        }


def load_groups(input_dir, output_dir, data_dir):
    metadata_by_benchmark = {}
    groups = defaultdict(list)
    for path in sorted(input_dir.rglob("*.png")):
        if output_dir in path.parents:
            continue
        record = parse_downloaded_name(path)
        benchmark = record["benchmark"]
        if benchmark not in metadata_by_benchmark:
            csv_path = data_dir / f"{benchmark}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing benchmark CSV: {csv_path}")
            metadata_by_benchmark[benchmark] = benchmark_rows(csv_path)
        try:
            row = metadata_by_benchmark[benchmark][record["source_id"]]
        except KeyError as error:
            raise ValueError(
                f"ID {record['source_id']} is absent from {benchmark}.csv"
            ) from error
        record["prompt"] = row["text"]
        record["seed"] = int(row["seed"])
        groups[(record["prompt"], record["seed"])].append(record)
    return groups


def benchmark_sort_key(benchmark):
    try:
        return int(benchmark.split("_", 1)[0])
    except ValueError:
        return benchmark


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_centered(draw, box, text, font, fill="black"):
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def concatenate_group(records, output_path):
    benchmarks = sorted(
        {record["benchmark"] for record in records},
        key=benchmark_sort_key,
    )
    record_by_cell = {
        (record["benchmark"], record["method"]): record
        for record in records
    }
    expected_cells = {
        (benchmark, method)
        for benchmark in benchmarks
        for method in METHOD_ORDER
    }
    if set(record_by_cell) != expected_cells:
        missing = sorted(expected_cells - set(record_by_cell))
        raise ValueError(f"Incomplete model versions for {output_path.name}: {missing}")

    loaded = {}
    for cell, record in record_by_cell.items():
        with Image.open(record["path"]) as image:
            loaded[cell] = image.convert("RGB").copy()
    tile_width = max(image.width for image in loaded.values())
    image_height = max(image.height for image in loaded.values())
    title_height = 44
    label_height = 34
    padding = 8
    tile_height = label_height + image_height
    canvas = Image.new(
        "RGB",
        (
            padding * 2 + tile_width * len(METHOD_ORDER),
            title_height + padding * 2 + tile_height * len(benchmarks),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    label_font = load_font(16)
    prompt = records[0]["prompt"]
    seed = records[0]["seed"]
    draw_centered(
        draw,
        (0, 0, canvas.width, title_height),
        f"{prompt} | seed {seed}",
        title_font,
    )

    for row_index, benchmark in enumerate(benchmarks):
        row_top = title_height + padding + row_index * tile_height
        for column_index, method in enumerate(METHOD_ORDER):
            cell = (benchmark, method)
            image = loaded[cell]
            left = padding + column_index * tile_width
            draw_centered(
                draw,
                (left, row_top, left + tile_width, row_top + label_height),
                f"{benchmark} | {METHOD_LABELS[method]}",
                label_font,
            )
            image_left = left + (tile_width - image.width) // 2
            canvas.paste(image, (image_left, row_top + label_height))
    canvas.save(output_path)


def output_name(prompt, seed):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", prompt).strip("_").lower()
    return f"{slug}__seed_{seed}.png"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Group downloaded benchmark images by prompt and seed, then create "
            "a benchmark-by-model comparison grid."
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir / "concat"
    )
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(input_dir, output_dir, args.data_dir.resolve())
    if not groups:
        raise SystemExit(f"No PNG images found in {input_dir}")

    for (prompt, seed), records in sorted(groups.items()):
        output_path = output_dir / output_name(prompt, seed)
        concatenate_group(records, output_path)
        print(f"Saved {output_path} ({len(records)} source images)")
    print(f"Created {len(groups)} concatenated images in {output_dir}")


if __name__ == "__main__":
    main()
