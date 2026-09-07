#!/usr/bin/env python3
"""Package downloaded few-style evaluation samples into a reviewable report."""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.template import painting_templates


TASKS = (("van_gogh", "Van Gogh"), ("picasso", "Picasso"), ("monet", "Monet"))
CONTENTS = ("Van Gogh", "Picasso", "Monet", "Paul Gauguin", "Caravaggio")
METHODS = ("original", "legacy", "target_global_pairwise_residual_subspace")
METHOD_LABELS = {"original": "SD v1.4", "legacy": "SPEED", "target_global_pairwise_residual_subspace": "Ours"}


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sanitized_prompt(prompt):
    return re.sub(r"[^\w\s]", "", prompt).replace(", ", "_")


def copy_sample(source, standalone, name):
    destination = standalone / name
    shutil.copy2(source, destination)
    return destination.name


def add_manifest(rows, index, category, task, content, prompt_index, prompt, model, source, local):
    rows.append({
        "sample_index": index,
        "category": category,
        "task": task,
        "content": content,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": 0,
        "model": model,
        "source_relpath": source.as_posix(),
        "local_path": f"standalone/{local}",
    })


def make_style_sheet(path, task_label, samples):
    thumb = 96
    label_width = 260
    cell_width = thumb * 3 + 12
    row_height = 108
    header_height = 54
    canvas = Image.new("RGB", (label_width + cell_width * len(CONTENTS), header_height + row_height * 30), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 7), f"Erase {task_label}: seed 0; each cell is SD v1.4 | SPEED | Ours", fill="black", font=font)
    for column, content in enumerate(CONTENTS):
        draw.text((label_width + column * cell_width + 4, 31), content, fill="black", font=font)
    for prompt_index in range(30):
        y = header_height + prompt_index * row_height
        draw.text((8, y + 4), f"p{prompt_index:02d} {painting_templates[prompt_index][:38]}", fill="black", font=font)
        for column, content in enumerate(CONTENTS):
            for method_index, method in enumerate(METHODS):
                image_path = samples[(content, prompt_index, method)]
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((thumb, thumb))
                    x = label_width + column * cell_width + method_index * thumb
                    canvas.paste(image, (x, y))
        draw.line((0, y + row_height - 1, canvas.width, y + row_height - 1), fill="#dddddd")
    canvas.save(path, optimize=True)


def make_coco_sheet(path, samples, coco_ids):
    thumb = 112
    label_width = 125
    row_height = 124
    columns = [("original", "shared", "SD v1.4")]
    for task_id, task_label in TASKS:
        columns.append(("legacy", task_id, f"SPEED\n{task_label}"))
        columns.append(("target_global_pairwise_residual_subspace", task_id, f"Ours\n{task_label}"))
    canvas = Image.new("RGB", (label_width + thumb * len(columns), 58 + row_height * len(coco_ids)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), "Matched MS-COCO preservation samples", fill="black", font=font)
    for column, (_, _, label) in enumerate(columns):
        draw.multiline_text((label_width + column * thumb + 3, 30), label, fill="black", font=font, spacing=1)
    for row, coco_id in enumerate(coco_ids):
        y = 58 + row * row_height
        draw.text((8, y + 5), coco_id.replace("COCO_val2014_", ""), fill="black", font=font)
        for column, (method, task_id, _) in enumerate(columns):
            with Image.open(samples[(coco_id, method, task_id)]) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb, thumb))
                canvas.paste(image, (label_width + column * thumb, y))
        draw.line((0, y + row_height - 1, canvas.width, y + row_height - 1), fill="#dddddd")
    canvas.save(path, optimize=True)


def build(args):
    output = args.output.resolve()
    staging = output / ".staging"
    standalone = output / "standalone"
    standalone.mkdir(parents=True, exist_ok=True)
    manifest = []
    local_index = 1
    originals = {}
    task_samples = {}

    for content in CONTENTS:
        for prompt_index, template in enumerate(painting_templates):
            prompt = template.format(content)
            basename = f"{sanitized_prompt(prompt)}_0.png"
            source = staging / "images/original/style/shared" / content / "original" / basename
            name = f"{local_index:04d}_style_shared_{slug(content)}_p{prompt_index:02d}_s00_original.png"
            local = copy_sample(source, standalone, name)
            originals[(content, prompt_index)] = standalone / local
            add_manifest(manifest, local_index, "style", "shared", content, prompt_index, prompt, "original", source.relative_to(staging), local)
            local_index += 1

    for task_id, task_label in TASKS:
        task_samples[task_id] = {}
        for content in CONTENTS:
            for prompt_index, template in enumerate(painting_templates):
                prompt = template.format(content)
                basename = f"{sanitized_prompt(prompt)}_0.png"
                task_samples[task_id][(content, prompt_index, "original")] = originals[(content, prompt_index)]
                for method in METHODS[1:]:
                    source = staging / f"images/{method}/style" / task_id / content / "edit" / basename
                    name = f"{local_index:04d}_style_{task_id}_{slug(content)}_p{prompt_index:02d}_s00_{method}.png"
                    local = copy_sample(source, standalone, name)
                    task_samples[task_id][(content, prompt_index, method)] = standalone / local
                    add_manifest(manifest, local_index, "style", task_label, content, prompt_index, prompt, method, source.relative_to(staging), local)
                    local_index += 1

    coco_ids = sorted(path.stem for path in (staging / "mscoco/original/coco/original").glob("*.png"))
    coco_samples = {}
    for coco_id in coco_ids:
        source = staging / "mscoco/original/coco/original" / f"{coco_id}.png"
        name = f"{local_index:04d}_coco_shared_original_{coco_id}.png"
        local = copy_sample(source, standalone, name)
        coco_samples[(coco_id, "original", "shared")] = standalone / local
        add_manifest(manifest, local_index, "coco", "shared", "coco", "", coco_id, "original", source.relative_to(staging), local)
        local_index += 1
        for task_id, task_label in TASKS:
            for method in METHODS[1:]:
                source = staging / f"mscoco/{method}" / task_id / "coco/edit" / f"{coco_id}.png"
                name = f"{local_index:04d}_coco_{task_id}_{method}_{coco_id}.png"
                local = copy_sample(source, standalone, name)
                coco_samples[(coco_id, method, task_id)] = standalone / local
                add_manifest(manifest, local_index, "coco", task_label, "coco", "", coco_id, method, source.relative_to(staging), local)
                local_index += 1

    with (output / "sample_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    for task_id, task_label in TASKS:
        make_style_sheet(output / f"contact_sheet_{task_id}.jpg", task_label, task_samples[task_id])
    make_coco_sheet(output / "contact_sheet_mscoco.jpg", coco_samples, coco_ids)

    lines = [
        "# Full few-style run: exhaustive seed-0 comparison",
        "",
        f"Server: `root@{args.host}`, SSH port `{args.port}`  ",
        "Run: SPEED threshold `0.1`, `aug_num=10`; Ours TGPRS threshold `0.3`, `aug_num=0`, residual rank 30.",
        "",
        f"**Standalone images:** `{len(manifest)}` PNG files in [`standalone/`](standalone/).",
        "",
        "Coverage: all 3 erase tasks, all 5 evaluated artists, all 30 style templates at seed 0, and 10 matched MS-COCO IDs across the original plus all 6 edited branches. Shared originals are stored once, so the 450 style comparisons reference 150 original and 900 edited PNGs.",
        "",
        "## Contact sheets",
        "",
        "Each style cell is ordered SD v1.4 | SPEED | Ours.",
        "",
    ]
    for task_id, task_label in TASKS:
        lines.extend((f"### Erase {task_label}", "", f"![Erase {task_label} contact sheet](contact_sheet_{task_id}.jpg)", ""))
    lines.extend(("### MS-COCO preservation", "", "![MS-COCO preservation contact sheet](contact_sheet_mscoco.jpg)", "", "## Complete paired style tables", ""))
    for task_id, task_label in TASKS:
        lines.extend((f"### Erase {task_label}", ""))
        for content in CONTENTS:
            lines.extend((f"#### Content: {content}", "", "| Prompt | SD v1.4 | SPEED | Ours |", "|---|---|---|---|"))
            for prompt_index, template in enumerate(painting_templates):
                prompt = template.format(content)
                paths = [task_samples[task_id][(content, prompt_index, method)].relative_to(output).as_posix() for method in METHODS]
                images = [f'<img src="{path}" width="180" alt="{task_label} {content} p{prompt_index:02d} {METHOD_LABELS[method]}">' for path, method in zip(paths, METHODS)]
                lines.append(f"| p{prompt_index:02d}: {prompt} | {images[0]} | {images[1]} | {images[2]} |")
            lines.append("")
    lines.extend(("## Matched MS-COCO samples", "", "| ID | SD v1.4 | SPEED / Ours for each erase task |", "|---|---|---|"))
    for coco_id in coco_ids:
        original_path = coco_samples[(coco_id, "original", "shared")].relative_to(output).as_posix()
        edits = []
        for task_id, task_label in TASKS:
            for method in METHODS[1:]:
                path = coco_samples[(coco_id, method, task_id)].relative_to(output).as_posix()
                edits.append(f'{task_label} {METHOD_LABELS[method]}<br><img src="{path}" width="150" alt="{coco_id} {task_label} {METHOD_LABELS[method]}">')
        lines.append(f'| {coco_id} | <img src="{original_path}" width="150" alt="{coco_id} original"> | {" ".join(edits)} |')
    (output / "comparison_table.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--host", default="180.189.55.43")
    parser.add_argument("--port", default="57595")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
