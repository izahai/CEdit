#!/usr/bin/env python3
"""Evaluate few-concept and MS-COCO outputs with CLIP score and FID."""

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

import yaml

try:
    from .workflow_config import (
        METHOD_LEGACY,
        METHOD_ORIGINAL,
        METHOD_TGPRS,
        config_fingerprint,
        load_config,
        task_specs,
    )
except ImportError:
    from workflow_config import (
        METHOD_LEGACY,
        METHOD_ORIGINAL,
        METHOD_TGPRS,
        config_fingerprint,
        load_config,
        task_specs,
    )


DETAILED_FIELDS = [
    "task_id",
    "erase_type",
    "target_concepts",
    "target_count",
    "content",
    "content_role",
    "model",
    "anchor_mode",
    "anchor_concepts",
    "configured_residual_rank",
    "applied_residual_rank",
    "target_global_residual_count",
    "params",
    "aug_num",
    "threshold",
    "retain_scale",
    "seed",
    "inference_timesteps",
    "guidance_scale",
    "n_images",
    "clip_score",
    "fid_vs_original",
    "image_manifest_sha256",
    "run_fingerprint",
]

SUMMARY_FIELDS = [
    "task_id",
    "erase_type",
    "target_concepts",
    "target_count",
    "model",
    "anchor_mode",
    "anchor_concepts",
    "configured_residual_rank",
    "applied_residual_rank",
    "target_global_residual_count",
    "params",
    "aug_num",
    "threshold",
    "retain_scale",
    "target_n_contents",
    "target_n_images",
    "target_clip_score_mean",
    "target_fid_mean",
    "non_target_n_contents",
    "non_target_n_images",
    "non_target_clip_score_mean",
    "non_target_fid_mean",
    "mscoco_n_images",
    "mscoco_clip_score",
    "mscoco_fid_vs_original",
]

COMPARISON_FIELDS = [
    "task_id",
    "erase_type",
    "target_concepts",
    "target_count",
    "legacy_aug_num",
    "tgprs_aug_num",
    "tgprs_configured_residual_rank",
    "tgprs_applied_residual_rank",
    "original_target_clip_score",
    "legacy_target_clip_score",
    "tgprs_target_clip_score",
    "target_clip_score_improvement",
    "legacy_non_target_fid",
    "tgprs_non_target_fid",
    "non_target_fid_improvement",
    "original_mscoco_clip_score",
    "legacy_mscoco_clip_score",
    "tgprs_mscoco_clip_score",
    "mscoco_clip_score_improvement",
    "legacy_mscoco_fid",
    "tgprs_mscoco_fid",
    "mscoco_fid_improvement",
]


def sanitized_prompt(prompt):
    """Match sample.py's filename sanitization."""
    return re.sub(r"[^\w\s]", "", prompt).replace(", ", "_")


def few_prompt_records(prompt_templates, content, num_samples):
    records = []
    for template in prompt_templates:
        prompt = template.format(content)
        basename = sanitized_prompt(prompt)
        for sample_index in range(num_samples):
            filename = f"{basename}_{sample_index}.png"
            # This reproduces src/clip_score_cal.py's text reconstruction.
            clip_text = "_".join(filename.split("_")[:-1])
            records.append({"filename": filename, "prompt": clip_text})
    filenames = [record["filename"] for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Few-concept prompts produce duplicate filenames: {content}")
    return records


def load_mscoco_records(csv_path, num_prompts):
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))[:num_prompts]
    if len(rows) != num_prompts:
        raise ValueError(
            f"Expected {num_prompts} MS-COCO prompts, found {len(rows)}"
        )
    records = [
        {
            "filename": f"COCO_val2014_{int(row['image_id']):012}.png",
            "prompt": row["text"],
        }
        for row in rows
    ]
    filenames = [record["filename"] for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Selected MS-COCO prompts contain duplicate image IDs")
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
        raise ValueError(
            f"Invalid image set in {image_dir}: {', '.join(details)}"
        )
    digest = hashlib.sha256()
    for filename in sorted(expected):
        path = image_dir / filename
        stat = path.stat()
        digest.update(filename.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def few_image_dir(output_root, method, task, content):
    output_root = Path(output_root)
    if method == METHOD_ORIGINAL:
        return (
            output_root
            / "images"
            / METHOD_ORIGINAL
            / task["erase_type"]
            / "shared"
            / content
            / "original"
        )
    return (
        output_root
        / "images"
        / method
        / task["erase_type"]
        / task["id"]
        / content
        / "edit"
    )


def mscoco_image_dir(output_root, method, task_id):
    output_root = Path(output_root)
    if method == METHOD_ORIGINAL:
        return output_root / "mscoco" / METHOD_ORIGINAL / "coco" / "original"
    return output_root / "mscoco" / method / task_id / "coco" / "edit"


def atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_existing_rows(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    result = {}
    for row in rows:
        key = (row["task_id"], row["model"], row["content"])
        if key in result:
            raise ValueError(f"Duplicate cached metric row: {key}")
        result[key] = row
    return result


def load_training_configs(workflow_dir):
    paths = {
        METHOD_LEGACY: Path(workflow_dir) / "train_config_legacy.yaml",
        METHOD_TGPRS: (
            Path(workflow_dir)
            / "train_config_target_global_pairwise_residual_subspace.yaml"
        ),
    }
    configs = {}
    for method, path in paths.items():
        with path.open(encoding="utf-8") as config_file:
            configs[method] = yaml.safe_load(config_file)
    return configs


def training_metadata(method, task, train_configs):
    if method == METHOD_ORIGINAL:
        return {
            "anchor_mode": "",
            "anchor_concepts": "",
            "configured_residual_rank": "",
            "applied_residual_rank": "",
            "target_global_residual_count": "",
            "params": "",
            "aug_num": "",
            "threshold": "",
            "retain_scale": "",
        }
    config = train_configs[method]
    is_tgprs = method == METHOD_TGPRS
    return {
        "anchor_mode": config["anchor_mode"],
        "anchor_concepts": task["anchor_concept"],
        "configured_residual_rank": (
            task["configured_residual_rank"] if is_tgprs else ""
        ),
        "applied_residual_rank": (
            task["applied_residual_rank"] if is_tgprs else ""
        ),
        "target_global_residual_count": (
            task["target_global_residual_count"] if is_tgprs else ""
        ),
        "params": config["params"],
        "aug_num": config["aug_num"],
        "threshold": config["threshold"],
        "retain_scale": config["retain_scale"],
    }


class ClipScorer:
    def __init__(self, model_name, device):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, image_dir, records, batch_size):
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
            image_inputs = self.processor(images=images, return_tensors="pt")
            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            image_inputs = {
                key: value.to(self.device) for key, value in image_inputs.items()
            }
            text_inputs = {
                key: value.to(self.device) for key, value in text_inputs.items()
            }
            with self.torch.inference_mode():
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)
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


def fid_score(
    edited_dir,
    original_dir,
    original_manifest,
    cache_root,
    batch_size,
    feature_layer,
    use_cuda,
):
    import torch_fidelity

    # torch-fidelity 0.3 stores trusted local NumPy statistics that PyTorch
    # 2.6+ cannot reopen with its new weights_only=True default.
    variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(variable)
    os.environ[variable] = "1"
    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=str(edited_dir),
            input2=str(original_dir),
            input2_cache_name=(
                f"eval-few-reference-{feature_layer}-{original_manifest}"
            ),
            cache_root=str(cache_root),
            cache=True,
            cuda=use_cuda,
            batch_size=batch_size,
            feature_layer_fid=feature_layer,
            fid=True,
            isc=False,
            kid=False,
            prc=False,
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    return float(metrics["frechet_inception_distance"])


def _float(row, key):
    return float(row[key])


def summarize_detailed_rows(rows, tasks, methods):
    by_group = {}
    for row in rows:
        by_group.setdefault((row["task_id"], row["model"]), []).append(row)

    summaries = []
    for task in tasks:
        for method in methods:
            key = (task["id"], method)
            group = by_group.get(key, [])
            target_rows = [row for row in group if row["content_role"] == "target"]
            non_target_rows = [
                row for row in group if row["content_role"] == "non_target"
            ]
            coco_rows = [row for row in group if row["content_role"] == "coco"]
            if not target_rows or not non_target_rows or len(coco_rows) != 1:
                raise ValueError(
                    f"Incomplete metric group {key}: target={len(target_rows)}, "
                    f"non_target={len(non_target_rows)}, coco={len(coco_rows)}"
                )
            first = group[0]
            coco = coco_rows[0]
            summaries.append({
                "task_id": task["id"],
                "erase_type": task["erase_type"],
                "target_concepts": ", ".join(task["target_concepts"]),
                "target_count": task["target_count"],
                "model": method,
                "anchor_mode": first["anchor_mode"],
                "anchor_concepts": first["anchor_concepts"],
                "configured_residual_rank": first["configured_residual_rank"],
                "applied_residual_rank": first["applied_residual_rank"],
                "target_global_residual_count": first[
                    "target_global_residual_count"
                ],
                "params": first["params"],
                "aug_num": first["aug_num"],
                "threshold": first["threshold"],
                "retain_scale": first["retain_scale"],
                "target_n_contents": len(target_rows),
                "target_n_images": sum(int(row["n_images"]) for row in target_rows),
                "target_clip_score_mean": statistics.fmean(
                    _float(row, "clip_score") for row in target_rows
                ),
                "target_fid_mean": statistics.fmean(
                    _float(row, "fid_vs_original") for row in target_rows
                ),
                "non_target_n_contents": len(non_target_rows),
                "non_target_n_images": sum(
                    int(row["n_images"]) for row in non_target_rows
                ),
                "non_target_clip_score_mean": statistics.fmean(
                    _float(row, "clip_score") for row in non_target_rows
                ),
                "non_target_fid_mean": statistics.fmean(
                    _float(row, "fid_vs_original") for row in non_target_rows
                ),
                "mscoco_n_images": int(coco["n_images"]),
                "mscoco_clip_score": _float(coco, "clip_score"),
                "mscoco_fid_vs_original": _float(coco, "fid_vs_original"),
            })
    return summaries


def build_comparison_rows(summary_rows, tasks):
    lookup = {
        (row["task_id"], row["model"]): row for row in summary_rows
    }
    comparisons = []
    for task in tasks:
        original = lookup[(task["id"], METHOD_ORIGINAL)]
        legacy = lookup[(task["id"], METHOD_LEGACY)]
        tgprs = lookup[(task["id"], METHOD_TGPRS)]
        comparisons.append({
            "task_id": task["id"],
            "erase_type": task["erase_type"],
            "target_concepts": ", ".join(task["target_concepts"]),
            "target_count": task["target_count"],
            "legacy_aug_num": legacy["aug_num"],
            "tgprs_aug_num": tgprs["aug_num"],
            "tgprs_configured_residual_rank": tgprs[
                "configured_residual_rank"
            ],
            "tgprs_applied_residual_rank": tgprs["applied_residual_rank"],
            "original_target_clip_score": original["target_clip_score_mean"],
            "legacy_target_clip_score": legacy["target_clip_score_mean"],
            "tgprs_target_clip_score": tgprs["target_clip_score_mean"],
            "target_clip_score_improvement": (
                legacy["target_clip_score_mean"]
                - tgprs["target_clip_score_mean"]
            ),
            "legacy_non_target_fid": legacy["non_target_fid_mean"],
            "tgprs_non_target_fid": tgprs["non_target_fid_mean"],
            "non_target_fid_improvement": (
                legacy["non_target_fid_mean"] - tgprs["non_target_fid_mean"]
            ),
            "original_mscoco_clip_score": original["mscoco_clip_score"],
            "legacy_mscoco_clip_score": legacy["mscoco_clip_score"],
            "tgprs_mscoco_clip_score": tgprs["mscoco_clip_score"],
            "mscoco_clip_score_improvement": (
                tgprs["mscoco_clip_score"] - legacy["mscoco_clip_score"]
            ),
            "legacy_mscoco_fid": legacy["mscoco_fid_vs_original"],
            "tgprs_mscoco_fid": tgprs["mscoco_fid_vs_original"],
            "mscoco_fid_improvement": (
                legacy["mscoco_fid_vs_original"]
                - tgprs["mscoco_fid_vs_original"]
            ),
        })
    return comparisons


def runtime_fingerprint(config, args):
    values = {
        "workflow": config_fingerprint(config),
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "mscoco_num_prompts": args.mscoco_num_prompts,
        "seed": args.seed,
        "inference_timesteps": args.inference_timesteps,
        "guidance_scale": args.guidance_scale,
        "clip_model": args.clip_model,
        "fid_feature_layer": args.fid_feature_layer,
    }
    encoded = json.dumps(values, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate(args):
    config = load_config(args.config)
    tasks = task_specs(config)
    methods = list(config["experiment"]["methods"])
    train_configs = load_training_configs(Path(args.config).parent)
    run_hash = runtime_fingerprint(config, args)
    detailed_path = Path(args.metrics_dir) / "detailed_metrics.csv"
    summary_path = Path(args.metrics_dir) / "summary.csv"
    comparison_path = Path(args.metrics_dir) / "comparison.csv"
    existing = {} if args.force else load_existing_rows(detailed_path)
    result_rows = []
    metric_memory = {}
    clip_scorer = None

    coco_records = load_mscoco_records(
        Path(args.repo_root) / "data" / "mscoco.csv",
        args.mscoco_num_prompts,
    )

    def get_metrics(image_dir, reference_dir, records, manifest, reference_manifest):
        nonlocal clip_scorer
        memory_key = (str(image_dir), manifest, tuple(
            (record["filename"], record["prompt"]) for record in records
        ))
        if memory_key in metric_memory:
            return metric_memory[memory_key]
        if clip_scorer is None:
            clip_scorer = ClipScorer(args.clip_model, args.device)
        clip = clip_scorer(image_dir, records, args.clip_batch_size)
        fid = 0.0
        if Path(image_dir) != Path(reference_dir):
            fid = fid_score(
                image_dir,
                reference_dir,
                reference_manifest,
                args.fid_cache_root,
                args.fid_batch_size,
                args.fid_feature_layer,
                args.device == "cuda",
            )
        metric_memory[memory_key] = (clip, fid)
        return clip, fid

    for task in tasks:
        target_names = {name.casefold() for name in task["target_concepts"]}
        few_records_by_content = {
            content: few_prompt_records(
                task["prompt_templates"],
                content,
                args.num_samples_per_prompt,
            )
            for content in task["contents"]
        }
        for method in methods:
            metadata = training_metadata(method, task, train_configs)
            for content in task["contents"] + ["coco"]:
                key = (task["id"], method, content)
                if content == "coco":
                    records = coco_records
                    reference_dir = mscoco_image_dir(
                        args.output_root, METHOD_ORIGINAL, "shared"
                    )
                    image_dir = mscoco_image_dir(
                        args.output_root, method, task["id"]
                    )
                    role = "coco"
                else:
                    records = few_records_by_content[content]
                    reference_dir = few_image_dir(
                        args.output_root, METHOD_ORIGINAL, task, content
                    )
                    image_dir = few_image_dir(
                        args.output_root, method, task, content
                    )
                    role = (
                        "target"
                        if content.casefold() in target_names
                        else "non_target"
                    )

                reference_manifest = validate_image_directory(
                    reference_dir, records
                )
                manifest = validate_image_directory(image_dir, records)
                cached = existing.get(key)
                if (
                    cached
                    and cached.get("run_fingerprint") == run_hash
                    and cached.get("image_manifest_sha256") == manifest
                    and int(cached.get("n_images", 0)) == len(records)
                ):
                    row = cached
                    print(f"Cached metrics: {task['id']} {method} {content}")
                else:
                    clip, fid = get_metrics(
                        image_dir,
                        reference_dir,
                        records,
                        manifest,
                        reference_manifest,
                    )
                    row = {
                        "task_id": task["id"],
                        "erase_type": task["erase_type"],
                        "target_concepts": ", ".join(task["target_concepts"]),
                        "target_count": task["target_count"],
                        "content": content,
                        "content_role": role,
                        "model": method,
                        **metadata,
                        "seed": args.seed,
                        "inference_timesteps": args.inference_timesteps,
                        "guidance_scale": args.guidance_scale,
                        "n_images": len(records),
                        "clip_score": clip,
                        "fid_vs_original": fid,
                        "image_manifest_sha256": manifest,
                        "run_fingerprint": run_hash,
                    }
                    print(
                        f"{task['id']} {method} {content}: "
                        f"CLIP={clip:.6f}, FID={fid:.6f}",
                        flush=True,
                    )
                result_rows.append(row)
                atomic_write_csv(detailed_path, result_rows, DETAILED_FIELDS)

    expected_rows = len(methods) * sum(
        len(task["contents"]) + 1 for task in tasks
    )
    if len(result_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} detailed rows, found {len(result_rows)}"
        )
    summary_rows = summarize_detailed_rows(result_rows, tasks, methods)
    comparison_rows = build_comparison_rows(summary_rows, tasks)
    atomic_write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    atomic_write_csv(comparison_path, comparison_rows, COMPARISON_FIELDS)
    print(f"Saved: {detailed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {comparison_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--fid-cache-root", type=Path, required=True)
    parser.add_argument("--num-samples-per-prompt", type=int, required=True)
    parser.add_argument("--mscoco-num-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inference-timesteps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--clip-batch-size", type=int, required=True)
    parser.add_argument("--fid-batch-size", type=int, required=True)
    parser.add_argument(
        "--fid-feature-layer",
        choices=("64", "192", "768", "2048"),
        required=True,
    )
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in (
        "num_samples_per_prompt",
        "mscoco_num_prompts",
        "inference_timesteps",
        "guidance_scale",
        "clip_batch_size",
        "fid_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
