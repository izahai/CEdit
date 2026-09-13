#!/usr/bin/env python3
"""Evaluate the Van Gogh four-model comparison with CLIP score and FID."""

import argparse
import csv
import hashlib
import html
import json
import statistics
import sys
from pathlib import Path


REPO_ROOT_FROM_FILE = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FROM_FILE))

from remote_scripts.eval_few.eval_few_style_1.evaluate_clip_fid import (  # noqa: E402
    ClipScorer,
    atomic_write_csv,
    fid_score,
    few_prompt_records,
    load_mscoco_records,
    validate_image_directory,
)
from exp_results.val_clip_adv.workflow_config import (  # noqa: E402
    METHOD_LEARNED,
    METHOD_LEGACY,
    METHOD_ORIGINAL,
    METHOD_TGPRS,
    config_fingerprint,
    load_config,
    load_train_configs,
)


DETAILED_FIELDS = [
    "task_id",
    "target_concept",
    "content",
    "content_role",
    "model",
    "anchor_source",
    "anchor_mode",
    "anchor_concepts",
    "subspace_anchor_count",
    "configured_residual_rank",
    "applied_residual_rank",
    "params",
    "aug_num",
    "threshold",
    "retain_scale",
    "learned_prefix_token_ids",
    "learned_decoded_prefix",
    "anchor_search_best_step",
    "anchor_search_validation_clip_similarity",
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
    "target_concept",
    "model",
    "anchor_source",
    "anchor_mode",
    "anchor_concepts",
    "subspace_anchor_count",
    "configured_residual_rank",
    "applied_residual_rank",
    "params",
    "aug_num",
    "threshold",
    "retain_scale",
    "learned_prefix_token_ids",
    "learned_decoded_prefix",
    "anchor_search_best_step",
    "anchor_search_validation_clip_similarity",
    "target_n_images",
    "target_clip_score",
    "target_fid_vs_original",
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
    "comparison_method",
    "legacy_target_clip_score",
    "method_target_clip_score",
    "target_clip_score_improvement",
    "legacy_target_fid",
    "method_target_fid",
    "target_fid_delta",
    "legacy_non_target_clip_score",
    "method_non_target_clip_score",
    "non_target_clip_score_delta",
    "legacy_non_target_fid",
    "method_non_target_fid",
    "non_target_fid_improvement",
    "legacy_mscoco_clip_score",
    "method_mscoco_clip_score",
    "mscoco_clip_score_improvement",
    "legacy_mscoco_fid",
    "method_mscoco_fid",
    "mscoco_fid_improvement",
]


def few_image_dir(output_root, method, task, content):
    root = Path(output_root)
    if method == METHOD_ORIGINAL:
        return root / "images" / method / task["erase_type"] / "shared" / content / "original"
    return root / "images" / method / task["erase_type"] / task["id"] / content / "edit"


def mscoco_image_dir(output_root, method, task_id):
    root = Path(output_root)
    if method == METHOD_ORIGINAL:
        return root / "mscoco" / method / "coco" / "original"
    return root / "mscoco" / method / task_id / "coco" / "edit"


def load_learned_anchor_metadata(output_root, task):
    path = Path(output_root) / "learned_anchors" / task["id"] / "manifest.json"
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("status") != "complete":
        raise ValueError("Learned-anchor artifact is incomplete")
    if manifest.get("target_order") != [task["target_concept"]]:
        raise ValueError("Learned-anchor target does not match the evaluation task")
    entries = manifest.get("targets")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Learned-anchor manifest must contain exactly one target")
    entry = entries[0]
    required = (
        "prefix_token_ids",
        "decoded_prefix",
        "best_step",
        "best_validation_clip_similarity",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"Learned-anchor manifest is missing: {missing}")
    return {
        "prefix_token_ids": entry["prefix_token_ids"],
        "decoded_prefix": entry["decoded_prefix"],
        "best_step": entry["best_step"],
        "best_validation_clip_similarity": entry[
            "best_validation_clip_similarity"
        ],
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def training_metadata(method, task, train_configs, learned_metadata):
    blank = {
        "anchor_source": "",
        "anchor_mode": "",
        "anchor_concepts": "",
        "subspace_anchor_count": "",
        "configured_residual_rank": "",
        "applied_residual_rank": "",
        "params": "",
        "aug_num": "",
        "threshold": "",
        "retain_scale": "",
        "learned_prefix_token_ids": "",
        "learned_decoded_prefix": "",
        "anchor_search_best_step": "",
        "anchor_search_validation_clip_similarity": "",
    }
    if method == METHOD_ORIGINAL:
        return blank

    config = train_configs[method]
    metadata = {
        **blank,
        "anchor_source": config["anchor_source"],
        "anchor_mode": config["anchor_mode"],
        "anchor_concepts": (
            "learned artifact" if method == METHOD_LEARNED else task["anchor_concept"]
        ),
        "params": config["params"],
        "aug_num": config["aug_num"],
        "threshold": config["threshold"],
        "retain_scale": config["retain_scale"],
    }
    if method == METHOD_TGPRS:
        anchors = config["subspace_anchor_concepts"]
        metadata.update({
            "subspace_anchor_count": len(anchors),
            "configured_residual_rank": config["residual_rank"],
            "applied_residual_rank": min(config["residual_rank"], len(anchors)),
        })
    elif method == METHOD_LEARNED:
        metadata.update({
            "learned_prefix_token_ids": json.dumps(
                learned_metadata["prefix_token_ids"]
            ),
            "learned_decoded_prefix": learned_metadata["decoded_prefix"],
            "anchor_search_best_step": learned_metadata["best_step"],
            "anchor_search_validation_clip_similarity": learned_metadata[
                "best_validation_clip_similarity"
            ],
        })
    return metadata


def load_existing_rows(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    indexed = {}
    for row in rows:
        key = (row["task_id"], row["model"], row["content"])
        if key in indexed:
            raise ValueError(f"Duplicate cached metric row: {key}")
        indexed[key] = row
    return indexed


def _float(row, field):
    return float(row[field])


def summarize_detailed_rows(rows, task, methods):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    summaries = []
    for method in methods:
        group = grouped.get(method, [])
        target_rows = [row for row in group if row["content_role"] == "target"]
        non_target_rows = [
            row for row in group if row["content_role"] == "non_target"
        ]
        coco_rows = [row for row in group if row["content_role"] == "coco"]
        if len(target_rows) != 1 or len(non_target_rows) != 4 or len(coco_rows) != 1:
            raise ValueError(
                f"Incomplete metric group for {method}: target={len(target_rows)}, "
                f"non-target={len(non_target_rows)}, coco={len(coco_rows)}"
            )
        first = group[0]
        target = target_rows[0]
        coco = coco_rows[0]
        summaries.append({
            "task_id": task["id"],
            "target_concept": task["target_concept"],
            "model": method,
            **{field: first[field] for field in SUMMARY_FIELDS[3:17]},
            "target_n_images": int(target["n_images"]),
            "target_clip_score": _float(target, "clip_score"),
            "target_fid_vs_original": _float(target, "fid_vs_original"),
            "non_target_n_contents": len(non_target_rows),
            "non_target_n_images": sum(int(row["n_images"]) for row in non_target_rows),
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


def build_comparison_rows(summary_rows, task):
    by_method = {row["model"]: row for row in summary_rows}
    legacy = by_method[METHOD_LEGACY]
    comparisons = []
    for method in (METHOD_TGPRS, METHOD_LEARNED):
        candidate = by_method[method]
        comparisons.append({
            "task_id": task["id"],
            "comparison_method": method,
            "legacy_target_clip_score": legacy["target_clip_score"],
            "method_target_clip_score": candidate["target_clip_score"],
            "target_clip_score_improvement": (
                legacy["target_clip_score"] - candidate["target_clip_score"]
            ),
            "legacy_target_fid": legacy["target_fid_vs_original"],
            "method_target_fid": candidate["target_fid_vs_original"],
            "target_fid_delta": (
                candidate["target_fid_vs_original"]
                - legacy["target_fid_vs_original"]
            ),
            "legacy_non_target_clip_score": legacy["non_target_clip_score_mean"],
            "method_non_target_clip_score": candidate[
                "non_target_clip_score_mean"
            ],
            "non_target_clip_score_delta": (
                candidate["non_target_clip_score_mean"]
                - legacy["non_target_clip_score_mean"]
            ),
            "legacy_non_target_fid": legacy["non_target_fid_mean"],
            "method_non_target_fid": candidate["non_target_fid_mean"],
            "non_target_fid_improvement": (
                legacy["non_target_fid_mean"] - candidate["non_target_fid_mean"]
            ),
            "legacy_mscoco_clip_score": legacy["mscoco_clip_score"],
            "method_mscoco_clip_score": candidate["mscoco_clip_score"],
            "mscoco_clip_score_improvement": (
                candidate["mscoco_clip_score"] - legacy["mscoco_clip_score"]
            ),
            "legacy_mscoco_fid": legacy["mscoco_fid_vs_original"],
            "method_mscoco_fid": candidate["mscoco_fid_vs_original"],
            "mscoco_fid_improvement": (
                legacy["mscoco_fid_vs_original"]
                - candidate["mscoco_fid_vs_original"]
            ),
        })
    return comparisons


def _format_metric(value):
    if value == "" or value is None:
        return "—"
    return f"{float(value):.4f}"


def write_report(path, config, summary_rows, comparison_rows, learned_metadata):
    labels = {
        METHOD_ORIGINAL: "SD v1.4 (original)",
        METHOD_LEGACY: "SPEED",
        METHOD_TGPRS: "TGPRS",
        METHOD_LEARNED: "CLIP-guided learned anchor",
    }
    decoded_prefix = html.escape(str(learned_metadata["decoded_prefix"]))
    lines = [
        "# Van Gogh learned-anchor evaluation",
        "",
        "Lower target CLIP score indicates stronger erasure. Lower non-target and "
        "MS-COCO FID and higher MS-COCO CLIP indicate better preservation. Target "
        "FID is descriptive and is not treated as a quality direction.",
        "",
        "## Learned anchor",
        "",
        f"- Prefix token IDs: `{json.dumps(learned_metadata['prefix_token_ids'])}`",
        f"- Decoded prefix: <code>{decoded_prefix}</code>",
        f"- Best validation step: {learned_metadata['best_step']}",
        "- Anchor-search validation CLIP similarity: "
        f"{float(learned_metadata['best_validation_clip_similarity']):.6f}",
        "",
        "## Four-model summary",
        "",
        "| Model | Target CS ↓ | Target FID | Non-target CS | Non-target FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {labels[row['model']]} | "
            f"{_format_metric(row['target_clip_score'])} | "
            f"{_format_metric(row['target_fid_vs_original'])} | "
            f"{_format_metric(row['non_target_clip_score_mean'])} | "
            f"{_format_metric(row['non_target_fid_mean'])} | "
            f"{_format_metric(row['mscoco_clip_score'])} | "
            f"{_format_metric(row['mscoco_fid_vs_original'])} |"
        )
    lines.extend([
        "",
        "## Change relative to SPEED",
        "",
        "Positive improvement values favor the comparison method. Non-target CLIP "
        "and target FID are raw deltas without an assumed preferred direction.",
        "",
        "| Method | Target CS improvement ↑ | Target FID Δ | Non-target CS Δ | Non-target FID improvement ↑ | MS-COCO CS improvement ↑ | MS-COCO FID improvement ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| {labels[row['comparison_method']]} | "
            f"{_format_metric(row['target_clip_score_improvement'])} | "
            f"{_format_metric(row['target_fid_delta'])} | "
            f"{_format_metric(row['non_target_clip_score_delta'])} | "
            f"{_format_metric(row['non_target_fid_improvement'])} | "
            f"{_format_metric(row['mscoco_clip_score_improvement'])} | "
            f"{_format_metric(row['mscoco_fid_improvement'])} |"
        )
    experiment = config["experiment"]
    lines.extend([
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{experiment['seed']}`",
        f"- Sampling: `{experiment['inference_timesteps']}` DPM-Solver steps, CFG `{experiment['guidance_scale']}`",
        f"- Style samples: `{experiment['num_samples_per_prompt']}` per each of 30 templates",
        f"- MS-COCO prompts: `{experiment['mscoco_num_prompts']}`",
        f"- CLIP: `{experiment['clip_model']}`",
        f"- FID feature layer: `{experiment['fid_feature_layer']}`",
        "",
    ])
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def runtime_fingerprint(config, args, learned_metadata):
    workflow_dir = Path(args.config).parent
    configuration_hashes = {}
    for path in sorted(workflow_dir.glob("*.yaml")):
        configuration_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "workflow": config_fingerprint(config),
        "configuration_files": configuration_hashes,
        "anchor_manifest": learned_metadata["manifest_sha256"],
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "mscoco_num_prompts": args.mscoco_num_prompts,
        "seed": args.seed,
        "inference_timesteps": args.inference_timesteps,
        "guidance_scale": args.guidance_scale,
        "clip_model": args.clip_model,
        "fid_feature_layer": args.fid_feature_layer,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def evaluate(args):
    config = load_config(args.config)
    task = config["task"]
    methods = config["experiment"]["methods"]
    train_configs = load_train_configs(Path(args.config).parent)
    learned_metadata = load_learned_anchor_metadata(args.output_root, task)
    run_hash = runtime_fingerprint(config, args, learned_metadata)
    metrics_dir = Path(args.metrics_dir)
    detailed_path = metrics_dir / "detailed_metrics.csv"
    summary_path = metrics_dir / "summary.csv"
    comparison_path = metrics_dir / "comparison.csv"
    report_path = metrics_dir / "report.md"
    existing = {} if args.force else load_existing_rows(detailed_path)
    result_rows = []
    metric_memory = {}
    clip_scorer = None

    coco_records = load_mscoco_records(
        Path(args.repo_root) / "data" / "mscoco.csv",
        args.mscoco_num_prompts,
    )
    from src.template import template_dict

    few_records = {
        content: few_prompt_records(
            template_dict["style"], content, args.num_samples_per_prompt
        )
        for content in task["contents"]
    }

    def get_metrics(image_dir, reference_dir, records, manifest, reference_manifest):
        nonlocal clip_scorer
        memory_key = (str(image_dir), manifest)
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

    for method in methods:
        metadata = training_metadata(
            method, task, train_configs, learned_metadata
        )
        for content in task["contents"] + ["coco"]:
            key = (task["id"], method, content)
            if content == "coco":
                records = coco_records
                reference_dir = mscoco_image_dir(
                    args.output_root, METHOD_ORIGINAL, task["id"]
                )
                image_dir = mscoco_image_dir(args.output_root, method, task["id"])
                role = "coco"
            else:
                records = few_records[content]
                reference_dir = few_image_dir(
                    args.output_root, METHOD_ORIGINAL, task, content
                )
                image_dir = few_image_dir(args.output_root, method, task, content)
                role = "target" if content == task["target_concept"] else "non_target"

            reference_manifest = validate_image_directory(reference_dir, records)
            image_manifest = validate_image_directory(image_dir, records)
            cached = existing.get(key)
            if (
                cached
                and cached.get("run_fingerprint") == run_hash
                and cached.get("image_manifest_sha256") == image_manifest
                and int(cached.get("n_images", 0)) == len(records)
            ):
                row = cached
                print(f"Cached metrics: {method} {content}")
            else:
                clip, fid = get_metrics(
                    image_dir,
                    reference_dir,
                    records,
                    image_manifest,
                    reference_manifest,
                )
                row = {
                    "task_id": task["id"],
                    "target_concept": task["target_concept"],
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
                    "image_manifest_sha256": image_manifest,
                    "run_fingerprint": run_hash,
                }
                print(f"{method} {content}: CLIP={clip:.6f}, FID={fid:.6f}")
            result_rows.append(row)
            atomic_write_csv(detailed_path, result_rows, DETAILED_FIELDS)

    expected_rows = len(methods) * (len(task["contents"]) + 1)
    if len(result_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(result_rows)}")
    summary_rows = summarize_detailed_rows(result_rows, task, methods)
    comparison_rows = build_comparison_rows(summary_rows, task)
    atomic_write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    atomic_write_csv(comparison_path, comparison_rows, COMPARISON_FIELDS)
    write_report(report_path, config, summary_rows, comparison_rows, learned_metadata)
    for path in (detailed_path, summary_path, comparison_path, report_path):
        print(f"Saved: {path}")


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
