#!/usr/bin/env python3
"""Run residual-rank and closed-form SPEED analyses without saving checkpoints."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from kmeans_pytorch import kmeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rank_analysis import (  # noqa: E402
    aggregate_layer_metrics,
    build_tgprs_basis,
    edit_statistic,
    layer_edit_metrics,
    normalize_rows,
    random_rank_residuals,
    retain_low_projection,
    second_moment,
    spectral_metrics,
    speed_delta_weight,
    speed_right_factor,
    tgprs_residuals_from_basis,
    truncated_svd_residuals,
)


SCHEMA_VERSION = "1.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rank_analysis.yaml",
        help="YAML experiment configuration",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--part",
        choices=("all", "a", "b"),
        default="all",
        help="Run both analyses or only one section",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Refuse to reuse an output directory containing result records",
    )
    return parser.parse_args()


def resolve_path(value):
    path = Path(os.path.expandvars(str(value)))
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path):
    with resolve_path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Analysis config must be a YAML mapping")
    return config


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def unique_csv_values(path, column, split=None):
    values = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV lacks column {column!r}: {path}")
        for row in reader:
            if split is not None and row.get("type") != split:
                continue
            value = (row.get(column) or "").strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
    if not values:
        raise ValueError(f"No values found in {path} for column {column!r}")
    return values


@torch.no_grad()
def encode_last_subject_embeddings(pipeline, prompts, device, batch_size):
    embeddings = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = pipeline.tokenizer(
            batch,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = pipeline.text_encoder(
            inputs.input_ids.to(device)
        ).last_hidden_state
        subject_indices = (inputs.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(hidden.shape[0], device=device)
        embeddings.append(hidden[batch_indices, subject_indices].float())
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def build_speed_k2(pipeline, device, seed):
    inputs = pipeline.tokenizer(
        "",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden = pipeline.text_encoder(inputs.input_ids.to(device)).last_hidden_state[0]
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    _, centers = kmeans(
        X=hidden[1:].float(),
        num_clusters=3,
        distance="euclidean",
        device=str(device),
    )
    return torch.cat([hidden[[0], :].float(), centers.to(device).float()], dim=0).T


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def existing_record_ids(path):
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["record_id"])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from error
    return ids


def existing_records(path):
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[record["record_id"]] = record
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from error
    return records


def subset_indices(total, count, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:count]


def subset_hash(concepts):
    return hashlib.sha256("\n".join(concepts).encode("utf-8")).hexdigest()[:16]


def geometry_record(
    record_id,
    seed,
    concepts,
    method,
    requested_rank,
    residuals,
    targets,
    rtol,
    spectra,
    diagnostics=None,
):
    statistic = edit_statistic(residuals, targets)
    residual_summary, residual_spectrum = spectral_metrics(residuals, rtol=rtol)
    normalized_summary, _ = spectral_metrics(
        normalize_rows(residuals), rtol=rtol
    )
    statistic_summary, statistic_spectrum = spectral_metrics(statistic, rtol=rtol)
    spectra[f"{record_id}__R"] = residual_spectrum.detach().cpu().numpy()
    spectra[f"{record_id}__D"] = statistic_spectrum.detach().cpu().numpy()
    return {
        "record_id": record_id,
        "section": "A",
        "seed": int(seed),
        "subset_hash": subset_hash(concepts),
        "target_count": len(concepts),
        "method": method,
        "requested_rank": requested_rank,
        "realized_rank": residual_summary["numerical_rank"],
        "residual_shape": list(residuals.shape),
        "residual": residual_summary,
        "direction_normalized_residual": normalized_summary,
        "edit_statistic": statistic_summary,
        "diagnostics": diagnostics or {},
    }


def run_part_a(config, targets, target_names, anchor, extra_anchors, output, spectra):
    section = config["part_a"]
    output_path = output / "geometry_metrics.jsonl"
    completed = existing_record_ids(output_path)
    rtol = float(section.get("numerical_rank_rtol", 1e-5))
    counts = [int(value) for value in section["target_counts"]]
    seeds = [int(value) for value in section["subset_seeds"]]
    ranks = [int(value) for value in section["tgprs_ranks"]]
    if max(counts) > len(target_names):
        raise ValueError("Part A target count exceeds available erase concepts")

    for seed in seeds:
        for count in counts:
            indices = subset_indices(len(target_names), count, seed).to(targets.device)
            selected_targets = targets[indices]
            selected_names = [target_names[index] for index in indices.cpu().tolist()]
            selected_anchor = anchor.expand_as(selected_targets)
            legacy = selected_anchor - selected_targets
            legacy_id = f"a-seed{seed}-n{count}-legacy"
            record = geometry_record(
                legacy_id,
                seed,
                selected_names,
                "legacy",
                None,
                legacy,
                selected_targets,
                rtol,
                spectra,
            )
            if legacy_id not in completed:
                append_jsonl(output_path, record)
                completed.add(legacy_id)
                print(f"[Part A] wrote {legacy_id}", flush=True)

            basis, pairwise_spectrum, basis_diagnostics = build_tgprs_basis(
                selected_targets, extra_anchors, max(ranks)
            )
            spectra[f"a-seed{seed}-n{count}__pairwise"] = (
                pairwise_spectrum.detach().cpu().numpy()
            )
            for rank in ranks:
                record_id = f"a-seed{seed}-n{count}-tgprs-k{rank}"
                residuals, direction_diagnostics = tgprs_residuals_from_basis(
                    selected_targets, legacy, basis, rank
                )
                diagnostics = dict(basis_diagnostics)
                diagnostics.update(direction_diagnostics)
                record = geometry_record(
                    record_id,
                    seed,
                    selected_names,
                    "tgprs",
                    rank,
                    residuals,
                    selected_targets,
                    rtol,
                    spectra,
                    diagnostics,
                )
                if record_id not in completed:
                    append_jsonl(output_path, record)
                    completed.add(record_id)
                    print(f"[Part A] wrote {record_id}", flush=True)


def residual_settings(part_b, targets, legacy, extra_anchors):
    ranks = [int(value) for value in part_b["ranks"]]
    yield "legacy_full", None, None, legacy, {}
    for rank in ranks:
        yield (
            "legacy_svd",
            rank,
            None,
            truncated_svd_residuals(legacy, rank),
            {},
        )
    for seed in [int(value) for value in part_b.get("random_seeds", [0])]:
        for rank in ranks:
            yield (
                "random_subspace",
                rank,
                seed,
                random_rank_residuals(legacy, rank, seed),
                {},
            )
    basis, _, basis_diagnostics = build_tgprs_basis(
        targets, extra_anchors, max(ranks)
    )
    for rank in ranks:
        residuals, direction_diagnostics = tgprs_residuals_from_basis(
            targets, legacy, basis, rank
        )
        diagnostics = dict(basis_diagnostics)
        diagnostics.update(direction_diagnostics)
        yield "tgprs", rank, None, residuals, diagnostics


def aggregate_record(base, rows):
    record = dict(base)
    record.update({
        "record_id": f"{base['configuration_id']}-aggregate",
        "layer_index": "aggregate",
        "layer_name": "aggregate",
        "layer_count": len(rows),
    })
    record.update(aggregate_layer_metrics(rows))
    return record


def run_part_b(
    config,
    targets,
    target_names,
    anchor,
    extra_anchors,
    retain_embeddings,
    k2,
    layer_weights,
    output,
):
    section = config["part_b"]
    output_path = output / "edit_metrics.jsonl"
    previous_records = existing_records(output_path)
    completed = set(previous_records)
    count = int(section["target_count"])
    if count > len(target_names):
        raise ValueError("Part B target count exceeds available erase concepts")
    seed = int(section.get("target_seed", 0))
    indices = subset_indices(len(target_names), count, seed).to(targets.device)
    selected_targets = targets[indices]
    selected_anchor = anchor.expand_as(selected_targets)
    legacy = selected_anchor - selected_targets
    target_covariance = second_moment(selected_targets)

    retain_config = config["retain"]
    if int(retain_config.get("aug_num", 0)) != 0:
        raise ValueError("Rank analysis currently requires retain.aug_num: 0")
    retain_projection, retain_spectrum, retain_rank = retain_low_projection(
        retain_embeddings, float(retain_config["threshold"])
    )
    np.save(output / "retain_singular_values.npy", retain_spectrum.cpu().numpy())
    right_factors = {
        float(scale): speed_right_factor(
            target_covariance,
            retain_projection,
            k2,
            float(scale),
            float(retain_config.get("lamb", 0.0)),
        )
        for scale in section["retain_scales"]
    }
    compute_delta_spectrum = bool(section.get("compute_delta_spectrum", False))
    rtol = float(section.get("numerical_rank_rtol", 1e-5))

    for method, requested_rank, random_seed, residuals, diagnostics in residual_settings(
        section, selected_targets, legacy, extra_anchors
    ):
        statistic = edit_statistic(residuals, selected_targets)
        residual_summary, _ = spectral_metrics(residuals, rtol=rtol)
        for scale, right_factor in right_factors.items():
            rank_label = "full" if requested_rank is None else f"k{requested_rank}"
            seed_label = "" if random_seed is None else f"-seed{random_seed}"
            configuration_id = (
                f"b-{method}-{rank_label}{seed_label}-rs{scale:g}"
            )
            aggregate_id = f"{configuration_id}-aggregate"
            if aggregate_id in completed:
                continue
            base = {
                "section": "B",
                "configuration_id": configuration_id,
                "target_count": count,
                "target_seed": seed,
                "target_subset_hash": subset_hash(
                    [target_names[index] for index in indices.cpu().tolist()]
                ),
                "method": method,
                "requested_rank": requested_rank,
                "realized_rank": residual_summary["numerical_rank"],
                "random_seed": random_seed,
                "retain_scale": scale,
                "retain_threshold": float(retain_config["threshold"]),
                "retain_low_rank": retain_rank,
                "diagnostics": diagnostics,
            }
            layer_rows = []
            for layer_index, (layer_name, layer_weight) in enumerate(layer_weights):
                layer_id = f"{configuration_id}-layer{layer_index:02d}"
                if layer_id in previous_records:
                    layer_rows.append(previous_records[layer_id])
                    continue
                delta = speed_delta_weight(layer_weight, statistic, right_factor)
                metrics = layer_edit_metrics(
                    layer_weight,
                    delta,
                    selected_targets,
                    residuals,
                    retain_embeddings,
                )
                if compute_delta_spectrum:
                    delta_summary, _ = spectral_metrics(delta, rtol=rtol)
                    metrics.update({
                        "delta_numerical_rank": delta_summary["numerical_rank"],
                        "delta_stable_rank": delta_summary["stable_rank"],
                        "delta_effective_rank": delta_summary["effective_rank"],
                    })
                row = dict(base)
                row.update({
                    "record_id": layer_id,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                })
                row.update(metrics)
                append_jsonl(output_path, row)
                completed.add(row["record_id"])
                layer_rows.append(row)
            aggregate = aggregate_record(base, layer_rows)
            append_jsonl(output_path, aggregate)
            completed.add(aggregate_id)
            print(f"[Part B] wrote {configuration_id}", flush=True)


def write_manifest(path, config, dataset_path, model_id, device, status, extra=None):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": git_commit(),
        "model": {
            "id": model_id,
            "device": str(device),
            "analysis_dtype": "float32",
            "edited_parameter": "attn2.to_v.weight",
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
        },
        "config": config,
        "artifacts": {
            "geometry_metrics": "geometry_metrics.jsonl",
            "edit_metrics": "edit_metrics.jsonl",
            "spectra": "spectra.npz",
            "retain_spectrum": "retain_singular_values.npy",
        },
    }
    if extra:
        manifest.update(extra)
    with path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]
    dataset_path = resolve_path(dataset_config["path"])
    target_names = unique_csv_values(
        dataset_path,
        dataset_config.get("concept_column", "concept"),
        dataset_config.get("target_split", "erase"),
    )
    retain_config = config["retain"]
    retain_path = resolve_path(retain_config["path"])
    retain_names = []
    for column in retain_config.get("columns", ["concept"]):
        for value in unique_csv_values(
            retain_path, column, retain_config.get("split")
        ):
            if value not in retain_names:
                retain_names.append(value)
    target_lower = {value.lower() for value in target_names}
    retain_names = [value for value in retain_names if value.lower() not in target_lower]
    if not retain_names:
        raise ValueError("Retain set is empty after excluding erase targets")

    experiment_name = config.get("experiment_name", "rank_analysis")
    output = (
        resolve_path(args.output_dir)
        if args.output_dir
        else resolve_path(config.get("output", {}).get("root", "logs/rank_analysis"))
        / experiment_name
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.fresh and any(
        (output / name).exists()
        for name in ("geometry_metrics.jsonl", "edit_metrics.jsonl")
    ):
        raise FileExistsError(f"Fresh run requested but results exist in {output}")

    requested_device = args.device or config.get("runtime", {}).get("device", "cuda")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    model_id = config.get("sd_ckpt", "CompVis/stable-diffusion-v1-4")
    write_manifest(
        output / "manifest.json", config, dataset_path, model_id, device, "running"
    )

    from diffusers import StableDiffusionPipeline

    model_dtype_name = config.get("runtime", {}).get("model_dtype", "float32")
    model_dtype = {"float32": torch.float32, "float16": torch.float16}[model_dtype_name]
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=model_dtype
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    batch_size = int(config.get("runtime", {}).get("batch_size", 128))
    common_anchor = str(dataset_config.get("common_anchor", "person"))
    extra_anchor_names = list(dataset_config.get("subspace_anchors", ["", "person"]))
    all_prompts = target_names + [common_anchor] + extra_anchor_names + retain_names
    all_embeddings = encode_last_subject_embeddings(
        pipeline, all_prompts, device, batch_size
    )
    target_count = len(target_names)
    targets = all_embeddings[:target_count]
    anchor = all_embeddings[target_count:target_count + 1]
    extra_start = target_count + 1
    extra_anchors = all_embeddings[
        extra_start:extra_start + len(extra_anchor_names)
    ]
    retain_embeddings = all_embeddings[extra_start + len(extra_anchor_names):]
    k2 = build_speed_k2(
        pipeline, device, config.get("runtime", {}).get("seed", 0)
    )
    layer_weights = [
        (name, value.detach().float())
        for name, value in pipeline.unet.state_dict().items()
        if name.endswith("attn2.to_v.weight")
    ]
    if not layer_weights:
        raise RuntimeError("No attn2.to_v.weight parameters found")

    spectra = {}
    spectra_path = output / "spectra.npz"
    if spectra_path.exists():
        with np.load(spectra_path) as existing:
            spectra.update({key: existing[key] for key in existing.files})
    if args.part in ("all", "a") and config.get("part_a", {}).get("enabled", True):
        run_part_a(
            config, targets, target_names, anchor, extra_anchors, output, spectra
        )
        np.savez_compressed(spectra_path, **spectra)
    if args.part in ("all", "b") and config.get("part_b", {}).get("enabled", True):
        run_part_b(
            config,
            targets,
            target_names,
            anchor,
            extra_anchors,
            retain_embeddings,
            k2,
            layer_weights,
            output,
        )
    write_manifest(
        output / "manifest.json",
        config,
        dataset_path,
        model_id,
        device,
        "complete",
        {
            "target_count": len(target_names),
            "retain_count": len(retain_names),
            "layer_count": len(layer_weights),
        },
    )
    print(f"Analysis complete: {output}")


if __name__ == "__main__":
    main()
