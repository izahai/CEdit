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
    low_rank_delta_spectrum,
    low_rank_delta_spectrum_from_target_factor,
    normalize_rows,
    projected_target_residuals,
    random_negative_target_residuals,
    random_rank_residuals,
    retain_eigensystem,
    retain_projection_from_eigensystem,
    second_moment,
    spectral_metrics,
    speed_retain_construction,
    speed_delta_weight_from_target_factor,
    speed_right_factor,
    tgprs_residuals_from_basis,
    truncated_svd_residuals,
)


SCHEMA_VERSION = "1.1"


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


def write_common_anchor_geometry(
    path,
    targets,
    target_names,
    anchor,
    anchor_name="person",
    count=10,
    seed=0,
    rtol=1e-5,
):
    """Persist a model-independent 2-D view and exact residual cosines."""
    if count <= 0 or count > len(target_names):
        raise ValueError("Common-anchor target count is outside the dataset")
    indices = subset_indices(len(target_names), count, seed).to(targets.device)
    selected_targets = targets[indices]
    selected_names = [target_names[index] for index in indices.cpu().tolist()]
    residuals = anchor.expand_as(selected_targets) - selected_targets
    points = torch.cat([selected_targets, anchor], dim=0).float()
    centered = points - points.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[:2].T
    energy = singular_values.square()
    explained = energy[:2] / energy.sum().clamp_min(1e-30)
    normalized = normalize_rows(residuals)
    rank_summary, _ = spectral_metrics(residuals, rtol=rtol)
    record = {
        "schema_version": SCHEMA_VERSION,
        "target_count": int(count),
        "seed": int(seed),
        "target_names": selected_names,
        "target_coordinates": coordinates[:-1].cpu().tolist(),
        "anchor_name": str(anchor_name),
        "anchor_coordinate": coordinates[-1].cpu().tolist(),
        "pca_explained_variance_ratio": explained.cpu().tolist(),
        "residual_cosine_similarity": (normalized @ normalized.T).cpu().tolist(),
        "residual": rank_summary,
    }
    with path.open("w", encoding="utf-8") as destination:
        json.dump(record, destination, indent=2, sort_keys=True)
        destination.write("\n")


def delta_rank_records(
    record_id,
    method,
    requested_rank,
    seed,
    target_count,
    residuals,
    targets,
    right_factor,
    layer_weights,
    rtol,
    spectra,
):
    rows = []
    transformed_targets = targets @ right_factor / targets.shape[0]
    for layer_index, (layer_name, layer_weight) in enumerate(layer_weights):
        singular_values = low_rank_delta_spectrum_from_target_factor(
            layer_weight, residuals, transformed_targets
        )
        summary, _ = spectral_metrics(
            singular_values.new_zeros((1, 1)),
            rtol=rtol,
            singular_values=singular_values,
        )
        spectra[f"{record_id}__delta_layer{layer_index:02d}"] = (
            singular_values.detach().cpu().numpy()
        )
        rows.append({
            "record_id": f"{record_id}-layer{layer_index:02d}",
            "section": "A",
            "method": method,
            "requested_rank": requested_rank,
            "seed": int(seed),
            "target_count": int(target_count),
            "layer_index": int(layer_index),
            "layer_name": layer_name,
            **{f"delta_{key}": value for key, value in summary.items()},
        })
    aggregate = {
        "record_id": f"{record_id}-aggregate",
        "section": "A",
        "method": method,
        "requested_rank": requested_rank,
        "seed": int(seed),
        "target_count": int(target_count),
        "layer_index": "aggregate",
        "layer_name": "aggregate",
        "layer_count": len(rows),
    }
    for key in ("numerical_rank", "stable_rank", "effective_rank"):
        values = [float(row[f"delta_{key}"]) for row in rows]
        aggregate[f"delta_{key}_mean"] = sum(values) / len(values)
        aggregate[f"delta_{key}_min"] = min(values)
        aggregate[f"delta_{key}_max"] = max(values)
    return rows + [aggregate]


def run_part_a(
    config,
    targets,
    target_names,
    anchor,
    extra_anchors,
    retain_embeddings,
    k2,
    layer_weights,
    output,
    spectra,
):
    section = config["part_a"]
    output_path = output / "geometry_metrics.jsonl"
    delta_path = output / "delta_rank_metrics.jsonl"
    completed = existing_record_ids(output_path)
    completed_delta = existing_record_ids(delta_path)
    rtol = float(section.get("numerical_rank_rtol", 1e-5))
    counts = [int(value) for value in section["target_counts"]]
    seeds = [int(value) for value in section["subset_seeds"]]
    ranks = [int(value) for value in section["tgprs_ranks"]]
    if max(counts) > len(target_names):
        raise ValueError("Part A target count exceeds available erase concepts")

    delta_config = section.get("delta_rank", {})
    delta_enabled = bool(delta_config.get("enabled", False))
    retain_u, retain_singular_values = retain_eigensystem(retain_embeddings)
    retain_projection, _ = retain_projection_from_eigensystem(
        retain_u,
        retain_singular_values,
        float(delta_config.get("threshold", 0.1)),
    )

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

            residual_settings = [(legacy_id, "legacy", None, legacy)]

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
                residual_settings.append((record_id, "tgprs", rank, residuals))

            if delta_enabled:
                target_covariance = second_moment(selected_targets)
                right_factor = speed_right_factor(
                    target_covariance,
                    retain_projection,
                    k2,
                    float(delta_config.get("retain_scale", 0.1)),
                    float(delta_config.get("lamb", 0.0)),
                )
                for base_id, method, requested_rank, residuals in residual_settings:
                    aggregate_id = f"{base_id}-aggregate"
                    if aggregate_id in completed_delta:
                        continue
                    for delta_record in delta_rank_records(
                        base_id,
                        method,
                        requested_rank,
                        seed,
                        count,
                        residuals,
                        selected_targets,
                        right_factor,
                        layer_weights,
                        rtol,
                        spectra,
                    ):
                        if delta_record["record_id"] not in completed_delta:
                            append_jsonl(delta_path, delta_record)
                            completed_delta.add(delta_record["record_id"])


def residual_settings(part_b, targets, legacy, extra_anchors):
    ranks = [int(value) for value in part_b["ranks"]]
    basis, _, basis_diagnostics = build_tgprs_basis(
        targets, extra_anchors, max(ranks)
    )
    canonical = {}
    for rank in ranks:
        canonical[rank], _ = tgprs_residuals_from_basis(
            targets, legacy, basis, rank
        )

    yield {
        "method": "legacy_full",
        "requested_rank": None,
        "basis_seed": None,
        "residuals": legacy,
        "canonical_residuals": None,
        "direction_variant": "legacy_anchor_delta",
        "diagnostics": {},
    }
    for rank in ranks:
        yield {
            "method": "legacy_svd",
            "requested_rank": rank,
            "basis_seed": None,
            "residuals": truncated_svd_residuals(legacy, rank),
            "canonical_residuals": canonical[rank],
            "direction_variant": "legacy_svd",
            "diagnostics": {},
        }
    for seed in [int(value) for value in part_b.get("random_seeds", [0])]:
        for rank in ranks:
            yield {
                "method": "random_subspace",
                "requested_rank": rank,
                "basis_seed": seed,
                "residuals": random_rank_residuals(legacy, rank, seed),
                "canonical_residuals": canonical[rank],
                "direction_variant": "projected_legacy_random",
                "diagnostics": {"basis_seed": seed},
            }
    for rank in ranks:
        residuals, direction_diagnostics = tgprs_residuals_from_basis(
            targets, legacy, basis, rank
        )
        diagnostics = dict(basis_diagnostics)
        diagnostics.update(direction_diagnostics)
        yield {
            "method": "tgprs",
            "requested_rank": rank,
            "basis_seed": None,
            "residuals": residuals,
            "canonical_residuals": canonical[rank],
            "direction_variant": "negative_target_projection",
            "diagnostics": diagnostics,
        }

    if not bool(part_b.get("direction_ablations", {}).get("enabled", False)):
        return
    for rank in ranks:
        positive, positive_diagnostics = projected_target_residuals(
            targets, legacy, basis, rank, sign=1.0
        )
        complement, complement_diagnostics = projected_target_residuals(
            targets, legacy, basis, rank, sign=-1.0, complement=True
        )
        yield {
            "method": "tgprs_positive",
            "requested_rank": rank,
            "basis_seed": None,
            "residuals": positive,
            "canonical_residuals": canonical[rank],
            "direction_variant": "positive_target_projection",
            "diagnostics": {**basis_diagnostics, **positive_diagnostics},
        }
        yield {
            "method": "tgprs_complement",
            "requested_rank": rank,
            "basis_seed": None,
            "residuals": complement,
            "canonical_residuals": canonical[rank],
            "direction_variant": "negative_target_complement",
            "diagnostics": {**basis_diagnostics, **complement_diagnostics},
        }
    for seed in [int(value) for value in part_b.get("random_seeds", [0])]:
        for rank in ranks:
            residuals, diagnostics = random_negative_target_residuals(
                targets, legacy, rank, seed
            )
            yield {
                "method": "random_negative_target",
                "requested_rank": rank,
                "basis_seed": seed,
                "residuals": residuals,
                "canonical_residuals": canonical[rank],
                "direction_variant": "negative_target_random",
                "diagnostics": diagnostics,
            }


def aggregate_record(base, rows):
    per_layer_fields = {
        "record_id",
        "layer_index",
        "layer_name",
        "target_effect",
        "target_rotation_deg",
        "retain_leakage",
        "directional_realization_cosine",
        "directional_relative_error",
        "delta_frobenius_norm",
        "edited_output_norm_ratio",
        "canonical_erasure_alignment",
        "anchor_distance_ratio",
        "delta_numerical_rank",
        "delta_stable_rank",
        "delta_effective_rank",
        "retain_low_rank",
        "retain_filtered_count",
        "retain_augmented_count",
        "retain_constructed_count",
        "retain_seed",
    }
    record = {key: value for key, value in base.items() if key not in per_layer_fields}
    record.update({
        "record_id": f"{base['configuration_id']}-aggregate",
        "layer_index": "aggregate",
        "layer_name": "aggregate",
        "layer_count": len(rows),
    })
    record.update(aggregate_layer_metrics(rows))
    for name in (
        "retain_low_rank",
        "retain_filtered_count",
        "retain_augmented_count",
        "retain_constructed_count",
        "delta_numerical_rank",
        "delta_stable_rank",
        "delta_effective_rank",
    ):
        if all(name in row for row in rows):
            values = [float(row[name]) for row in rows]
            record[f"{name}_mean"] = sum(values) / len(values)
            record[f"{name}_min"] = min(values)
            record[f"{name}_max"] = max(values)
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
    thresholds = [
        float(value)
        for value in section.get(
            "retain_thresholds", [retain_config.get("threshold", 0.1)]
        )
    ]
    retain_scales = [float(value) for value in section["retain_scales"]]
    fixed_u, fixed_spectrum = retain_eigensystem(retain_embeddings)
    np.save(output / "retain_singular_values.npy", fixed_spectrum.cpu().numpy())
    retain_spectra = {"fixed": fixed_spectrum.detach().cpu().numpy()}
    compute_delta_spectrum = bool(section.get("compute_delta_spectrum", False))
    rtol = float(section.get("numerical_rank_rtol", 1e-5))
    robustness = section.get("full_speed_robustness", {})
    robustness_ranks = {
        int(value) for value in robustness.get("tgprs_ranks", [5, 10, 30])
    }
    runtime_seed = int(config.get("runtime", {}).get("seed", 0))
    fixed_factors = {}
    for threshold in thresholds:
        projection, projection_rank = retain_projection_from_eigensystem(
            fixed_u, fixed_spectrum, threshold
        )
        for scale in retain_scales:
            fixed_factors[(threshold, scale)] = (
                projection_rank,
                speed_right_factor(
                    target_covariance,
                    projection,
                    k2,
                    scale,
                    float(retain_config.get("lamb", 0.0)),
                ),
            )
    fixed_factors = {
        key: (rank, factor, selected_targets @ factor / selected_targets.shape[0])
        for key, (rank, factor) in fixed_factors.items()
    }

    for setting in residual_settings(section, selected_targets, legacy, extra_anchors):
        method = setting["method"]
        requested_rank = setting["requested_rank"]
        random_seed = setting["basis_seed"]
        residuals = setting["residuals"]
        diagnostics = setting["diagnostics"]
        statistic = edit_statistic(residuals, selected_targets)
        residual_summary, _ = spectral_metrics(residuals, rtol=rtol)
        profiles = ["fixed"]
        if bool(robustness.get("enabled", False)) and (
            method == "legacy_full"
            or (method == "tgprs" and requested_rank in robustness_ranks)
        ):
            profiles.append("full_speed")

        for profile in profiles:
            layer_contexts = []
            for layer_index, (layer_name, layer_weight) in enumerate(layer_weights):
                if profile == "fixed":
                    context_embeddings = retain_embeddings
                    retain_u = fixed_u
                    retain_spectrum = fixed_spectrum
                    retain_diagnostics = {
                        "retain_original_count": int(retain_embeddings.shape[0]),
                        "retain_filtered_count": int(retain_embeddings.shape[0]),
                        "retain_augmented_count": 0,
                        "retain_constructed_count": int(retain_embeddings.shape[0]),
                        "retain_aug_num": 0,
                        "retain_filter_enabled": False,
                        "retain_seed": runtime_seed,
                    }
                else:
                    construction_seed = (
                        runtime_seed
                        + layer_index
                        + 100 * int(requested_rank or 0)
                    )
                    context_embeddings, retain_diagnostics = speed_retain_construction(
                        retain_embeddings,
                        layer_weight,
                        statistic,
                        target_covariance,
                        aug_num=int(robustness.get("aug_num", 10)),
                        filter_enabled=bool(robustness.get("filter_enabled", True)),
                        seed=construction_seed,
                    )
                    retain_u, retain_spectrum = retain_eigensystem(context_embeddings)
                    spectrum_key = (
                        f"{profile}__{method}__k{requested_rank or 'full'}"
                        f"__layer{layer_index:02d}"
                    )
                    retain_spectra[spectrum_key] = (
                        retain_spectrum.detach().cpu().numpy()
                    )
                layer_contexts.append((
                    layer_name,
                    layer_weight,
                    retain_u,
                    retain_spectrum,
                    retain_diagnostics,
                ))

            for threshold in thresholds:
                for scale in retain_scales:
                    rank_label = (
                        "full" if requested_rank is None else f"k{requested_rank}"
                    )
                    seed_label = (
                        "" if random_seed is None else f"-seed{random_seed}"
                    )
                    configuration_id = (
                        f"b-{profile}-{method}-{rank_label}{seed_label}"
                        f"-t{threshold:g}-rs{scale:g}"
                    )
                    aggregate_id = f"{configuration_id}-aggregate"
                    if aggregate_id in completed:
                        continue
                    layer_rows = []
                    for layer_index, context in enumerate(layer_contexts):
                        (
                            layer_name,
                            layer_weight,
                            retain_u,
                            retain_spectrum,
                            retain_diagnostics,
                        ) = context
                        layer_id = f"{configuration_id}-layer{layer_index:02d}"
                        if layer_id in previous_records:
                            layer_rows.append(previous_records[layer_id])
                            continue
                        if profile == "fixed":
                            retain_rank, right_factor, transformed_targets = fixed_factors[
                                (threshold, scale)
                            ]
                        else:
                            retain_projection, retain_rank = (
                                retain_projection_from_eigensystem(
                                    retain_u, retain_spectrum, threshold
                                )
                            )
                            right_factor = speed_right_factor(
                                target_covariance,
                                retain_projection,
                                k2,
                                scale,
                                float(retain_config.get("lamb", 0.0)),
                            )
                            transformed_targets = (
                                selected_targets @ right_factor
                                / selected_targets.shape[0]
                            )
                        delta = speed_delta_weight_from_target_factor(
                            layer_weight,
                            residuals,
                            transformed_targets,
                        )
                        metrics = layer_edit_metrics(
                            layer_weight,
                            delta,
                            selected_targets,
                            residuals,
                            retain_embeddings,
                            anchor_embeddings=selected_anchor,
                            canonical_residuals=setting["canonical_residuals"],
                        )
                        if compute_delta_spectrum:
                            singular_values = low_rank_delta_spectrum(
                                layer_weight,
                                residuals,
                                selected_targets,
                                right_factor,
                            )
                            delta_summary, _ = spectral_metrics(
                                singular_values.new_zeros((1, 1)),
                                rtol=rtol,
                                singular_values=singular_values,
                            )
                            metrics.update({
                                "delta_numerical_rank": delta_summary["numerical_rank"],
                                "delta_stable_rank": delta_summary["stable_rank"],
                                "delta_effective_rank": delta_summary["effective_rank"],
                            })
                        base = {
                            "section": "B",
                            "configuration_id": configuration_id,
                            "target_count": count,
                            "target_seed": seed,
                            "target_subset_hash": subset_hash(
                                [target_names[index] for index in indices.cpu().tolist()]
                            ),
                            "method": method,
                            "direction_variant": setting["direction_variant"],
                            "requested_rank": requested_rank,
                            "realized_rank": residual_summary["numerical_rank"],
                            "random_seed": random_seed,
                            "basis_seed": random_seed,
                            "retain_profile": profile,
                            "retain_scale": scale,
                            "retain_threshold": threshold,
                            "retain_low_rank": retain_rank,
                            "aug_num": retain_diagnostics["retain_aug_num"],
                            "filter_enabled": retain_diagnostics[
                                "retain_filter_enabled"
                            ],
                            **retain_diagnostics,
                            "diagnostics": diagnostics,
                        }
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
                    aggregate = aggregate_record(layer_rows[0], layer_rows)
                    aggregate["record_id"] = aggregate_id
                    append_jsonl(output_path, aggregate)
                    completed.add(aggregate_id)
                    print(f"[Part B] wrote {configuration_id}", flush=True)
            np.savez_compressed(output / "retain_spectra.npz", **retain_spectra)


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
            "delta_rank_metrics": "delta_rank_metrics.jsonl",
            "edit_metrics": "edit_metrics.jsonl",
            "spectra": "spectra.npz",
            "retain_spectrum": "retain_singular_values.npy",
            "retain_spectra": "retain_spectra.npz",
            "common_anchor_geometry": "common_anchor_geometry.json",
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
        for name in (
            "geometry_metrics.jsonl",
            "delta_rank_metrics.jsonl",
            "edit_metrics.jsonl",
        )
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

    common_config = config.get("common_anchor_figure", {})
    if bool(common_config.get("enabled", False)):
        write_common_anchor_geometry(
            output / "common_anchor_geometry.json",
            targets,
            target_names,
            anchor,
            anchor_name=common_anchor,
            count=int(common_config.get("target_count", 10)),
            seed=int(common_config.get("seed", 0)),
            rtol=float(common_config.get("numerical_rank_rtol", 1e-5)),
        )

    spectra = {}
    spectra_path = output / "spectra.npz"
    if spectra_path.exists():
        with np.load(spectra_path) as existing:
            spectra.update({key: existing[key] for key in existing.files})
    if args.part in ("all", "a") and config.get("part_a", {}).get("enabled", True):
        run_part_a(
            config,
            targets,
            target_names,
            anchor,
            extra_anchors,
            retain_embeddings,
            k2,
            layer_weights,
            output,
            spectra,
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
