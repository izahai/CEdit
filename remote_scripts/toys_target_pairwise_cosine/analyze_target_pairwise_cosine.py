#!/usr/bin/env python3
"""Analyze natural target, person, and empty-prompt geometry across to_v layers."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import torch
import yaml


PAIR_TYPES = (
    "target_target",
    "target_person",
    "target_empty",
    "person_empty",
)


def pairwise_cosine(vectors, eps=1e-8):
    """Return the full row-wise cosine matrix using float32 arithmetic."""
    if vectors.ndim != 2:
        raise ValueError("Pairwise cosine vectors must be two-dimensional")
    vectors = vectors.float()
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(eps)
    normalized = vectors / norms
    return (normalized @ normalized.T).clamp(-1.0, 1.0)


def classify_pair(left_index, right_index, target_count):
    """Classify an unordered pair with person and empty after all targets."""
    if not 0 <= left_index < right_index < target_count + 2:
        raise ValueError("Pair indices must satisfy 0 <= left < right < target_count + 2")
    person_index = target_count
    empty_index = target_count + 1
    if right_index < target_count:
        return "target_target"
    if right_index == person_index:
        return "target_person"
    if left_index < target_count and right_index == empty_index:
        return "target_empty"
    if left_index == person_index and right_index == empty_index:
        return "person_empty"
    raise AssertionError("Unhandled pair classification")


def build_pair_rows(space_index, space_name, labels, cosine_matrix, target_count):
    """Flatten the upper triangle into labeled pair records."""
    expected_size = target_count + 2
    if cosine_matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Cosine matrix must have shape {(expected_size, expected_size)}, "
            f"got {tuple(cosine_matrix.shape)}"
        )
    if len(labels) != expected_size:
        raise ValueError("Label count must equal target_count + 2")
    rows = []
    for left_index in range(expected_size):
        for right_index in range(left_index + 1, expected_size):
            cosine = cosine_matrix[left_index, right_index].item()
            rows.append({
                "space_index": space_index,
                "space_name": space_name,
                "pair_type": classify_pair(
                    left_index, right_index, target_count
                ),
                "left_index": left_index,
                "left_label": labels[left_index],
                "right_index": right_index,
                "right_label": labels[right_index],
                "cosine": cosine,
                "angle_degrees": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            })
    return rows


def pearson_correlation(left, right, eps=1e-12):
    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must have equal non-zero length")
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= eps:
        return 1.0 if all(
            abs(left_value - right_value) <= eps
            for left_value, right_value in zip(left, right)
        ) else 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def summarize_space(pair_rows, vector_norms, target_count, embedding_target_cosines):
    """Summarize pair classes and distortion from embedding target geometry."""
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["pair_type"]].append(float(row["cosine"]))
    summary = {
        "space_index": pair_rows[0]["space_index"],
        "space_name": pair_rows[0]["space_name"],
    }
    for pair_type in PAIR_TYPES:
        values = grouped[pair_type]
        summary[f"{pair_type}_count"] = len(values)
        summary[f"{pair_type}_cosine_mean"] = fmean(values)
        summary[f"{pair_type}_cosine_std"] = pstdev(values)
        summary[f"{pair_type}_cosine_min"] = min(values)
        summary[f"{pair_type}_cosine_max"] = max(values)

    target_cosines = grouped["target_target"]
    summary["target_correlation_to_embedding"] = pearson_correlation(
        embedding_target_cosines, target_cosines
    )
    summary["target_rmse_from_embedding"] = math.sqrt(
        fmean(
            (current - original) ** 2
            for current, original in zip(
                target_cosines, embedding_target_cosines
            )
        )
    )
    norms = vector_norms.float().tolist()
    target_norms = norms[:target_count]
    summary.update({
        "target_norm_mean": fmean(target_norms),
        "target_norm_std": pstdev(target_norms),
        "target_norm_min": min(target_norms),
        "target_norm_max": max(target_norms),
        "person_norm": norms[target_count],
        "empty_norm": norms[target_count + 1],
    })
    return summary


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path, labels, matrix):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])


def load_target_concepts(csv_path, expected_count):
    concepts = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "concept" not in reader.fieldnames:
            raise ValueError(f"Target CSV lacks a concept column: {csv_path}")
        for row in reader:
            concept = (row.get("concept") or "").strip()
            if row.get("type") == "erase" and concept and concept not in concepts:
                concepts.append(concept)
    if len(concepts) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique erase concepts in {csv_path}, "
            f"found {len(concepts)}"
        )
    return concepts


@torch.no_grad()
def encode_last_subject_embeddings(text_encoder, tokenizer, prompts, device, batch_size):
    """Match the last-subject-token extraction used by the TGPRS workflow."""
    embeddings = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden_states = text_encoder(
            inputs.input_ids.to(device)
        ).last_hidden_state
        subject_indices = (inputs.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(len(batch), device=device)
        embeddings.append(hidden_states[batch_indices, subject_indices])
    return torch.cat(embeddings, dim=0)


def resolve_dtype(name):
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model dtype: {name}") from error


def make_embedding_heatmap(matrix, labels, target_count, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix.numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    anchor_ticks = [0, target_count - 1, target_count, target_count + 1]
    axis.set_xticks(anchor_ticks)
    axis.set_xticklabels(
        ["target 0", f"target {target_count - 1}", "person", "empty"],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(anchor_ticks)
    axis.set_yticklabels(
        ["target 0", f"target {target_count - 1}", "person", "empty"]
    )
    axis.axhline(target_count - 0.5, color="black", linewidth=1.5)
    axis.axvline(target_count - 0.5, color="black", linewidth=1.5)
    axis.set_title("Embedding-space pairwise cosine: 50 targets + person + empty")
    figure.colorbar(image, ax=axis, label="cosine similarity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_layer_heatmap_grid(layer_matrices, target_count, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 4, figsize=(17, 16), constrained_layout=True)
    image = None
    for layer_index, (layer_name, matrix) in enumerate(layer_matrices):
        axis = axes.flat[layer_index]
        image = axis.imshow(matrix.numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.axhline(target_count - 0.5, color="black", linewidth=1.0)
        axis.axvline(target_count - 0.5, color="black", linewidth=1.0)
        axis.set_xticks([0, target_count - 1, target_count, target_count + 1])
        axis.set_xticklabels(["0", "49", "P", "E"], fontsize=7)
        axis.set_yticks([0, target_count - 1, target_count, target_count + 1])
        axis.set_yticklabels(["0", "49", "P", "E"], fontsize=7)
        axis.set_title(f"Layer {layer_index}", fontsize=10)
    figure.suptitle(
        "Original to_v pairwise cosine matrices (P=person, E=empty)",
        fontsize=16,
    )
    figure.colorbar(image, ax=axes, label="cosine similarity", shrink=0.75)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_pair_type_summary(summary_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_rows = [row for row in summary_rows if row["space_index"] >= 0]
    figure, axis = plt.subplots(figsize=(12, 6))
    labels = {
        "target_target": "target-target mean",
        "target_person": "target-person mean",
        "target_empty": "target-empty mean",
        "person_empty": "person-empty",
    }
    for pair_type in PAIR_TYPES:
        axis.plot(
            [row["space_index"] for row in layer_rows],
            [row[f"{pair_type}_cosine_mean"] for row in layer_rows],
            marker="o",
            label=labels[pair_type],
        )
        embedding_value = summary_rows[0][f"{pair_type}_cosine_mean"]
        axis.axhline(embedding_value, linestyle="--", linewidth=0.8, alpha=0.5)
    axis.set_xticks(range(len(layer_rows)))
    axis.set_xlabel("original to_v layer index")
    axis.set_ylabel("mean cosine similarity")
    axis.set_title("Natural target and anchor geometry across to_v layers")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(config_path, output_override=None):
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("REPO_ROOT", str(repo_root))
    os.environ.setdefault("WORKSPACE_DIR", str(repo_root.parent))
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("Workflow config must contain an experiment mapping")

    experiment = config["experiment"]
    raw_csv_path = os.path.expandvars(str(experiment["target_csv"]))
    csv_path = Path(raw_csv_path)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    target_count = int(experiment.get("target_count", 50))
    target_concepts = load_target_concepts(csv_path, target_count)
    anchor_prompts = experiment.get("anchor_prompts", ["person", ""])
    if anchor_prompts != ["person", ""]:
        raise ValueError("anchor_prompts must be ['person', '']")
    prompts = [*target_concepts, *anchor_prompts]
    labels = [*target_concepts, "person", "<empty>"]

    checkpoint = str(experiment["sd_ckpt"])
    dtype = resolve_dtype(str(experiment.get("model_dtype", "float32")))
    device = torch.device(str(experiment.get("device", "cuda")))
    batch_size = int(experiment.get("batch_size", len(prompts)))
    eps = float(experiment.get("eps", 1e-8))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    raw_output = output_override or config.get("paths", {}).get(
        "output_root", "${WORKSPACE_DIR}/cedit_toys_target_pairwise_cosine"
    )
    output_dir = Path(os.path.expandvars(str(raw_output))).resolve()
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"Loading tokenizer and text encoder: {checkpoint}")
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        checkpoint, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()
    embeddings = encode_last_subject_embeddings(
        text_encoder, tokenizer, prompts, device, batch_size
    ).cpu()
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    embedding_matrix = pairwise_cosine(embeddings, eps=eps).cpu()
    embedding_rows = build_pair_rows(
        -1, "embedding", labels, embedding_matrix, target_count
    )
    embedding_target_cosines = [
        float(row["cosine"])
        for row in embedding_rows
        if row["pair_type"] == "target_target"
    ]
    all_pair_rows = list(embedding_rows)
    summaries = [
        summarize_space(
            embedding_rows,
            torch.linalg.vector_norm(embeddings.float(), dim=1),
            target_count,
            embedding_target_cosines,
        )
    ]
    write_matrix_csv(matrix_dir / "embedding.csv", labels, embedding_matrix)

    print(f"Loading original UNet weights: {checkpoint}")
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint, subfolder="unet", torch_dtype=dtype
    )
    layer_weights = [
        (name, value.detach().cpu())
        for name, value in unet.state_dict().items()
        if name.endswith("attn2.to_v.weight")
    ]
    if not layer_weights:
        raise RuntimeError("No attn2.to_v weights were found in the UNet")

    layer_matrices = []
    vectors = embeddings.float()
    for layer_index, (layer_name, layer_weight) in enumerate(layer_weights):
        output_vectors = vectors @ layer_weight.float().T
        matrix = pairwise_cosine(output_vectors, eps=eps).cpu()
        rows = build_pair_rows(
            layer_index, layer_name, labels, matrix, target_count
        )
        all_pair_rows.extend(rows)
        summaries.append(
            summarize_space(
                rows,
                torch.linalg.vector_norm(output_vectors, dim=1),
                target_count,
                embedding_target_cosines,
            )
        )
        layer_matrices.append((layer_name, matrix))
        write_matrix_csv(
            matrix_dir / f"layer_{layer_index:02d}.csv", labels, matrix
        )

    write_csv(output_dir / "pairwise_metrics.csv", all_pair_rows)
    write_csv(output_dir / "space_summary.csv", summaries)
    write_csv(
        output_dir / "labels.csv",
        [
            {
                "index": index,
                "label": label,
                "kind": "target" if index < target_count else "anchor",
            }
            for index, label in enumerate(labels)
        ],
    )
    make_embedding_heatmap(
        embedding_matrix,
        labels,
        target_count,
        output_dir / "embedding_pairwise_cosine_heatmap.png",
    )
    make_layer_heatmap_grid(
        layer_matrices,
        target_count,
        output_dir / "layer_pairwise_cosine_heatmaps.png",
    )
    make_pair_type_summary(
        summaries,
        output_dir / "layer_pair_type_summary.png",
    )

    metadata = {
        "checkpoint": checkpoint,
        "target_csv": str(csv_path),
        "target_count": target_count,
        "anchor_prompts": anchor_prompts,
        "labels": labels,
        "layer_count": len(layer_weights),
        "space_count": len(summaries),
        "unordered_pairs_per_space": len(embedding_rows),
        "pair_measurement_count": len(all_pair_rows),
        "pair_counts": {
            pair_type: sum(
                row["pair_type"] == pair_type for row in embedding_rows
            )
            for pair_type in PAIR_TYPES
        },
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)

    target_correlations = [
        row["target_correlation_to_embedding"] for row in summaries[1:]
    ]
    target_cosine_means = [
        row["target_target_cosine_mean"] for row in summaries[1:]
    ]
    print(
        "Target pairwise cosine analysis complete | "
        f"targets={target_count} | anchors=2 | layers={len(layer_weights)} | "
        f"pairs/space={len(embedding_rows)} | "
        f"layer target-target cosine mean range="
        f"{min(target_cosine_means):.6f}-{max(target_cosine_means):.6f} | "
        f"correlation-to-embedding range="
        f"{min(target_correlations):.6f}-{max(target_correlations):.6f}"
    )
    print(f"Artifacts: {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
