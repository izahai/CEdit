#!/usr/bin/env python3
"""
SVD experiment for the Stable Diffusion 1.4 CLIP text encoder.

What it does
------------
1. Loads tokenizer + CLIPTextModel from CompVis/stable-diffusion-v1-4.
2. Builds three prompt groups:
   - synonyms: same concept, paraphrased
   - same_template: different concepts, same linguistic templates
   - diverse: varied concepts and sentence structures
3. Extracts one representation per prompt using:
   - eos: hidden state at the EOS token (default)
   - mean: attention-mask-weighted mean of token hidden states
   - flatten: flattened [77, 768] SD conditioning matrix
4. Runs SVD both before and after centering.
5. Saves spectra, cumulative explained variance, PCA scatter, cosine
   similarity distributions, CSV metrics, prompts, and raw embeddings.

Example
-------
python clip_sd14_svd_experiment.py --output-dir results
python clip_sd14_svd_experiment.py --representation mean --normalize
python clip_sd14_svd_experiment.py --representation flatten --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


MODEL_ID = "CompVis/stable-diffusion-v1-4"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _deduplicate(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = " ".join(item.strip().split())
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_prompt_groups(seed: int = 42) -> Dict[str, List[str]]:
    """
    Construct controlled prompt groups of equal size.

    The synonym group targets one semantic concept: a red sports car on a
    mountain road at sunset. Variations alter wording while preserving meaning.
    """
    synonym_subjects = [
        "a red sports car",
        "a crimson sports automobile",
        "a scarlet performance car",
        "a red high-performance vehicle",
        "a ruby-colored sports coupe",
        "a bright red racing-style car",
        "a sleek red sports vehicle",
        "a red performance coupe",
        "a vermilion sports car",
        "a cherry-red fast car",
    ]
    synonym_actions = [
        "driving along",
        "traveling on",
        "moving down",
        "cruising along",
        "speeding through",
        "riding on",
    ]
    synonym_places = [
        "a winding mountain road",
        "a curving road in the mountains",
        "a serpentine alpine highway",
        "a twisting hillside road",
        "a bendy mountain pass",
        "a meandering road through the mountains",
    ]
    synonym_times = [
        "at sunset",
        "during sunset",
        "at dusk with the sun setting",
        "in the golden light of sunset",
        "under an evening sunset",
    ]
    synonym_styles = [
        "",
        "a photograph of",
        "a realistic photo of",
        "a detailed image of",
        "a cinematic photograph of",
        "a high-quality picture of",
    ]

    synonyms = []
    rng = random.Random(seed)
    combinations = [
        (s, a, p, t, st)
        for s in synonym_subjects
        for a in synonym_actions
        for p in synonym_places
        for t in synonym_times
        for st in synonym_styles
    ]
    rng.shuffle(combinations)
    for subject, action, place, time, style in combinations:
        core = f"{subject} {action} {place} {time}"
        synonyms.append(f"{style} {core}".strip())
        if len(synonyms) >= 120:
            break

    # Different meanings, deliberately held under the same small template set.
    concepts = [
        "a golden retriever playing in snow",
        "a blue ceramic teapot on a wooden table",
        "an astronaut walking on the moon",
        "a bowl of fresh strawberries",
        "a medieval castle above a river",
        "a violin resting beside sheet music",
        "a green frog sitting on a leaf",
        "a crowded subway station",
        "a lighthouse during a storm",
        "a child flying a colorful kite",
        "a robot cooking in a kitchen",
        "a field of purple lavender",
        "an old library filled with books",
        "a surfer riding a large wave",
        "a white owl in a dark forest",
        "a glass skyscraper in a modern city",
        "a plate of sushi and wasabi",
        "a steam locomotive crossing a bridge",
        "a coral reef with tropical fish",
        "a cyclist climbing a steep hill",
        "a wooden cabin beside a frozen lake",
        "a scientist looking through a microscope",
        "a hot-air balloon above farmland",
        "a black cat sleeping on a sofa",
        "a farmer driving a tractor",
        "a ballerina performing on stage",
        "a waterfall in a tropical jungle",
        "a chessboard in the middle of a game",
        "a cargo ship entering a harbor",
        "a chef decorating a wedding cake",
    ]
    fixed_templates = [
        "a photo of {}",
        "a realistic photograph of {}",
        "a detailed image of {}",
        "a cinematic picture of {}",
    ]
    same_template = [
        template.format(concept)
        for concept in concepts
        for template in fixed_templates
    ]

    # Broad semantic and syntactic variation.
    diverse = [
        "macro photograph of dew drops on a spider web",
        "minimalist logo shaped like a paper crane",
        "the feeling of nostalgia represented as abstract art",
        "two engineers discussing a circuit diagram",
        "aerial view of rice terraces after rain",
        "charcoal sketch of an elderly musician",
        "isometric cutaway of a space habitat",
        "underwater ruins illuminated by divers",
        "a tiny bakery inside a tree trunk",
        "news photograph of a marathon finish line",
        "an exploded technical diagram of a mechanical watch",
        "soft watercolor landscape with distant mountains",
        "three red cubes casting long shadows",
        "a funny cartoon about debugging software",
        "portrait of a queen in renaissance clothing",
        "satellite image of a spiral hurricane",
        "street food market at midnight",
        "a handwritten note next to a cup of coffee",
        "futuristic medical laboratory with holograms",
        "close-up product photo of wireless headphones",
        "ancient cave paintings under torchlight",
        "aerial drone shot of a coastal highway",
        "children's book illustration of a friendly dragon",
        "dense mathematical equations on a blackboard",
        "fashion editorial in a brutalist building",
        "a peaceful zen garden covered in snow",
        "low-poly model of a tropical island",
        "infrared photograph of a sleeping fox",
        "architectural blueprint for a small museum",
        "a claymation scene of vegetables dancing",
        "concert audience holding glowing phones",
        "microscopic view of plant cells",
        "an empty football stadium before sunrise",
        "a paper collage of birds migrating south",
        "a detective examining evidence in an office",
        "a neon sign reflected in a rainy street",
        "food photography of noodles with chili oil",
        "a vintage travel poster for an imaginary planet",
        "a family having dinner in a small apartment",
        "diagram showing the water cycle",
        "a monochrome portrait with dramatic side lighting",
        "a fantasy map drawn on aged parchment",
        "workers assembling solar panels on a roof",
        "a close-up of hands shaping wet pottery",
        "a quiet train compartment at night",
        "a surreal staircase floating above clouds",
        "a scientific illustration of a blue whale",
        "an abandoned amusement park overgrown with plants",
        "a cozy reading corner with warm lamplight",
        "a security camera view of an empty warehouse",
        "a mosaic made from pieces of colored glass",
        "an orchestra rehearsing in a concert hall",
        "a snowplow clearing a rural road",
        "a museum display of dinosaur fossils",
        "a computer-generated fluid simulation",
        "a crowded election debate stage",
        "a hand-drawn storyboard for an action sequence",
        "a small sailboat under the northern lights",
        "a geometric tattoo design inspired by mountains",
        "aerial photograph of shipping containers arranged in rows",
        "a beekeeper inspecting a honeycomb",
        "a ceramic sculpture with an uneven glazed surface",
        "a subway map for a fictional city",
        "an athlete tying running shoes before a race",
        "a long-exposure image of stars above a desert",
        "a classroom demonstration of a chemical reaction",
        "a bright kitchen photographed for a real-estate listing",
        "a lonely bench beside a foggy lake",
        "a 3D render of an ergonomic office chair",
        "a crowded coral-colored festival parade",
        "an archaeological excavation viewed from above",
        "a stack of pancakes shaped like a bear",
        "a cinematic still of explorers entering a cavern",
        "a botanical illustration of medicinal herbs",
        "a repair technician opening a laptop",
        "an ink drawing of boats in a harbor",
        "a futuristic electric bus at a charging station",
        "a documentary photo of fishermen pulling a net",
        "a stained-glass window depicting the night sky",
        "a clean infographic about renewable energy",
        "a dog wearing safety goggles in a workshop",
        "a luxury perfume bottle on polished stone",
        "an ice cave glowing blue from within",
        "a crowded flea market photographed from eye level",
        "a topographic model of a volcanic island",
        "a chef tossing vegetables in a flaming wok",
        "an origami city built from newspaper",
        "a cyclist reflected in a shop window",
        "a bedroom transformed into a miniature jungle",
        "a black-and-white photograph of factory machinery",
        "a ceramic tile pattern based on ocean waves",
        "a rescue helicopter above a flooded valley",
        "a diagram of a neural network drawn with chalk",
        "a theatrical mask lying on red velvet",
        "a rural bus stop in heavy rain",
        "an editorial portrait of a climate scientist",
        "a toy train moving around a Christmas tree",
        "a transparent anatomical model of the human heart",
        "a rooftop greenhouse in a dense city",
        "a close-up of an old mechanical typewriter",
        "aerial image of a river delta",
        "a miniature diorama of a busy airport",
        "a cubist painting of musicians",
        "a solar eclipse seen above a mountain ridge",
        "a food truck serving customers at lunchtime",
        "a technical cutaway of an electric motor",
        "a quiet monastery courtyard in autumn",
        "a motion-blurred photograph of commuters",
        "a glowing jellyfish in deep ocean water",
        "a handmade quilt with a geometric pattern",
        "a photojournalist standing in a dusty street",
        "a library robot returning books to shelves",
        "a windswept tree on a rocky coastline",
        "a glass of sparkling water with lemon",
        "a nighttime construction site illuminated by floodlights",
        "a retro computer terminal displaying green text",
        "a group of hikers crossing a suspension bridge",
        "a model train landscape with tiny villages",
        "an abstract visualization of sound waves",
        "a historic observatory beneath a clear sky",
    ]

    groups = {
        "synonyms": _deduplicate(synonyms),
        "same_template": _deduplicate(same_template),
        "diverse": _deduplicate(diverse),
    }

    n = min(len(v) for v in groups.values())
    # Equal group sizes are important for direct comparison.
    for key in groups:
        groups[key] = groups[key][:n]

    return groups


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_prompts(
    prompts: Sequence[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    batch_size: int,
    representation: str,
    normalize: bool,
) -> np.ndarray:
    vectors: List[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        tokens = tokenizer(
            list(batch_prompts),
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state.float()  # [B, 77, 768]

        if representation == "eos":
            # CLIP's tokenizer places EOS at the highest token id in each row
            # for this vocabulary; argmax is the convention used by CLIP.
            eos_positions = input_ids.argmax(dim=-1)
            rows = torch.arange(hidden.shape[0], device=device)
            vector = hidden[rows, eos_positions]
        elif representation == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        elif representation == "flatten":
            vector = hidden.reshape(hidden.shape[0], -1)
        else:
            raise ValueError(f"Unknown representation: {representation}")

        if normalize:
            vector = torch.nn.functional.normalize(vector, p=2, dim=-1)

        vectors.append(vector.cpu())

    return torch.cat(vectors, dim=0).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class SpectrumMetrics:
    group: str
    centered: bool
    n_samples: int
    dimension: int
    numerical_rank: int
    r90: int
    r95: int
    r99: int
    top1_ratio: float
    top5_ratio: float
    participation_ratio: float
    entropy_effective_rank: float
    condition_number_nonzero: float
    mean_pairwise_cosine: float
    std_pairwise_cosine: float


def pairwise_cosines(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norm, 1e-12)
    sim = xn @ xn.T
    upper = np.triu_indices(sim.shape[0], k=1)
    return sim[upper]


def analyze_matrix(
    x: np.ndarray,
    group: str,
    centered: bool,
) -> Tuple[SpectrumMetrics, Dict[str, np.ndarray]]:
    original = x
    work = x - x.mean(axis=0, keepdims=True) if centered else x.copy()

    # full_matrices=False gives min(N, D) singular values.
    _, singular_values, vh = np.linalg.svd(work, full_matrices=False)

    denominator = max(work.shape[0] - 1, 1) if centered else max(work.shape[0], 1)
    eigenvalues = singular_values**2 / denominator
    total = eigenvalues.sum()

    if total <= 0:
        ratios = np.zeros_like(eigenvalues)
    else:
        ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)

    def rank_at(frac: float) -> int:
        if total <= 0:
            return 0
        return int(np.searchsorted(cumulative, frac, side="left") + 1)

    tol = singular_values.max(initial=0.0) * max(work.shape) * np.finfo(np.float64).eps
    nonzero = singular_values[singular_values > tol]
    numerical_rank = int(nonzero.size)
    condition = (
        float(nonzero[0] / nonzero[-1])
        if nonzero.size >= 2
        else float("inf")
    )

    positive_ratios = ratios[ratios > 0]
    entropy_rank = (
        float(np.exp(-np.sum(positive_ratios * np.log(positive_ratios))))
        if positive_ratios.size
        else 0.0
    )
    participation_ratio = (
        float(1.0 / np.sum(ratios**2))
        if np.sum(ratios**2) > 0
        else 0.0
    )

    cosines = pairwise_cosines(original)

    metrics = SpectrumMetrics(
        group=group,
        centered=centered,
        n_samples=x.shape[0],
        dimension=x.shape[1],
        numerical_rank=numerical_rank,
        r90=rank_at(0.90),
        r95=rank_at(0.95),
        r99=rank_at(0.99),
        top1_ratio=float(ratios[0]) if ratios.size else 0.0,
        top5_ratio=float(ratios[:5].sum()),
        participation_ratio=participation_ratio,
        entropy_effective_rank=entropy_rank,
        condition_number_nonzero=condition,
        mean_pairwise_cosine=float(cosines.mean()) if cosines.size else float("nan"),
        std_pairwise_cosine=float(cosines.std()) if cosines.size else float("nan"),
    )

    artifacts = {
        "singular_values": singular_values,
        "eigenvalues": eigenvalues,
        "explained_ratio": ratios,
        "cumulative_ratio": cumulative,
        "principal_axes": vh,
        "centered_matrix": work,
        "pairwise_cosines": cosines,
    }
    return metrics, artifacts


def project_joint_pca(group_embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    labels = list(group_embeddings)
    sizes = [len(group_embeddings[k]) for k in labels]
    x = np.concatenate([group_embeddings[k] for k in labels], axis=0)
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    coords = u[:, :2] * s[:2]

    result = {}
    offset = 0
    for label, size in zip(labels, sizes):
        result[label] = coords[offset : offset + size]
        offset += size
    return result


# ---------------------------------------------------------------------------
# Plotting and output
# ---------------------------------------------------------------------------

def save_spectrum_plot(
    analyses: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    title: str,
    log_scale: bool,
) -> None:
    plt.figure(figsize=(9, 6))
    for group, artifact in analyses.items():
        y = artifact["explained_ratio"]
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=group, linewidth=1.8)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Component index")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_cumulative_plot(
    analyses: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(9, 6))
    for group, artifact in analyses.items():
        y = artifact["cumulative_ratio"]
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=group, linewidth=1.8)
    for threshold in (0.90, 0.95, 0.99):
        plt.axhline(threshold, linestyle="--", linewidth=0.8)
    plt.ylim(0.0, 1.01)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_pca_plot(coords: Dict[str, np.ndarray], output_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    for group, xy in coords.items():
        plt.scatter(xy[:, 0], xy[:, 1], s=24, alpha=0.75, label=group)
    plt.xlabel("Joint PC1")
    plt.ylabel("Joint PC2")
    plt.title("Joint PCA of all prompt embeddings")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_cosine_histogram(
    analyses: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    for group, artifact in analyses.items():
        values = artifact["pairwise_cosines"]
        plt.hist(values, bins=40, alpha=0.45, density=True, label=group)
    plt.xlabel("Pairwise cosine similarity")
    plt.ylabel("Density")
    plt.title("Within-group cosine similarity")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metrics_csv(metrics: Sequence[SpectrumMetrics], path: Path) -> None:
    rows = [asdict(m) for m in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(metrics: Sequence[SpectrumMetrics]) -> None:
    print()
    print("=" * 118)
    print(
        f"{'group':<16} {'center':<7} {'N':>4} {'D':>7} {'rank':>6} "
        f"{'r90':>5} {'r95':>5} {'r99':>5} {'top1':>9} {'top5':>9} "
        f"{'PR-rank':>9} {'H-rank':>9} {'mean cos':>10}"
    )
    print("-" * 118)
    for m in metrics:
        print(
            f"{m.group:<16} {str(m.centered):<7} {m.n_samples:>4} "
            f"{m.dimension:>7} {m.numerical_rank:>6} {m.r90:>5} "
            f"{m.r95:>5} {m.r99:>5} {m.top1_ratio:>9.4f} "
            f"{m.top5_ratio:>9.4f} {m.participation_ratio:>9.2f} "
            f"{m.entropy_effective_rank:>9.2f} "
            f"{m.mean_pairwise_cosine:>10.4f}"
        )
    print("=" * 118)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the SVD spectrum of SD 1.4 CLIP text embeddings."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("sd14_clip_svd_results"))
    parser.add_argument(
        "--representation",
        choices=["eos", "mean", "flatten"],
        default="eos",
        help=(
            "eos: EOS-token vector; mean: masked token mean; "
            "flatten: full [77,768] conditioning flattened."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each prompt vector before SVD.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-prompts-per-group",
        type=int,
        default=None,
        help="Optional cap. Keep this equal across groups.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16"],
        default="auto",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        if device.type == "cpu":
            raise ValueError("float16 on CPU is unsupported/slow; use float32.")
        return torch.float16
    return torch.float16 if device.type in {"cuda", "mps"} else torch.float32


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Model dtype: {dtype}")
    print(f"Representation: {args.representation}")
    print(f"L2 normalization: {args.normalize}")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder.eval().to(device)

    groups = build_prompt_groups(args.seed)
    if args.max_prompts_per_group is not None:
        groups = {
            name: prompts[: args.max_prompts_per_group]
            for name, prompts in groups.items()
        }

    # Save exact prompts for reproducibility.
    with (output_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    embeddings: Dict[str, np.ndarray] = {}
    for group, prompts in groups.items():
        print(f"Encoding {group}: {len(prompts)} prompts...")
        embeddings[group] = encode_prompts(
            prompts=prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            batch_size=args.batch_size,
            representation=args.representation,
            normalize=args.normalize,
        )
        np.save(output_dir / f"embeddings_{group}.npy", embeddings[group])
        print(f"  shape = {embeddings[group].shape}")

    all_metrics: List[SpectrumMetrics] = []
    raw_analyses: Dict[str, Dict[str, np.ndarray]] = {}
    centered_analyses: Dict[str, Dict[str, np.ndarray]] = {}

    for group, x in embeddings.items():
        raw_metrics, raw_artifact = analyze_matrix(x, group, centered=False)
        centered_metrics, centered_artifact = analyze_matrix(x, group, centered=True)

        all_metrics.extend([raw_metrics, centered_metrics])
        raw_analyses[group] = raw_artifact
        centered_analyses[group] = centered_artifact

        np.savez_compressed(
            output_dir / f"spectrum_{group}.npz",
            raw_singular_values=raw_artifact["singular_values"],
            raw_eigenvalues=raw_artifact["eigenvalues"],
            raw_explained_ratio=raw_artifact["explained_ratio"],
            centered_singular_values=centered_artifact["singular_values"],
            centered_eigenvalues=centered_artifact["eigenvalues"],
            centered_explained_ratio=centered_artifact["explained_ratio"],
        )

    save_metrics_csv(all_metrics, output_dir / "metrics.csv")
    print_metrics(all_metrics)

    save_spectrum_plot(
        raw_analyses,
        output_dir / "spectrum_raw_linear.png",
        "Uncentered spectrum",
        log_scale=False,
    )
    save_spectrum_plot(
        raw_analyses,
        output_dir / "spectrum_raw_log.png",
        "Uncentered spectrum (log scale)",
        log_scale=True,
    )
    save_spectrum_plot(
        centered_analyses,
        output_dir / "spectrum_centered_linear.png",
        "Centered covariance spectrum",
        log_scale=False,
    )
    save_spectrum_plot(
        centered_analyses,
        output_dir / "spectrum_centered_log.png",
        "Centered covariance spectrum (log scale)",
        log_scale=True,
    )
    save_cumulative_plot(
        raw_analyses,
        output_dir / "cumulative_raw.png",
        "Uncentered cumulative energy",
    )
    save_cumulative_plot(
        centered_analyses,
        output_dir / "cumulative_centered.png",
        "Centered cumulative explained variance",
    )
    save_cosine_histogram(raw_analyses, output_dir / "cosine_histogram.png")

    joint_coords = project_joint_pca(embeddings)
    save_pca_plot(joint_coords, output_dir / "joint_pca.png")

    config = {
        "model_id": args.model_id,
        "representation": args.representation,
        "normalize": args.normalize,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "notes": {
            "uncentered": (
                "Dominated by the common mean direction when prompts in a "
                "group point similarly."
            ),
            "centered": (
                "Measures within-group variation after removing the group mean."
            ),
            "rank_limit": (
                "For centered data, algebraic rank is at most N-1, so do not "
                "interpret zero tail eigenvalues as model-wide low rank."
            ),
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all outputs to: {output_dir.resolve()}")
    print("\nInterpretation checklist:")
    print("  1. Compare synonyms vs same_template vs diverse using CENTERED results.")
    print("  2. Inspect r95, participation_ratio, and entropy_effective_rank.")
    print("  3. A huge raw top-1 component mostly reflects the common mean direction.")
    print("  4. Centered rank cannot exceed N-1.")
    print("  5. Repeat with several seeds/concepts before making a general claim.")


if __name__ == "__main__":
    main()
