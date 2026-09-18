#!/usr/bin/env python3
"""Compute full-encoder CLIP text cosine similarity between Van Gogh and style artists.

Uses openai/clip-vit-large-patch14 with CLIPTextModelWithProjection, which matches the
underlying text encoder in Stable Diffusion v1.4 while including the trained multi-modal
text projection layer to produce a single normalized 768-d embedding per prompt.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP cosine similarities between a target concept and artists in a CSV."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Van Gogh",
        help="Target concept to compare against (default: 'Van Gogh').",
    )
    parser.add_argument(
        "--style_csv",
        type=str,
        default="data/style.csv",
        help="Path to CSV containing candidate style concepts (default: 'data/style.csv').",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="remote_scripts/eval_ReSAM/erase_van_gogh/van_gogh_style_similarity.csv",
        help="Destination path for output ranking CSV.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model ID for CLIP text encoder with projection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for text encoding (default: 128).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda', 'cpu', 'mps').",
    )
    return parser.parse_args()


def load_artists(csv_path: str | Path) -> List[Dict[str, str]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Style CSV not found: {path}")

    artists = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "concept" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {path} must contain a 'concept' column.")
        has_id = "id" in (reader.fieldnames or [])

        for idx, row in enumerate(reader, start=1):
            concept = (row.get("concept") or "").strip()
            if not concept:
                continue
            item_id = (row.get("id") or "").strip() if has_id else str(idx)
            artists.append({"id": item_id, "artist": concept})

    if not artists:
        raise ValueError(f"No valid artist entries found in {path}")
    return artists


@torch.no_grad()
def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: torch.device | str,
    batch_size: int = 128,
) -> torch.Tensor:
    """Encode a list of prompt strings into L2-normalized CLIP projection embeddings."""
    device = torch.device(device)
    embeddings: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)
        # text_embeds is [batch_size, projection_dim], projected from the pooled EOS token
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            batch_embeds = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            batch_embeds = outputs.pooler_output
        else:
            # Fallback to last hidden state at EOS position
            last_hidden = outputs.last_hidden_state
            eos_indices = inputs.attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(batch), device=device)
            batch_embeds = last_hidden[batch_indices, eos_indices]

        # L2-normalize embeddings to unit sphere
        norm = batch_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        batch_embeds = batch_embeds / norm
        embeddings.append(batch_embeds.cpu().float())

    return torch.cat(embeddings, dim=0)


def compute_similarity_rankings(
    target_emb: torch.Tensor,
    artist_embs: torch.Tensor,
    artists: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Compute cosine similarities and sort descending by similarity score."""
    # Cosine similarity between normalized vectors is inner product
    cosines = (artist_embs @ target_emb.view(-1, 1)).squeeze(-1)

    sorted_indices = torch.argsort(cosines, descending=True).tolist()

    ranked_rows: List[Dict[str, Any]] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        artist_info = artists[idx]
        sim_val = float(cosines[idx].item())
        ranked_rows.append(
            {
                "rank": rank,
                "id": artist_info["id"],
                "artist": artist_info["artist"],
                "similarity": f"{sim_val:.6f}",
            }
        )
    return ranked_rows


def write_csv(output_path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "id", "artist", "similarity"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading candidate artists from: {args.style_csv}")
    artists = load_artists(args.style_csv)
    print(f"Loaded {len(artists)} artists.")

    print(f"Loading CLIP model and tokenizer from: {args.model_id}")
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.model_id).to(device)
    text_encoder.eval()

    print(f"Encoding target concept: '{args.target}'")
    target_emb = encode_texts(text_encoder, tokenizer, [args.target], device, batch_size=1)

    print(f"Encoding {len(artists)} artist concepts on {device} (batch size {args.batch_size})...")
    artist_texts = [item["artist"] for item in artists]
    artist_embs = encode_texts(text_encoder, tokenizer, artist_texts, device, batch_size=args.batch_size)

    print("Computing cosine similarity rankings...")
    ranked_rows = compute_similarity_rankings(target_emb, artist_embs, artists)

    output_path = Path(args.output_csv)
    write_csv(output_path, ranked_rows)
    print(f"Successfully saved {len(ranked_rows)} ranked records to: {output_path}")

    print("\n--- Top 15 Most Similar Artists to Van Gogh ---")
    for row in ranked_rows[:15]:
        print(f"Rank {row['rank']:>3} | ID: {row['id']:>4} | Sim: {row['similarity']} | Artist: {row['artist']}")

    print("\n--- Bottom 5 Least Similar Artists to Van Gogh ---")
    for row in ranked_rows[-5:]:
        print(f"Rank {row['rank']:>4} | ID: {row['id']:>4} | Sim: {row['similarity']} | Artist: {row['artist']}")


if __name__ == "__main__":
    main()

