#!/usr/bin/env python3
"""Evaluate exact GCD probabilities for two configured celebrity identities."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def normalize_name(name):
    return re.sub(r"[\W_]+", "", name, flags=re.UNICODE).casefold()


def display_label(label):
    return str(label).split("_[", 1)[0].replace("_", " ")


class ImageDataset(Dataset):
    def __init__(self, directory):
        self.paths = sorted(
            path for path in Path(directory).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        return str(path.resolve()), pixels


def collate(samples):
    return samples


def probability_map(predictions):
    if not predictions:
        return {}, "", 0.0
    labels_and_probabilities, known_score = predictions[0]
    values = {
        normalize_name(display_label(label)): float(probability)
        for label, probability in labels_and_probabilities
    }
    top1 = display_label(labels_and_probabilities[0][0])
    return values, top1, float(known_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--prompt-identity", required=True)
    parser.add_argument("--target-a", required=True)
    parser.add_argument("--anchor-b", required=True)
    parser.add_argument("--model-state", choices=("original", "edit"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()

    from model_training.helpers.face_recognizer import FaceRecognizer
    from model_training.helpers.labels import Labels
    from model_training.preprocessors.face_detection.face_detector import FaceDetector
    from model_training.utils import preprocess_image

    dataset = ImageDataset(args.image_dir)
    if not dataset.paths:
        raise SystemExit(f"No images found in {args.image_dir}")
    loader_options = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
    }
    if args.num_workers:
        loader_options.update({
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        })
    loader = DataLoader(**loader_options)

    resources = os.environ["APP_DATA_DIR"]
    labels = Labels(resources_path=resources)
    # CE-Eval uses numpy argsort, so requesting all classes exposes the exact
    # softmax entries for A and B rather than treating absence from top-5 as 0.
    recognizer = FaceRecognizer(
        labels=labels,
        resources_path=resources,
        use_cuda=os.getenv("USE_CUDA") == "true",
        top_n=len(labels.labels_list),
    )
    detector = FaceDetector(
        resources,
        margin=float(os.getenv("APP_FACE_MARGIN", "0.2")),
        use_cuda=os.getenv("APP_USE_CUDA") == "true",
    )
    image_size = int(os.getenv("APP_FACE_SIZE", "224"))
    key_a = normalize_name(args.target_a)
    key_b = normalize_name(args.anchor_b)
    rows = []
    for samples in loader:
        paths = [path for path, _ in samples]
        images = [image for _, image in samples]
        faces_by_path = detector.perform_bulk(images, paths)
        for path in paths:
            faces = [
                preprocess_image(face, image_size)
                for face, _ in faces_by_path.get(path, [])
            ]
            predictions = recognizer.perform(faces)
            probabilities, top1, known_score = probability_map(predictions)
            rows.append({
                "image_path": path,
                "model_state": args.model_state,
                "prompt_identity": args.prompt_identity,
                "face_detected": bool(predictions),
                "top1_identity": top1,
                "p_target_a": probabilities.get(key_a, 0.0),
                "p_anchor_b": probabilities.get(key_b, 0.0),
                "face_known_score": known_score,
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved {len(rows)} per-image predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

