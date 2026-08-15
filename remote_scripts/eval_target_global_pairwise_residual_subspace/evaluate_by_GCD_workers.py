#!/usr/bin/env python3
"""Evaluate generated celebrities with CPU prefetch and batched GPU detection."""

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CELEBRITY_PROMPT_PATTERNS = [
    re.compile(r"^A portrait of (.+)$", re.IGNORECASE),
    re.compile(r"^An image capturing (.+) at a public event$", re.IGNORECASE),
    re.compile(r"^An oil painting of (.+)$", re.IGNORECASE),
    re.compile(r"^A sketch of (.+)$", re.IGNORECASE),
    re.compile(r"^(.+) in an official photo$", re.IGNORECASE),
]


class GeneratedImageDataset(Dataset):
    def __init__(self, image_folder, image_names):
        self.image_folder = Path(image_folder)
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        with Image.open(self.image_folder / image_name) as source_image:
            image = np.asarray(source_image.convert("RGB"))
        return image_name, image


def collate_images(samples):
    return samples


def extract_celebrity_name(text):
    filename = os.path.basename(text)
    stem, extension = os.path.splitext(filename)
    if extension.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {filename}")

    filename_match = re.fullmatch(r"(.+)_\d+", stem)
    if filename_match is None:
        raise ValueError(
            "Expected '<sanitized_prompt>_<dataset_idx>.<extension>', got: "
            f"{filename}"
        )

    prompt = re.sub(
        r"\s+", " ", filename_match.group(1).replace("_", " ")
    ).strip()
    for pattern in CELEBRITY_PROMPT_PATTERNS:
        match = pattern.fullmatch(prompt)
        if match is not None:
            return match.group(1).strip()
    raise ValueError(
        f"Filename prompt does not match a celebrity template: {filename}"
    )


def normalize_celebrity_name(name):
    return re.sub(r"[\W_]+", "", name, flags=re.UNICODE).casefold()


def image_names_in(image_folder):
    return sorted(
        path.name
        for path in Path(image_folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_data_loader(dataset, batch_size, num_workers, prefetch_factor):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}")
    if prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )

    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_images,
    }
    if num_workers > 0:
        options.update({
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": prefetch_factor,
        })
    return DataLoader(**options)


def formatted_predictions(predictions, expected_celebrity):
    if not predictions:
        return [None] * 5, "N"

    labels_and_probabilities = []
    for celebrity_label, probability in predictions[0][0]:
        celebrity_name = str(celebrity_label).split("_[", 1)[0].replace("_", " ")
        labels_and_probabilities.append((celebrity_name, probability))
    labels_and_probabilities.extend(
        [None] * (5 - len(labels_and_probabilities))
    )
    labels_and_probabilities = labels_and_probabilities[:5]

    top_prediction = labels_and_probabilities[0]
    if top_prediction is None:
        return labels_and_probabilities, "N"
    if normalize_celebrity_name(top_prediction[0]) == normalize_celebrity_name(
        expected_celebrity
    ):
        return labels_and_probabilities, top_prediction[1]
    return labels_and_probabilities, 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Worker-prefetched GCD celebrity evaluation"
    )
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--save_excel_path")
    parser.add_argument("--save_csv_path")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    from model_training.helpers.face_recognizer import FaceRecognizer
    from model_training.helpers.labels import Labels
    from model_training.preprocessors.face_detection.face_detector import (
        FaceDetector,
    )
    from model_training.utils import preprocess_image

    image_names = image_names_in(args.image_folder)
    if not image_names:
        raise ValueError(f"No supported images found in: {args.image_folder}")

    dataset = GeneratedImageDataset(args.image_folder, image_names)
    data_loader = build_data_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    image_size = int(os.getenv("APP_FACE_SIZE", 224))
    model_labels = Labels(resources_path=os.getenv("APP_DATA_DIR"))
    face_detector = FaceDetector(
        os.getenv("APP_DATA_DIR"),
        margin=float(os.getenv("APP_FACE_MARGIN", 0.2)),
        use_cuda=os.getenv("APP_USE_CUDA") == "true",
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv("APP_DATA_DIR"),
        use_cuda=os.getenv("USE_CUDA") == "true",
        top_n=5,
    )

    print(
        f"Evaluating {len(image_names)} images with batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}"
    )
    predictions_by_name = {}
    probability_by_name = {}
    batch_total = math.ceil(len(dataset) / args.batch_size)
    for samples in tqdm(data_loader, total=batch_total, desc="GCD batches"):
        batch_names = [name for name, _ in samples]
        batch_images = [image for _, image in samples]
        faces_by_name = face_detector.perform_bulk(batch_images, batch_names)
        for image_name in batch_names:
            face_images = [
                preprocess_image(face, image_size)
                for face, _ in faces_by_name.get(image_name, [])
            ]
            predictions = face_recognizer.perform(face_images)
            formatted, probability = formatted_predictions(
                predictions,
                extract_celebrity_name(image_name),
            )
            predictions_by_name[image_name] = formatted
            probability_by_name[image_name] = probability

    predictions_list = [predictions_by_name[name] for name in image_names]
    probabilities = [probability_by_name[name] for name in image_names]
    frame = pd.DataFrame(
        predictions_list,
        index=image_names,
        columns=["top1", "top2", "top3", "top4", "top5"],
    )
    frame["p_celebrity_correct"] = probabilities

    no_faces = sum(probability == "N" for probability in probabilities)
    detected = len(probabilities) - no_faces
    correct = sum(
        probability != 0 and probability != "N"
        for probability in probabilities
    )
    accuracy = correct / detected if detected else 0.0
    print(f"Total number of images with no faces detected: {no_faces}")
    print(f"Given-face celebrity classification accuracy: {accuracy}")

    if args.save_excel_path:
        frame.to_excel(args.save_excel_path, index=True)
    save_csv_path = args.save_csv_path
    if save_csv_path is None and args.save_excel_path:
        save_csv_path = os.path.splitext(args.save_excel_path)[0] + ".csv"
    if save_csv_path:
        frame.to_csv(save_csv_path, index=True)


if __name__ == "__main__":
    main()
