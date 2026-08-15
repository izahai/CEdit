import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from remote_scripts.eval_target_global_pairwise_residual_subspace.evaluate_by_GCD_workers import (
    GeneratedImageDataset,
    build_data_loader,
    extract_celebrity_name,
    formatted_predictions,
    image_names_in,
)


class GcdWorkerEvaluationTests(unittest.TestCase):
    def test_cpu_workers_load_images_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            image = np.full((16, 16, 3), 127, dtype=np.uint8)
            for image_name in (
                "A_portrait_of_Grace_Hopper_1.png",
                "A_portrait_of_Ada_Lovelace_0.png",
            ):
                Image.fromarray(image).save(image_dir / image_name)

            image_names = image_names_in(image_dir)
            loader = build_data_loader(
                GeneratedImageDataset(image_dir, image_names),
                batch_size=2,
                num_workers=2,
                prefetch_factor=2,
            )
            samples = next(iter(loader))

        self.assertEqual(
            [name for name, _ in samples],
            [
                "A_portrait_of_Ada_Lovelace_0.png",
                "A_portrait_of_Grace_Hopper_1.png",
            ],
        )
        self.assertTrue(all(image.shape == (16, 16, 3) for _, image in samples))

    def test_extracts_sanitized_celebrity_prompt(self):
        self.assertEqual(
            extract_celebrity_name("An_oil_painting_of_Ada_Lovelace_17.png"),
            "Ada Lovelace",
        )

    def test_formats_correct_and_incorrect_top_predictions(self):
        predictions = [
            [
                [
                    ("Ada_Lovelace_[0]", 0.75),
                    ("Grace_Hopper_[1]", 0.20),
                ]
            ]
        ]

        formatted, correct_probability = formatted_predictions(
            predictions,
            "Ada Lovelace",
        )
        _, incorrect_probability = formatted_predictions(
            predictions,
            "Grace Hopper",
        )

        self.assertEqual(formatted[0], ("Ada Lovelace", 0.75))
        self.assertEqual(correct_probability, 0.75)
        self.assertEqual(incorrect_probability, 0)


if __name__ == "__main__":
    unittest.main()
