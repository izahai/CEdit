import importlib.machinery
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install stubs for optional heavy dependencies if needed
for name in ("pandas", "tqdm", "kmeans_pytorch", "diffusers"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
        if name == "tqdm":
            stub.tqdm = lambda values, **_: values
        elif name == "kmeans_pytorch":
            stub.kmeans = None
        elif name == "diffusers":
            stub.StableDiffusionPipeline = object
            stub.DPMSolverMultistepScheduler = object
        sys.modules[name] = stub

from remote_scripts.eval_zero_anchor.eval_metrics.compute_quick_metrics import (
    detect_default_image_dir,
    evaluate_metrics,
    format_summary_table,
    slice_three_way_image,
)


class MockEvaluator:
    """CPU-only synthetic evaluator for fast unit testing without loading neural weights."""
    def __init__(self):
        self.device = "cpu"

    def get_text_features(self, texts: list[str]) -> torch.Tensor:
        # Return synthetic normalized 16-dim vectors deterministically based on text length
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 100
            torch.manual_seed(seed)
            v = torch.randn(16)
            vectors.append(v / v.norm(dim=-1, keepdim=True))
        return torch.stack(vectors)

    def get_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        # Return synthetic normalized 16-dim vectors based on pixel mean
        vectors = []
        for img in images:
            px = list(img.getpixel((0, 0)))
            seed = sum(px) % 100
            torch.manual_seed(seed)
            v = torch.randn(16)
            vectors.append(v / v.norm(dim=-1, keepdim=True))
        return torch.stack(vectors)

    @staticmethod
    def cosine_similarity(feats1: torch.Tensor, feats2: torch.Tensor) -> torch.Tensor:
        return (feats1 * feats2).sum(dim=-1)


class EvalMetricsTests(unittest.TestCase):
    def test_slice_three_way_image(self):
        # Create a 1536x512 image with 3 distinct color bands
        full_img = Image.new("RGB", (1536, 512))
        # Band 1: Red (0..511)
        red_part = Image.new("RGB", (512, 512), color=(255, 0, 0))
        # Band 2: Green (512..1023)
        green_part = Image.new("RGB", (512, 512), color=(0, 255, 0))
        # Band 3: Blue (1024..1535)
        blue_part = Image.new("RGB", (512, 512), color=(0, 0, 255))

        full_img.paste(red_part, (0, 0))
        full_img.paste(green_part, (512, 0))
        full_img.paste(blue_part, (1024, 0))

        img_orig, img_art, img_zero = slice_three_way_image(full_img)

        self.assertEqual(img_orig.size, (512, 512))
        self.assertEqual(img_art.size, (512, 512))
        self.assertEqual(img_zero.size, (512, 512))

        self.assertEqual(img_orig.getpixel((50, 50)), (255, 0, 0))
        self.assertEqual(img_art.getpixel((50, 50)), (0, 255, 0))
        self.assertEqual(img_zero.getpixel((50, 50)), (0, 0, 255))

    def test_slice_rejects_invalid_dimensions(self):
        img = Image.new("RGB", (1000, 512))
        with self.assertRaises(ValueError):
            slice_three_way_image(img)

    def test_detect_default_image_dir(self):
        detected = detect_default_image_dir()
        self.assertTrue(isinstance(detected, str))
        self.assertTrue(len(detected) > 0)

    def test_format_summary_table(self):
        dummy_summary = {
            "target_erasure": {
                "concept_clip": {
                    "original": 0.28,
                    "art_anchor": 0.18,
                    "zero_anchor": 0.15,
                    "delta_zero_minus_art": -0.03,
                    "zero_erasure_rate_pct": 46.4,
                    "art_erasure_rate_pct": 35.7,
                },
                "prompt_clip": {
                    "original": 0.30,
                    "art_anchor": 0.22,
                    "zero_anchor": 0.19,
                    "delta_zero_minus_art": -0.03,
                },
            },
            "retain_preservation": {
                "artist_clip": {
                    "original": 0.26,
                    "art_anchor": 0.24,
                    "zero_anchor": 0.25,
                    "delta_zero_minus_art": 0.01,
                    "zero_preservation_ratio_pct": 96.1,
                    "art_preservation_ratio_pct": 92.3,
                },
                "prompt_clip": {
                    "original": 0.29,
                    "art_anchor": 0.27,
                    "zero_anchor": 0.28,
                    "delta_zero_minus_art": 0.01,
                },
                "image_fidelity_to_orig": {
                    "art_anchor": 0.88,
                    "zero_anchor": 0.91,
                    "delta_zero_minus_art": 0.03,
                },
            },
        }

        table_str = format_summary_table(dummy_summary)
        self.assertIn("Target Erasure", table_str)
        self.assertIn("Retain Style", table_str)
        self.assertIn("Image Fidelity to Orig", table_str)
        self.assertIn("Zero (Better)", table_str)

    def test_evaluate_metrics_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vg_dir = tmp_path / "van_gogh"
            rnd_dir = tmp_path / "random_artists"
            vg_dir.mkdir()
            rnd_dir.mkdir()

            # Create 10 dummy 1536x512 images for each
            dummy_img = Image.new("RGB", (1536, 512), color=(100, 150, 200))
            for i in range(10):
                dummy_img.save(vg_dir / f"{i:02d}_vg.png")
                dummy_img.save(rnd_dir / f"{i:02d}_rnd.png")

            # Create minimal style csv with concept header
            style_csv = tmp_path / "style.csv"
            with open(style_csv, "w", encoding="utf-8") as f:
                f.write("concept\nVan Gogh\n")
                for i in range(15):
                    f.write(f"Artist_{i}\n")

            mock_evaluator = MockEvaluator()
            summary, rows = evaluate_metrics(
                evaluator=mock_evaluator,
                image_dir=tmp_path,
                style_csv=style_csv,
                seed=42,
            )

            self.assertIn("target_erasure", summary)
            self.assertIn("retain_preservation", summary)
            self.assertEqual(len(rows), 20)  # 10 target + 10 retain
            self.assertIn("orig_concept_clip", rows[0])
            self.assertIn("art_concept_clip", rows[0])
            self.assertIn("zero_concept_clip", rows[0])

    def test_evaluate_metrics_100_mocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vg_dir = tmp_path / "van_gogh"
            rnd_dir = tmp_path / "random_artists"
            vg_dir.mkdir()
            rnd_dir.mkdir()

            # Create 100 dummy 1536x512 images for each
            dummy_img = Image.new("RGB", (1536, 512), color=(50, 100, 150))
            for i in range(1, 101):
                dummy_img.save(vg_dir / f"{i:03d}_van_gogh_prompt_{i}.png")
                dummy_img.save(rnd_dir / f"{i:03d}_artist_{i}_prompt_{i}.png")

            # Create style csv with 105 concepts
            style_csv = tmp_path / "style.csv"
            with open(style_csv, "w", encoding="utf-8") as f:
                f.write("concept\nVan Gogh\n")
                for i in range(1, 105):
                    f.write(f"Artist_{i}\n")

            mock_evaluator = MockEvaluator()
            summary, rows = evaluate_metrics(
                evaluator=mock_evaluator,
                image_dir=tmp_path,
                style_csv=style_csv,
                seed=42,
                prompt_mode="canonical_5",
            )

            self.assertEqual(summary["num_target_images"], 100)
            self.assertEqual(summary["num_retain_images"], 100)
            self.assertEqual(summary["prompt_mode"], "canonical_5")
            self.assertEqual(len(rows), 200)  # 100 target + 100 retain
            self.assertIn("orig_concept_clip", rows[0])
            self.assertIn("art_concept_clip", rows[0])
            self.assertIn("zero_concept_clip", rows[0])


if __name__ == "__main__":
    unittest.main()
