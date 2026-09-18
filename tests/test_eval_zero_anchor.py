import sys
from pathlib import Path
import tempfile
import types
import unittest

from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install stubs for optional heavy dependencies if needed
import importlib.machinery

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

try:
    from remote_scripts.eval_zero_anchor.preview_concat import (
        CANONICAL_STYLE_TEMPLATES,
        build_random_artist_prompts,
        build_retain_100_prompts,
        build_van_gogh_100_prompts,
        build_van_gogh_prompts,
        concat_three_images,
        load_random_artists,
        slugify,
    )
except ModuleNotFoundError:
    from remote_scripts.eval_zero_anchor.eval_zero_anchor_img.preview_concat import (
        CANONICAL_STYLE_TEMPLATES,
        build_random_artist_prompts,
        build_retain_100_prompts,
        build_van_gogh_100_prompts,
        build_van_gogh_prompts,
        concat_three_images,
        load_random_artists,
        slugify,
    )
from train_erase_null import parse_args


def find_config(filename: str) -> Path:
    cand1 = REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / filename
    if cand1.exists():
        return cand1
    cand2 = REPO_ROOT / "remote_scripts" / "eval_zero_anchor" / "eval_zero_anchor_img" / filename
    if cand2.exists():
        return cand2
    return cand1


class EvalZeroAnchorWorkflowTests(unittest.TestCase):
    def test_train_art_yaml_validity_and_options(self):
        config_path = find_config("train_art.yaml")
        self.assertTrue(config_path.exists(), f"Missing config: {config_path}")

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["target_concepts"], "Van Gogh")
        self.assertEqual(data["anchor_concepts"], "art")
        self.assertEqual(data["baseline"], "SPEED")
        self.assertEqual(data["params"], "V")
        self.assertEqual(data["retain_projection_rank"], 100)

        _, args = parse_args(["--config", str(config_path)])
        self.assertEqual(args.target_concepts, "Van Gogh")
        self.assertEqual(args.anchor_concepts, "art")

    def test_train_zero_yaml_validity_and_options(self):
        config_path = find_config("train_zero.yaml")
        self.assertTrue(config_path.exists(), f"Missing config: {config_path}")

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["target_concepts"], "Van Gogh")
        self.assertEqual(data["anchor_concepts"], "<zero>")
        self.assertEqual(data["baseline"], "SPEED")
        self.assertEqual(data["params"], "V")
        self.assertEqual(data["retain_projection_rank"], 100)

        _, args = parse_args(["--config", str(config_path)])
        self.assertEqual(args.target_concepts, "Van Gogh")
        self.assertEqual(args.anchor_concepts, "<zero>")

    def test_van_gogh_prompt_generation(self):
        prompts = build_van_gogh_prompts(10)
        self.assertEqual(len(prompts), 10)
        for prompt in prompts:
            self.assertIn("Van Gogh", prompt)
            self.assertTrue(len(prompt) > 10)

    def test_random_artist_sampling_and_prompts(self):
        style_csv = REPO_ROOT / "data" / "style.csv"
        artists = load_random_artists(style_csv, target="Van Gogh", count=10, seed=42)
        self.assertEqual(len(artists), 10)
        self.assertEqual(len(set(artists)), 10)
        self.assertNotIn("Van Gogh", artists)

        # Reproducibility check with same seed
        artists_repeat = load_random_artists(style_csv, target="Van Gogh", count=10, seed=42)
        self.assertEqual(artists, artists_repeat)

        pairs = build_random_artist_prompts(artists)
        self.assertEqual(len(pairs), 10)
        for idx, (artist, prompt) in enumerate(pairs):
            self.assertEqual(artist, artists[idx])
            self.assertIn(artist, prompt)

    def test_concat_three_images_dimensions_and_order(self):
        # Create 3 images with distinct colors: Red (Original), Green (Art), Blue (Zero)
        img_orig = Image.new("RGB", (512, 512), color=(255, 0, 0))
        img_art = Image.new("RGB", (512, 512), color=(0, 255, 0))
        img_zero = Image.new("RGB", (512, 512), color=(0, 0, 255))

        combined = concat_three_images(img_orig, img_art, img_zero)
        self.assertEqual(combined.size, (1536, 512))

        # Check colors in the 3 regions
        # Region 1 (0..511): Red (Orig)
        self.assertEqual(combined.getpixel((100, 100)), (255, 0, 0))
        # Region 2 (512..1023): Green (Art)
        self.assertEqual(combined.getpixel((600, 100)), (0, 255, 0))
        # Region 3 (1024..1535): Blue (Zero)
        self.assertEqual(combined.getpixel((1200, 100)), (0, 0, 255))

    def test_concat_three_images_rejects_height_mismatch(self):
        img1 = Image.new("RGB", (512, 512))
        img2 = Image.new("RGB", (512, 256))
        img3 = Image.new("RGB", (512, 512))
        with self.assertRaises(ValueError):
            concat_three_images(img1, img2, img3)

    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "Hello_World")
        self.assertEqual(slugify("Van Gogh style -- high quality"), "Van_Gogh_style_high_quality")

    def test_build_van_gogh_100_prompts(self):
        items = build_van_gogh_100_prompts(count=100, base_seed=42)
        self.assertEqual(len(items), 100)

        prompts = [p for p, _ in items]
        seeds = [s for _, s in items]

        # 100 unique seeds
        self.assertEqual(len(set(seeds)), 100)

        # Templates cycle across 5 canonical templates
        for idx, (prompt, _) in enumerate(items):
            expected_template = CANONICAL_STYLE_TEMPLATES[idx % len(CANONICAL_STYLE_TEMPLATES)]
            self.assertEqual(prompt, expected_template.format(artist="Van Gogh"))

    def test_build_retain_100_prompts(self):
        style_csv = REPO_ROOT / "data" / "style_100_retain_eval.csv"
        if not style_csv.exists():
            style_csv = REPO_ROOT / "data" / "style.csv"

        items = build_retain_100_prompts(style_csv, count=100, base_seed=42)
        self.assertEqual(len(items), 100)

        artists = [a for a, _, _ in items]
        prompts = [p for _, p, _ in items]
        seeds = [s for _, _, s in items]

        # 100 distinct artists and seeds
        self.assertEqual(len(set(artists)), 100)
        self.assertEqual(len(set(seeds)), 100)
        self.assertNotIn("Van Gogh", artists)

        # Prompts pair each artist with cyclic templates
        for idx, (artist, prompt, _) in enumerate(items):
            expected_template = CANONICAL_STYLE_TEMPLATES[idx % len(CANONICAL_STYLE_TEMPLATES)]
            self.assertEqual(prompt, expected_template.format(artist=artist))


if __name__ == "__main__":
    unittest.main()

