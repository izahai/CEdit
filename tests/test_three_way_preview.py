import csv
import sys
import tempfile
import unittest
from pathlib import Path
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.machinery
import types


def _install_optional_dependency_stubs():
    for mod in ("pandas", "tqdm", "kmeans_pytorch", "diffusers"):
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.__spec__ = importlib.machinery.ModuleSpec(mod, None)
            if mod == "tqdm":
                stub.tqdm = lambda values, **_: values
            elif mod == "kmeans_pytorch":
                stub.kmeans = None
            elif mod == "diffusers":
                stub.StableDiffusionPipeline = object
            sys.modules[mod] = stub


_install_optional_dependency_stubs()

from remote_scripts.eval_ReSAM.erase_van_gogh.preview_samples import (
    combine_three_way,
    load_retain_artists,
    parse_args,
)
from train_erase_null import build_argument_parser


class TestThreeWayPreview(unittest.TestCase):
    def test_combine_three_way_dimensions(self):
        w, h = 64, 64
        img1 = Image.new("RGB", (w, h), color=(255, 0, 0))
        img2 = Image.new("RGB", (w, h), color=(0, 255, 0))
        img3 = Image.new("RGB", (w, h), color=(0, 0, 255))

        combined = combine_three_way(
            img1, img2, img3,
            label_left="Original SD 1.4",
            label_mid="SPEED",
            label_right="Ours",
        )

        self.assertEqual(combined.size, (w * 3, h + 40))
        self.assertEqual(combined.mode, "RGB")

    def test_speed_config_yaml_and_parser(self):
        config_path = REPO_ROOT / "remote_scripts/eval_ReSAM/erase_van_gogh/speed_config.yaml"
        self.assertTrue(config_path.is_file(), f"speed_config.yaml not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["target_concepts"], "Van Gogh")
        self.assertEqual(cfg["anchor_concepts"], "art")
        self.assertEqual(cfg["retain_projection_rank"], 100)
        self.assertEqual(cfg["baseline"], "SPEED")
        self.assertEqual(cfg["params"], "V")

        # Verify train_erase_null argument parser supports retain_projection_rank
        parser = build_argument_parser()
        args = parser.parse_args(["--config", str(config_path), "--retain_projection_rank", "100"])
        self.assertEqual(args.retain_projection_rank, 100)

    def test_load_retain_artists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "style.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "concept"])
                writer.writerow(["1", "Claude Monet"])
                writer.writerow(["2", "Van Gogh"])
                writer.writerow(["3", "Vincent van Gogh"])
                writer.writerow(["4", "Paul Gauguin"])
                writer.writerow(["5", "Picasso"])

            artists = load_retain_artists(csv_path, target="Van Gogh", count=2, seed=42)
            self.assertEqual(len(artists), 2)
            for a in artists:
                self.assertNotIn("van gogh", a.lower())


if __name__ == "__main__":
    unittest.main()
