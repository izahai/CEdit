import tempfile
import unittest
from pathlib import Path

from learn_anchor import parse_args


class LearnAnchorConfigTests(unittest.TestCase):
    def test_requires_targets_and_save_root(self):
        with self.assertRaises(SystemExit):
            parse_args([])
        with self.assertRaises(SystemExit):
            parse_args(["--target_concepts", "Snoopy"])

    def test_defaults_match_first_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, args, targets, config = parse_args(
                [
                    "--target_concepts",
                    "Snoopy",
                    "--save_root",
                    str(Path(temp_dir) / "artifact"),
                ]
            )

        self.assertEqual(targets, ["Snoopy"])
        self.assertEqual(config.num_prefix_tokens, 4)
        self.assertEqual(config.num_reference_images, 4)
        self.assertEqual(config.num_validation_images, 1)
        self.assertEqual(config.iterations, 1000)
        self.assertEqual(config.validation_samples, 16)
        self.assertEqual(config.validation_interval, 50)
        self.assertEqual(args.dtype, "float16")

    def test_yaml_lists_and_cli_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "anchor.yaml"
            save_root = Path(temp_dir) / "artifact"
            config_path.write_text(
                "target_concepts: [Snoopy, Mickey]\n"
                f"save_root: {save_root}\n"
                "iterations: 20\n"
                "dtype: float32\n"
                "device: cpu\n",
                encoding="utf-8",
            )

            _, _, targets, config = parse_args(
                ["--config", str(config_path), "--iterations", "30"]
            )

        self.assertEqual(targets, ["Snoopy", "Mickey"])
        self.assertEqual(config.iterations, 30)
        self.assertEqual(config.dtype, "float32")

    def test_rejects_unknown_yaml_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "anchor.yaml"
            config_path.write_text("unknown: true\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config_path)])

    def test_rejects_cpu_float16_and_nudity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_root = str(Path(temp_dir) / "artifact")
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target_concepts",
                        "Snoopy",
                        "--save_root",
                        save_root,
                        "--device",
                        "cpu",
                    ]
                )
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target_concepts",
                        "nudity",
                        "--save_root",
                        save_root,
                    ]
                )

    def test_rejects_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_root = Path(temp_dir) / "artifact"
            save_root.mkdir()
            (save_root / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--target_concepts",
                        "Snoopy",
                        "--save_root",
                        str(save_root),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
