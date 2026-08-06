import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path


def _install_optional_dependency_stubs():
    pandas = types.ModuleType("pandas")
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda values, **_: values
    kmeans_module = types.ModuleType("kmeans_pytorch")
    kmeans_module.kmeans = None
    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusionPipeline = object
    sys.modules.setdefault("pandas", pandas)
    sys.modules.setdefault("tqdm", tqdm_module)
    sys.modules.setdefault("kmeans_pytorch", kmeans_module)
    sys.modules.setdefault("diffusers", diffusers)


_install_optional_dependency_stubs()

from train_erase_null import normalize_concepts, parse_args


class TrainConfigTests(unittest.TestCase):
    def test_loads_yaml_lists_and_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text(
                "target_concepts: [Snoopy, Mickey]\n"
                "anchor_concepts: ['']\n"
                "threshold: 0.25\n",
                encoding="utf-8",
            )

            _, args = parse_args(["--config", str(config_path)])

        self.assertEqual(args.target_concepts, ["Snoopy", "Mickey"])
        self.assertEqual(args.anchor_concepts, [""])
        self.assertEqual(args.threshold, 0.25)

    def test_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text(
                "target_concepts: [Snoopy]\n"
                "anchor_concepts: ['']\n"
                "threshold: 0.25\n",
                encoding="utf-8",
            )

            _, args = parse_args(
                ["--config", str(config_path), "--threshold", "0.5"]
            )

        self.assertEqual(args.threshold, 0.5)

    def test_rejects_unknown_yaml_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text("unknown_option: true\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config_path)])

    def test_expands_environment_variables_in_yaml_strings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text(
                "target_concepts: [Snoopy]\n"
                "anchor_concepts: ['']\n"
                "save_path: ${TRAIN_OUTPUT_DIR}\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"TRAIN_OUTPUT_DIR": "/tmp/checkpoints"}):
                _, args = parse_args(["--config", str(config_path)])

        self.assertEqual(args.save_path, "/tmp/checkpoints")

    def test_normalizes_cli_and_yaml_concepts(self):
        self.assertEqual(
            normalize_concepts("Snoopy, Mickey", "target_concepts"),
            ["Snoopy", "Mickey"],
        )
        self.assertEqual(
            normalize_concepts(["", "person"], "anchor_concepts", allow_empty=True),
            ["", "person"],
        )


if __name__ == "__main__":
    unittest.main()
