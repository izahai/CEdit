import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path


import importlib.machinery


def _install_optional_dependency_stubs():
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
            sys.modules[name] = stub


_install_optional_dependency_stubs()

from train_erase_null import (
    DEFAULT_SUBSPACE_ANCHOR_CONCEPTS,
    is_zero_anchor_concept,
    load_subspace_concepts,
    normalize_concepts,
    normalize_subspace_anchor_concepts,
    parse_args,
    target_embedding_prompts,
)


class TrainConfigTests(unittest.TestCase):
    def test_erase_style_defaults_to_false_and_cli_enables_it(self):
        _, default_args = parse_args([])
        _, enabled_args = parse_args(["--erase_style"])

        self.assertFalse(default_args.erase_style)
        self.assertTrue(enabled_args.erase_style)

    def test_erase_style_loads_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text("erase_style: true\n", encoding="utf-8")

            _, args = parse_args(["--config", str(config_path)])

        self.assertTrue(args.erase_style)

    def test_target_embedding_prompts_only_add_style_when_enabled(self):
        concepts = ["Van Gogh", "Picasso"]

        self.assertEqual(
            target_embedding_prompts(concepts),
            ["Van Gogh", "Picasso"],
        )
        self.assertEqual(
            target_embedding_prompts(concepts, erase_style=True),
            ["Van Gogh style", "Picasso style"],
        )
        self.assertEqual(concepts, ["Van Gogh", "Picasso"])

    def test_subspace_anchor_override_defaults_and_cli_values(self):
        _, default_args = parse_args([])
        _, custom_args = parse_args([
            "--subspace_anchor_concepts",
            "",
            "art",
        ])
        _, empty_args = parse_args(["--subspace_anchor_concepts"])

        self.assertIsNone(default_args.subspace_anchor_concepts)
        self.assertEqual(
            normalize_subspace_anchor_concepts(
                default_args.subspace_anchor_concepts
            ),
            list(DEFAULT_SUBSPACE_ANCHOR_CONCEPTS),
        )
        self.assertEqual(custom_args.subspace_anchor_concepts, ["", "art"])
        self.assertEqual(empty_args.subspace_anchor_concepts, [])

    def test_subspace_anchor_override_loads_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text(
                "subspace_anchor_concepts: ['', art, painting]\n",
                encoding="utf-8",
            )

            _, args = parse_args(["--config", str(config_path)])

        self.assertEqual(
            normalize_subspace_anchor_concepts(args.subspace_anchor_concepts),
            ["", "art", "painting"],
        )

    def test_mean_norm_target_global_mode_does_not_require_subspace_csv(self):
        _, args = parse_args(
            [
                "--anchor_mode",
                "mean_norm_target_global_pairwise_residual_subspace",
            ]
        )

        self.assertEqual(
            args.anchor_mode,
            "mean_norm_target_global_pairwise_residual_subspace",
        )
        self.assertIsNone(args.subspace_concepts_path)

    def test_target_global_mode_does_not_require_subspace_csv(self):
        _, args = parse_args(
            ["--anchor_mode", "target_global_pairwise_residual_subspace"]
        )

        self.assertEqual(
            args.anchor_mode,
            "target_global_pairwise_residual_subspace",
        )
        self.assertIsNone(args.subspace_concepts_path)

    def test_target_projection_direction_defaults_negative_and_accepts_positive(self):
        _, default_args = parse_args([])
        _, positive_args = parse_args(
            ["--target_projection_direction", "positive"]
        )

        self.assertEqual(default_args.target_projection_direction, "negative")
        self.assertEqual(positive_args.target_projection_direction, "positive")
        with self.assertRaises(SystemExit):
            parse_args(["--target_projection_direction", "sideways"])

    def test_accepts_norm_matched_truncated_svd_mode(self):
        _, args = parse_args(
            [
                "--anchor_mode",
                "norm_matched_truncated_svd_residual",
                "--residual_rank",
                "30",
            ]
        )

        self.assertEqual(
            args.anchor_mode,
            "norm_matched_truncated_svd_residual",
        )
        self.assertEqual(args.residual_rank, 30)

    def test_retain_aware_target_global_mode_does_not_require_subspace_csv(self):
        _, args = parse_args(
            [
                "--anchor_mode",
                "retain_aware_target_global_pairwise_residual_subspace",
            ]
        )

        self.assertEqual(
            args.anchor_mode,
            "retain_aware_target_global_pairwise_residual_subspace",
        )
        self.assertIsNone(args.subspace_concepts_path)

    def test_global_mode_requires_explicit_subspace_csv(self):
        with self.assertRaises(SystemExit):
            parse_args(
                ["--anchor_mode", "global_pairwise_residual_subspace"]
            )

        _, args = parse_args(
            [
                "--anchor_mode",
                "global_pairwise_residual_subspace",
                "--subspace_concepts_path",
                "data/10_celebrity.csv",
            ]
        )
        self.assertEqual(
            args.subspace_concepts_path,
            "data/10_celebrity.csv",
        )

    def test_loads_all_unique_subspace_concepts_regardless_of_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "concepts.csv"
            csv_path.write_text(
                "type,concept\n"
                "erase,Ada\n"
                "retain,Grace\n"
                "retain,Ada\n",
                encoding="utf-8",
            )

            concepts = load_subspace_concepts(csv_path)

        self.assertEqual(concepts, ["Ada", "Grace"])

    def test_rejects_invalid_subspace_concept_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_column = Path(temp_dir) / "missing.csv"
            missing_column.write_text("type,name\nerase,Ada\n", encoding="utf-8")
            blank_concept = Path(temp_dir) / "blank.csv"
            blank_concept.write_text("type,concept\nerase,\n", encoding="utf-8")
            empty = Path(temp_dir) / "empty.csv"
            empty.write_text("type,concept\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "concept.*column"):
                load_subspace_concepts(missing_column)
            with self.assertRaisesRegex(ValueError, "blank concept"):
                load_subspace_concepts(blank_concept)
            with self.assertRaisesRegex(ValueError, "no concepts"):
                load_subspace_concepts(empty)
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

    def test_is_zero_anchor_concept(self):
        self.assertTrue(is_zero_anchor_concept("<zero>"))
        self.assertTrue(is_zero_anchor_concept("<ZERO>"))
        self.assertTrue(is_zero_anchor_concept(" <zero> "))
        self.assertFalse(is_zero_anchor_concept("zero"))
        self.assertFalse(is_zero_anchor_concept("<zeros>"))
        self.assertFalse(is_zero_anchor_concept(""))
        self.assertFalse(is_zero_anchor_concept("person"))
        self.assertFalse(is_zero_anchor_concept(None))
        self.assertFalse(is_zero_anchor_concept(123))

    def test_zero_anchor_parses_from_cli_and_yaml(self):
        _, args = parse_args(["--anchor_concepts", "<zero>"])
        self.assertEqual(args.anchor_concepts, "<zero>")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train.yaml"
            config_path.write_text("anchor_concepts: '<zero>'\n", encoding="utf-8")
            _, yaml_args = parse_args(["--config", str(config_path)])
        self.assertEqual(yaml_args.anchor_concepts, "<zero>")


if __name__ == "__main__":
    unittest.main()
