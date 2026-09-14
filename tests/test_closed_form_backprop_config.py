import tempfile
import unittest
from pathlib import Path

from train_closed_form_backprop import (
    load_prompt_csv,
    parse_args,
    resolve_prompts,
    validate_args,
)


class ClosedFormBackpropConfigTests(unittest.TestCase):
    def _required(self):
        return [
            "--target_concepts",
            "Snoopy",
            "--anchor_concepts",
            "",
            "--retain_path",
            "data/instance.csv",
            "--heads",
            "concept",
        ]

    def test_cli_overrides_yaml_and_empty_anchor_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text(
                "target_concepts: [Snoopy]\n"
                "anchor_concepts: ['']\n"
                "retain_path: data/instance.csv\n"
                "heads: concept\n"
                "anchor_lr: 0.1\n"
                "max_anchor_norm: target\n",
                encoding="utf-8",
            )
            parser, args = parse_args(
                ["--config", str(config), "--anchor_lr", "0.02"]
            )
            targets, anchors = validate_args(parser, args)

        self.assertEqual(targets, ["Snoopy"])
        self.assertEqual(anchors, [""])
        self.assertEqual(args.anchor_lr, 0.02)
        self.assertEqual(args.max_anchor_norm, "target")

    def test_rejects_modes_outside_the_prototype(self):
        for option, value in (
            ("--params", "KV"),
            ("--anchor_mode", "shared_residual_mean"),
            ("--baseline", "other"),
            ("--aug_num", "10"),
            ("--max_anchor_norm", "60"),
        ):
            with self.subTest(option=option):
                parser, args = parse_args(self._required() + [option, value])
                with self.assertRaises(SystemExit):
                    validate_args(parser, args)

    def test_rejects_multiple_targets(self):
        values = self._required()
        values[1] = "Snoopy,Mickey"
        parser, args = parse_args(values)
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

    def test_prompt_fallback_and_csv_validation(self):
        self.assertEqual(resolve_prompts(None, ["Snoopy"]), ["Snoopy"])
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "valid.csv"
            valid.write_text("prompt\nSnoopy running\n", encoding="utf-8")
            self.assertEqual(load_prompt_csv(valid), ["Snoopy running"])

            invalid = Path(temp_dir) / "invalid.csv"
            invalid.write_text("text\nSnoopy running\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt.*column"):
                load_prompt_csv(invalid)

    def test_unknown_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text("mystery: true\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config)])
            config.write_text("max_residual_norm: 1.0\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config)])

    def test_yaml_types_are_validated_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for bad_field in (
                "anchor_lr: 'fast'",
                "anchor_steps: 2.5",
                "seed: 'zero'",
                "erase_style: 'false'",
                "use_k2: 'true'",
                "use_null_retain_loss: 'true'",
                "retain_projection_rank: true",
                "retain_projection_rank: -1",
                "retain_projection_rank: 2.5",
                "validation_seed: 0",
            ):
                with self.subTest(bad_field=bad_field):
                    config = Path(temp_dir) / "config.yaml"
                    config.write_text(bad_field + "\n", encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        parser, args = parse_args(
                            ["--config", str(config)] + self._required()
                        )
                        validate_args(parser, args)

    def test_retain_projection_rank_yaml_null_and_cli_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text("retain_projection_rank: null\n", encoding="utf-8")
            parser, args = parse_args(["--config", str(config)] + self._required())
            validate_args(parser, args)
            self.assertIsNone(args.retain_projection_rank)

            config.write_text("retain_projection_rank: 3\n", encoding="utf-8")
            parser, args = parse_args(["--config", str(config)] + self._required())
            validate_args(parser, args)
            self.assertEqual(args.retain_projection_rank, 3)

            parser, args = parse_args(
                ["--config", str(config)]
                + self._required()
                + ["--retain_projection_rank", "1"]
            )
            validate_args(parser, args)
            self.assertEqual(args.retain_projection_rank, 1)

    def test_retain_projection_rank_allows_zero(self):
        parser, args = parse_args(
            self._required() + ["--retain_projection_rank", "0"]
        )
        validate_args(parser, args)
        self.assertEqual(args.retain_projection_rank, 0)

    def test_use_k2_cli_and_yaml(self):
        # Default is True
        parser, args = parse_args(self._required())
        validate_args(parser, args)
        self.assertTrue(args.use_k2)

        # Explicit --no-use_k2
        parser, args = parse_args(self._required() + ["--no-use_k2"])
        validate_args(parser, args)
        self.assertFalse(args.use_k2)

        # Explicit --use_k2
        parser, args = parse_args(self._required() + ["--use_k2"])
        validate_args(parser, args)
        self.assertTrue(args.use_k2)

        # Via YAML config
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text("use_k2: false\n", encoding="utf-8")
            parser, args = parse_args(["--config", str(config)] + self._required())
            validate_args(parser, args)
            self.assertFalse(args.use_k2)

    def test_null_retain_loss_cli_yaml_and_precedence(self):
        parser, args = parse_args(self._required())
        validate_args(parser, args)
        self.assertFalse(args.use_null_retain_loss)

        parser, args = parse_args(
            self._required() + ["--use_null_retain_loss"]
        )
        validate_args(parser, args)
        self.assertTrue(args.use_null_retain_loss)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text(
                "use_null_retain_loss: true\n",
                encoding="utf-8",
            )
            parser, args = parse_args(
                ["--config", str(config)]
                + self._required()
                + ["--no-use_null_retain_loss"]
            )
            validate_args(parser, args)
            self.assertFalse(args.use_null_retain_loss)


if __name__ == "__main__":
    unittest.main()
