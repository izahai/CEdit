"""CPU-only unit tests for ReSAM configuration, CLI validation, and checkpoint round-tripping."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.edit_checkpoint import load_edit_checkpoint, save_speed_checkpoint
from train_resam import (
    normalize_target,
    parse_args,
    validate_args,
)


class ReSAMConfigTests(unittest.TestCase):
    def _create_dummy_candidate_file(self, temp_dir: Path) -> Path:
        cand_path = temp_dir / "candidates.csv"
        cand_path.write_text("id,concept\n1,dog\n2,cartoon dog\n3,beagle\n", encoding="utf-8")
        return cand_path

    def _create_dummy_retain_file(self, temp_dir: Path) -> Path:
        retain_path = temp_dir / "instance.csv"
        retain_path.write_text("id,concept\n1,cat\n2,mouse\n", encoding="utf-8")
        return retain_path

    def _required_args(self, temp_dir: Path) -> list[str]:
        cand_path = self._create_dummy_candidate_file(temp_dir)
        retain_path = self._create_dummy_retain_file(temp_dir)
        return [
            "--target_concepts",
            "Snoopy",
            "--candidate_concepts",
            str(cand_path),
            "--retain_path",
            str(retain_path),
            "--heads",
            "concept",
        ]

    def test_cli_overrides_yaml_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cand_path = self._create_dummy_candidate_file(temp_path)
            retain_path = self._create_dummy_retain_file(temp_path)

            config = temp_path / "config.yaml"
            config.write_text(
                f"target_concepts: [Snoopy]\n"
                f"candidate_concepts: {cand_path}\n"
                f"retain_path: {retain_path}\n"
                f"heads: concept\n"
                f"resam_k: 2\n"
                f"resam_temperature: 1.0\n"
                f"resam_lr: 0.05\n"
                f"use_null_retain_loss: true\n",
                encoding="utf-8",
            )

            # Override resam_k and resam_lr via CLI
            parser, args = parse_args([
                "--config", str(config),
                "--resam_k", "3",
                "--resam_lr", "0.01",
            ])
            target, candidates, file_hash = validate_args(parser, args)

            self.assertEqual(target, "Snoopy")
            self.assertEqual(len(candidates), 3)
            self.assertEqual(args.resam_k, 3)
            self.assertEqual(args.resam_lr, 0.01)
            self.assertTrue(args.use_null_retain_loss)
            self.assertEqual(args.retain_projection_rank, 1)
            self.assertTrue(args.resam_ste)

    def test_use_null_retain_loss_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Default is False
            parser, args = parse_args(self._required_args(temp_path))
            validate_args(parser, args)
            self.assertFalse(args.use_null_retain_loss)

            # Explicit flag sets to True
            parser, args = parse_args(
                self._required_args(temp_path) + ["--use_null_retain_loss"]
            )
            validate_args(parser, args)
            self.assertTrue(args.use_null_retain_loss)

            # Negation sets to False
            parser, args = parse_args(
                self._required_args(temp_path) + ["--no-use_null_retain_loss"]
            )
            validate_args(parser, args)
            self.assertFalse(args.use_null_retain_loss)

    def test_num_train_inference_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Default falls back to num_inference_steps (50)
            parser, args = parse_args(self._required_args(temp_path))
            validate_args(parser, args)
            self.assertEqual(args.num_train_inference_steps, 50)
            self.assertEqual(args.num_inference_steps, 50)

            # Explicit CLI --num_train_inference_steps
            parser, args = parse_args(
                self._required_args(temp_path) + ["--num_train_inference_steps", "25"]
            )
            validate_args(parser, args)
            self.assertEqual(args.num_train_inference_steps, 25)
            self.assertEqual(args.num_inference_steps, 50)

            # Fallback to --num_inference_steps when --num_train_inference_steps is not given
            parser, args = parse_args(
                self._required_args(temp_path) + ["--num_inference_steps", "30"]
            )
            validate_args(parser, args)
            self.assertEqual(args.num_train_inference_steps, 30)

            # YAML config with num_train_inference_steps and alias train_inference_steps
            cand_path = self._create_dummy_candidate_file(temp_path)
            retain_path = self._create_dummy_retain_file(temp_path)

            config1 = temp_path / "cfg1.yaml"
            config1.write_text(
                f"target_concepts: [Snoopy]\n"
                f"candidate_concepts: {cand_path}\n"
                f"retain_path: {retain_path}\n"
                f"num_train_inference_steps: 40\n",
                encoding="utf-8",
            )
            p1, a1 = parse_args(["--config", str(config1)])
            validate_args(p1, a1)
            self.assertEqual(a1.num_train_inference_steps, 40)

            config2 = temp_path / "cfg2.yaml"
            config2.write_text(
                f"target_concepts: [Snoopy]\n"
                f"candidate_concepts: {cand_path}\n"
                f"retain_path: {retain_path}\n"
                f"train_inference_steps: 35\n",
                encoding="utf-8",
            )
            p2, a2 = parse_args(["--config", str(config2)])
            validate_args(p2, a2)
            self.assertEqual(a2.num_train_inference_steps, 35)

            # Invalid values rejected
            for bad_val in ["0", "-5"]:
                p_bad, a_bad = parse_args(
                    self._required_args(temp_path) + ["--num_train_inference_steps", bad_val]
                )
                with self.assertRaises(SystemExit):
                    validate_args(p_bad, a_bad)

    def test_rejects_multiple_target_concepts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args_list = self._required_args(temp_path)

            # Multiple comma-separated
            args_list[1] = "Snoopy,Mickey"
            parser, args = parse_args(args_list)
            with self.assertRaises(SystemExit):
                validate_args(parser, args)

            # Multiple in YAML
            cand_path = self._create_dummy_candidate_file(temp_path)
            retain_path = self._create_dummy_retain_file(temp_path)
            config = temp_path / "multi.yaml"
            config.write_text(
                f"target_concepts:\n  - Snoopy\n  - Mickey\n"
                f"candidate_concepts: {cand_path}\n"
                f"retain_path: {retain_path}\n",
                encoding="utf-8",
            )
            parser2, args2 = parse_args(["--config", str(config)])
            with self.assertRaises(SystemExit):
                validate_args(parser2, args2)

    def test_rejects_k_outside_bank_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Bank has 3 candidates
            args_list = self._required_args(temp_path) + ["--resam_k", "5"]
            parser, args = parse_args(args_list)
            with self.assertRaises(SystemExit):
                validate_args(parser, args)

            args_list_zero = self._required_args(temp_path) + ["--resam_k", "0"]
            parser_zero, args_zero = parse_args(args_list_zero)
            with self.assertRaises(SystemExit):
                validate_args(parser_zero, args_zero)

    def test_rejects_unsupported_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for opt, val in [
                ("--params", "KV"),
                ("--baseline", "other"),
                ("--anchor_mode", "shared_residual_mean"),
                ("--aug_num", "10"),
                ("--retain_projection_rank", "-1"),
            ]:
                with self.subTest(option=opt):
                    parser, args = parse_args(self._required_args(temp_path) + [opt, val])
                    with self.assertRaises(SystemExit):
                        validate_args(parser, args)

    def test_rejects_unknown_yaml_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text("invalid_unsupported_key: 123\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(["--config", str(config)])

    def test_safetensors_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = Path(temp_dir) / "weight.safetensors"
            weights = {
                "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight": torch.randn(4, 4),
                "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight": torch.randn(4, 4),
            }
            save_speed_checkpoint(
                weights,
                ckpt_path,
                metadata={"target_concept": "Snoopy", "format": "speed-checkpoint-v1"},
            )
            reloaded, meta = load_edit_checkpoint(ckpt_path)
            for k, expected in weights.items():
                self.assertIn(k, reloaded)
                self.assertTrue(torch.equal(reloaded[k], expected))


if __name__ == "__main__":
    unittest.main()
