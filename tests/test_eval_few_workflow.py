import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from remote_scripts.eval_few.evaluate_clip_fid import (
    build_comparison_rows,
    few_prompt_records,
    summarize_detailed_rows,
    validate_image_directory,
)
from remote_scripts.eval_few.workflow_config import (
    EXPECTED_METHODS,
    expected_image_counts,
    load_config,
    task_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "remote_scripts" / "eval_few"


class EvalFewWorkflowTests(unittest.TestCase):
    def test_full_workflow_reproduces_eval_few_matrix(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        tasks = task_specs(config)

        self.assertEqual(len(tasks), 6)
        self.assertEqual(
            [task["id"] for task in tasks],
            [
                "snoopy",
                "snoopy_mickey",
                "snoopy_mickey_spongebob",
                "van_gogh",
                "picasso",
                "monet",
            ],
        )
        self.assertEqual(
            [task["applied_residual_rank"] for task in tasks],
            [2, 3, 4, 2, 2, 2],
        )
        self.assertEqual(expected_image_counts(config)["total"], 51500)

    def test_smoke_workflow_has_exact_reduced_size(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        self.assertEqual(expected_image_counts(config)["total"], 170)
        self.assertEqual(len(task_specs(config)), 2)

    def test_prompt_records_match_sample_filename_and_legacy_clip_text(self):
        records = few_prompt_records(["a photo of a {}."], "Snoopy", 2)
        self.assertEqual(records, [
            {
                "filename": "a photo of a Snoopy_0.png",
                "prompt": "a photo of a Snoopy",
            },
            {
                "filename": "a photo of a Snoopy_1.png",
                "prompt": "a photo of a Snoopy",
            },
        ])

    def test_image_validation_rejects_missing_and_unexpected_files(self):
        records = [{"filename": "expected.png", "prompt": "prompt"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            (image_dir / "unexpected.png").touch()
            with self.assertRaisesRegex(ValueError, "missing 1"):
                validate_image_directory(image_dir, records)

            (image_dir / "unexpected.png").unlink()
            (image_dir / "expected.png").touch()
            digest = validate_image_directory(image_dir, records)
            self.assertEqual(len(digest), 64)

    def test_summary_and_comparison_delta_directions(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        task = task_specs(config)[0]
        rows = []
        values = {
            "original": (30.0, 0.0, 28.0, 0.0),
            "legacy": (20.0, 12.0, 25.0, 10.0),
            "target_global_pairwise_residual_subspace": (
                18.0,
                8.0,
                26.0,
                7.0,
            ),
        }
        for model in EXPECTED_METHODS:
            target_clip, non_target_fid, coco_clip, coco_fid = values[model]
            metadata = {
                "anchor_mode": "" if model == "original" else model,
                "anchor_concepts": "",
                "configured_residual_rank": 30 if "target_global" in model else "",
                "applied_residual_rank": 2 if "target_global" in model else "",
                "target_global_residual_count": 2 if "target_global" in model else "",
                "params": "" if model == "original" else "V",
                "aug_num": "" if model == "original" else (10 if model == "legacy" else 0),
                "threshold": "" if model == "original" else 0.1,
                "retain_scale": "" if model == "original" else 1.0,
            }
            base = {
                "task_id": task["id"],
                "erase_type": task["erase_type"],
                "target_concepts": "Snoopy",
                "target_count": 1,
                "model": model,
                "n_images": 10,
                **metadata,
            }
            rows.extend([
                {
                    **base,
                    "content": "Snoopy",
                    "content_role": "target",
                    "clip_score": target_clip,
                    "fid_vs_original": 0.0,
                },
                {
                    **base,
                    "content": "Pikachu",
                    "content_role": "non_target",
                    "clip_score": 24.0,
                    "fid_vs_original": non_target_fid,
                },
                {
                    **base,
                    "content": "coco",
                    "content_role": "coco",
                    "clip_score": coco_clip,
                    "fid_vs_original": coco_fid,
                },
            ])

        summaries = summarize_detailed_rows(rows, [task], EXPECTED_METHODS)
        comparison = build_comparison_rows(summaries, [task])[0]
        self.assertEqual(comparison["target_clip_score_improvement"], 2.0)
        self.assertEqual(comparison["non_target_fid_improvement"], 4.0)
        self.assertEqual(comparison["mscoco_clip_score_improvement"], 1.0)
        self.assertEqual(comparison["mscoco_fid_improvement"], 3.0)

    def test_train_configs_keep_documented_augmentation_confound(self):
        with (WORKFLOW_DIR / "train_config_legacy.yaml").open() as config_file:
            legacy = yaml.safe_load(config_file)
        with (
            WORKFLOW_DIR
            / "train_config_target_global_pairwise_residual_subspace.yaml"
        ).open() as config_file:
            tgprs = yaml.safe_load(config_file)

        self.assertEqual(legacy["anchor_mode"], "legacy")
        self.assertEqual(legacy["aug_num"], 10)
        self.assertEqual(tgprs["anchor_mode"], "target_global_pairwise_residual_subspace")
        self.assertEqual(tgprs["aug_num"], 0)
        self.assertEqual(tgprs["residual_rank"], 30)

    def test_run_all_orders_every_stage(self):
        run_all = (WORKFLOW_DIR / "run_all.sh").read_text(encoding="utf-8")
        positions = [
            run_all.index(stage)
            for stage in (
                "00_validate.sh",
                "01_setup_environment.sh",
                "02_generate_original.sh",
                "03_train.sh",
                "04_generate_edits.sh",
                "05_generate_mscoco.sh",
                "06_evaluate.sh",
            )
        ]
        self.assertEqual(positions, sorted(positions))

    def test_sample2_supports_configurable_max_samples(self):
        sample2 = (REPO_ROOT / "sample2.py").read_text(encoding="utf-8")
        self.assertIn("--max_samples", sample2)
        self.assertIn("max_num=args.max_samples or 1000", sample2)


if __name__ == "__main__":
    unittest.main()
