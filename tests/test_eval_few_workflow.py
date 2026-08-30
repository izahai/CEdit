import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from remote_scripts.eval_few.eval_few_ins_1.evaluate_clip_fid import (
    build_comparison_rows,
    few_prompt_records,
    summarize_detailed_rows,
    validate_image_directory,
)
from remote_scripts.eval_few.eval_few_ins_1.workflow_config import (
    EXPECTED_METHODS,
    expected_image_counts,
    load_config,
    task_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = (
    REPO_ROOT / "remote_scripts" / "eval_few" / "eval_few_ins_1"
)
TGPRS_SUBSPACE_ANCHORS = [
    "Cheerful Cartoon Mouse",
    "Playful Cartoon Dog",
    "Goofy Sea Creature",
    "Cute Electric Fantasy Creature",
    "Minimalist Cute Cat",
    "Friendly Woodland Mascot",
    "Retro Animal Cartoon",
    "Kawaii Fantasy Mascot",
    "Funny Everyday Hero",
    "Universal Cute Mascot",
]
TGPRS_SUBSPACE_ANCHORS_JSON = json.dumps(
    TGPRS_SUBSPACE_ANCHORS,
    ensure_ascii=False,
)


class EvalFewWorkflowTests(unittest.TestCase):
    def test_full_workflow_reproduces_eval_few_matrix(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        tasks = task_specs(config)

        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            [task["id"] for task in tasks],
            [
                "snoopy",
                "snoopy_mickey",
                "snoopy_mickey_spongebob",
            ],
        )
        self.assertTrue(all(task["erase_type"] == "instance" for task in tasks))
        self.assertEqual(
            [task["applied_residual_rank"] for task in tasks],
            [10, 11, 12],
        )
        self.assertTrue(all(
            task["subspace_anchor_concepts"] == TGPRS_SUBSPACE_ANCHORS
            for task in tasks
        ))
        self.assertEqual(expected_image_counts(config)["total"], 35000)

    def test_task_subspace_anchor_override_updates_rank_and_residual_count(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        target_only = copy.deepcopy(config)
        target_only["tasks"][1]["subspace_anchor_concepts"] = []
        custom_anchor = copy.deepcopy(config)
        custom_anchor["tasks"][0]["subspace_anchor_concepts"] = ["extra anchor"]

        target_only_task = task_specs(target_only)[1]
        custom_anchor_task = task_specs(custom_anchor)[0]

        self.assertEqual(target_only_task["subspace_anchor_concepts"], [])
        self.assertEqual(target_only_task["subspace_anchor_count"], 0)
        self.assertEqual(target_only_task["target_global_residual_count"], 2)
        self.assertEqual(target_only_task["applied_residual_rank"], 1)
        self.assertEqual(
            custom_anchor_task["subspace_anchor_concepts"], ["extra anchor"]
        )
        self.assertEqual(custom_anchor_task["target_global_residual_count"], 1)
        self.assertEqual(custom_anchor_task["applied_residual_rank"], 1)

    def test_one_target_without_subspace_anchors_is_invalid(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        config["tasks"][0]["subspace_anchor_concepts"] = []

        with self.assertRaisesRegex(ValueError, "no TGPRS residual vectors"):
            task_specs(config)

    def test_tasks_inherit_custom_training_config_subspace_anchors(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        config.pop("_resolved_tgprs_train_config", None)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workflow_path = temp_path / "workflow.yaml"
            workflow_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            (temp_path / "train_config_target_global_pairwise_residual_subspace.yaml").write_text(
                "residual_rank: 30\nsubspace_anchor_concepts: [extra anchor]\n",
                encoding="utf-8",
            )

            loaded = load_config(workflow_path)

        tasks = task_specs(loaded)
        self.assertTrue(all(
            task["subspace_anchor_concepts"] == ["extra anchor"] for task in tasks
        ))
        self.assertTrue(all(task["applied_residual_rank"] == 1 for task in tasks))

    def test_task_anchor_command_preserves_empty_and_empty_list_overrides(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        config.pop("_resolved_tgprs_train_config", None)
        config["tasks"][0]["subspace_anchor_concepts"] = ["", "extra"]
        config["tasks"][1]["subspace_anchor_concepts"] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "workflow.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            command = WORKFLOW_DIR / "workflow_config.py"
            custom = subprocess.run(
                [
                    sys.executable,
                    str(command),
                    "--config",
                    str(config_path),
                    "--task-id",
                    "snoopy",
                    "task-anchors",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            empty = subprocess.run(
                [
                    sys.executable,
                    str(command),
                    "--config",
                    str(config_path),
                    "--task-id",
                    "snoopy_mickey",
                    "task-anchors",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(custom.stdout.splitlines(), ["1", "2", "", "extra"])
        self.assertEqual(empty.stdout.splitlines(), ["1", "0"])

    def test_smoke_workflow_has_exact_reduced_size(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        self.assertEqual(expected_image_counts(config)["total"], 90)
        self.assertEqual(len(task_specs(config)), 1)
        self.assertEqual(config["experiment"]["fid_feature_layer"], 64)

    def test_full_workflow_uses_standard_fid_feature_layer(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        self.assertEqual(config["experiment"]["fid_feature_layer"], 2048)

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
                "subspace_anchor_concepts": (
                    TGPRS_SUBSPACE_ANCHORS_JSON
                    if "target_global" in model else ""
                ),
                "subspace_anchor_count": (
                    len(TGPRS_SUBSPACE_ANCHORS)
                    if "target_global" in model else ""
                ),
                "configured_residual_rank": 30 if "target_global" in model else "",
                "applied_residual_rank": 10 if "target_global" in model else "",
                "target_global_residual_count": (
                    len(TGPRS_SUBSPACE_ANCHORS)
                    if "target_global" in model else ""
                ),
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
        self.assertEqual(
            comparison["tgprs_subspace_anchor_concepts"],
            TGPRS_SUBSPACE_ANCHORS_JSON,
        )
        self.assertEqual(
            comparison["tgprs_subspace_anchor_count"],
            len(TGPRS_SUBSPACE_ANCHORS),
        )

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
        self.assertEqual(
            tgprs["subspace_anchor_concepts"], TGPRS_SUBSPACE_ANCHORS
        )

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

        train_script = (WORKFLOW_DIR / "03_train.sh").read_text(encoding="utf-8")
        self.assertIn("task-anchors", train_script)
        self.assertIn("--subspace_anchor_concepts", train_script)

    def test_sample2_supports_configurable_max_samples(self):
        sample2 = (REPO_ROOT / "sample2.py").read_text(encoding="utf-8")
        self.assertIn("--max_samples", sample2)
        self.assertIn("max_num=args.max_samples or 1000", sample2)


if __name__ == "__main__":
    unittest.main()
