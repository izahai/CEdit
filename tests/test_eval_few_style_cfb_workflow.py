import unittest
from pathlib import Path

import yaml

from remote_scripts.eval_few.eval_few_style_cfb.evaluate_clip_fid import (
    build_comparison_rows,
    summarize_detailed_rows,
)
from remote_scripts.eval_few.eval_few_style_cfb.workflow_config import (
    EXPECTED_EDITED_METHODS,
    EXPECTED_METHODS,
    expected_image_counts,
    load_config,
    task_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = (
    REPO_ROOT / "remote_scripts" / "eval_few" / "eval_few_style_cfb"
)


class EvalFewStyleCfbWorkflowTests(unittest.TestCase):
    def test_full_workflow_has_expected_methods_tasks_and_counts(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        tasks = task_specs(config)

        self.assertEqual(EXPECTED_METHODS, ["original", "legacy", "cfb"])
        self.assertEqual(EXPECTED_EDITED_METHODS, ["legacy", "cfb"])
        self.assertEqual(config["experiment"]["methods"], EXPECTED_METHODS)
        self.assertEqual(
            config["experiment"]["edited_methods"], EXPECTED_EDITED_METHODS
        )
        self.assertEqual(
            [task["id"] for task in tasks], ["van_gogh", "picasso", "monet"]
        )
        self.assertTrue(all(task["erase_type"] == "style" for task in tasks))
        self.assertTrue(all(task["anchor_concept"] == "art" for task in tasks))
        self.assertTrue(all(task["target_count"] == 1 for task in tasks))
        self.assertTrue(all(len(task["prompt_templates"]) == 30 for task in tasks))
        self.assertEqual(config["experiment"]["num_samples_per_prompt"], 1)
        self.assertEqual(config["experiment"]["mscoco_num_prompts"], 100)
        self.assertEqual(expected_image_counts(config), {
            "original_few": 150,
            "edited_few": 900,
            "original_coco": 100,
            "edited_coco": 600,
            "total": 1750,
        })

    def test_smoke_workflow_has_exact_reduced_size(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        tasks = task_specs(config)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], "van_gogh")
        self.assertEqual(tasks[0]["contents"], ["Van Gogh", "Caravaggio"])
        self.assertEqual(config["experiment"]["fid_feature_layer"], 64)
        self.assertEqual(expected_image_counts(config), {
            "original_few": 20,
            "edited_few": 40,
            "original_coco": 10,
            "edited_coco": 20,
            "total": 90,
        })

    def test_training_configs_lock_legacy_and_cfb_settings(self):
        with (WORKFLOW_DIR / "train_config_legacy.yaml").open() as config_file:
            legacy = yaml.safe_load(config_file)
        with (WORKFLOW_DIR / "train_config_cfb.yaml").open() as config_file:
            cfb = yaml.safe_load(config_file)

        self.assertEqual(legacy["anchor_mode"], "legacy")
        self.assertEqual(legacy["aug_num"], 10)
        self.assertEqual(legacy["threshold"], 0.1)
        self.assertEqual(legacy["retain_scale"], 1.0)
        self.assertEqual(cfb["anchor_mode"], "legacy")
        self.assertTrue(cfb["erase_style"])
        self.assertEqual(cfb["retain_projection_rank"], 150)
        self.assertEqual(cfb["threshold"], 0.1)
        self.assertEqual(cfb["retain_scale"], 100.0)
        self.assertEqual(cfb["anchor_steps"], 200)
        self.assertEqual(cfb["anchor_lr"], 0.01)
        self.assertFalse(cfb["use_k2"])
        self.assertEqual(cfb["validation_samples"], 1)
        self.assertEqual(cfb["validation_interval"], 9999)

    def test_scripts_route_trainers_and_checkpoint_formats(self):
        common = (WORKFLOW_DIR / "common.sh").read_text(encoding="utf-8")
        train = (WORKFLOW_DIR / "03_train.sh").read_text(encoding="utf-8")
        run_all = (WORKFLOW_DIR / "run_all.sh").read_text(encoding="utf-8")

        self.assertIn('if [[ "$1" == "cfb" ]]', common)
        self.assertIn('extension="safetensors"', common)
        self.assertIn('entrypoint="train_erase_null.py"', train)
        self.assertIn('entrypoint="train_closed_form_backprop.py"', train)
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

    def test_comparison_improvements_positive_when_cfb_is_better(self):
        task = task_specs(load_config(WORKFLOW_DIR / "workflow.yaml"))[0]
        rows = []
        values = {
            "original": (29.0, 0.0, 27.0, 0.0),
            "legacy": (22.0, 18.0, 25.0, 12.0),
            "cfb": (20.0, 14.0, 26.0, 9.0),
        }
        empty_metadata = {
            "anchor_mode": "",
            "anchor_concepts": "",
            "params": "",
            "aug_num": "",
            "erase_style": "",
            "retain_projection_rank": "",
            "threshold": "",
            "retain_scale": "",
            "residual_scale": "",
            "use_k2": "",
            "anchor_steps": "",
            "anchor_lr": "",
            "anchor_batch_size": "",
            "validation_samples": "",
            "validation_interval": "",
        }
        for model, (target_clip, non_target_fid, coco_clip, coco_fid) in values.items():
            metadata = dict(empty_metadata)
            if model != "original":
                metadata.update({
                    "anchor_mode": "legacy",
                    "anchor_concepts": "art",
                    "params": "V",
                    "aug_num": 10 if model == "legacy" else 0,
                })
            if model == "cfb":
                metadata.update({"retain_projection_rank": 150, "anchor_steps": 200})
            base = {
                "task_id": task["id"],
                "erase_type": "style",
                "target_concepts": "Van Gogh",
                "target_count": 1,
                "model": model,
                **metadata,
                "n_images": 30,
            }
            rows.append({
                **base,
                "content": "Van Gogh",
                "content_role": "target",
                "clip_score": target_clip,
                "fid_vs_original": 0.0,
            })
            for content in ("Picasso", "Monet", "Paul Gauguin", "Caravaggio"):
                rows.append({
                    **base,
                    "content": content,
                    "content_role": "non_target",
                    "clip_score": 25.0,
                    "fid_vs_original": non_target_fid,
                })
            rows.append({
                **base,
                "content": "coco",
                "content_role": "coco",
                "clip_score": coco_clip,
                "fid_vs_original": coco_fid,
                "n_images": 100,
            })

        summaries = summarize_detailed_rows(rows, [task], EXPECTED_METHODS)
        comparison = build_comparison_rows(summaries, [task])[0]
        self.assertEqual(comparison["target_clip_score_improvement"], 2.0)
        self.assertEqual(comparison["non_target_fid_improvement"], 4.0)
        self.assertEqual(comparison["mscoco_clip_score_improvement"], 1.0)
        self.assertEqual(comparison["mscoco_fid_improvement"], 3.0)


if __name__ == "__main__":
    unittest.main()
