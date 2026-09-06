import unittest
from pathlib import Path

import yaml

from remote_scripts.eval_few.eval_few_style_1.evaluate_clip_fid import (
    build_comparison_rows,
    summarize_detailed_rows,
)
from remote_scripts.eval_few.eval_few_style_1.workflow_config import (
    EXPECTED_METHODS,
    expected_image_counts,
    load_config,
    task_specs,
)
from src.template import painting_templates


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = (
    REPO_ROOT / "remote_scripts" / "eval_few" / "eval_few_style_1"
)


class EvalFewStyleWorkflowTests(unittest.TestCase):
    def test_full_workflow_reproduces_style_matrix(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        tasks = task_specs(config)

        self.assertEqual(
            [task["id"] for task in tasks],
            ["van_gogh", "picasso", "monet"],
        )
        self.assertTrue(all(task["erase_type"] == "style" for task in tasks))
        self.assertTrue(all(task["anchor_concept"] == "art" for task in tasks))
        self.assertEqual(
            [task["target_concepts"] for task in tasks],
            [["Van Gogh"], ["Picasso"], ["Monet"]],
        )
        self.assertTrue(all(
            task["contents"] == [
                "Van Gogh",
                "Picasso",
                "Monet",
                "Paul Gauguin",
                "Caravaggio",
            ]
            for task in tasks
        ))
        self.assertTrue(all(task["applied_residual_rank"] == 30 for task in tasks))
        self.assertEqual(expected_image_counts(config), {
            "original_few": 150,
            "edited_few": 900,
            "original_coco": 100,
            "edited_coco": 600,
            "total": 1750,
        })
        self.assertEqual(
            len(EXPECTED_METHODS) * sum(len(task["contents"]) + 1 for task in tasks),
            54,
        )

    def test_tgprs_anchors_are_all_artist_neutral_style_prompts(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        expected_anchors = [template.format("art") for template in painting_templates]
        evaluated_artists = {
            "van gogh",
            "picasso",
            "monet",
            "paul gauguin",
            "caravaggio",
        }

        self.assertEqual(len(expected_anchors), 30)
        for task in task_specs(config):
            self.assertEqual(task["subspace_anchor_concepts"], expected_anchors)
            self.assertEqual(task["subspace_anchor_count"], 30)
            normalized = " ".join(expected_anchors).casefold()
            self.assertTrue(all(artist not in normalized for artist in evaluated_artists))

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

    def test_training_configs_match_documented_comparison(self):
        with (WORKFLOW_DIR / "train_config_legacy.yaml").open() as config_file:
            legacy = yaml.safe_load(config_file)
        with (
            WORKFLOW_DIR
            / "train_config_target_global_pairwise_residual_subspace.yaml"
        ).open() as config_file:
            tgprs = yaml.safe_load(config_file)

        self.assertEqual(legacy["anchor_mode"], "legacy")
        self.assertEqual(legacy["aug_num"], 10)
        self.assertEqual(legacy["threshold"], 0.1)
        self.assertEqual(legacy["retain_scale"], 1.0)
        self.assertEqual(
            tgprs["anchor_mode"],
            "target_global_pairwise_residual_subspace",
        )
        self.assertEqual(tgprs["aug_num"], 0)
        self.assertEqual(tgprs["threshold"], 0.3)
        self.assertEqual(tgprs["retain_scale"], 0.5)
        self.assertEqual(tgprs["residual_rank"], 30)
        self.assertEqual(tgprs["residual_scale"], 1.0)

    def test_summary_classifies_single_target_and_four_non_targets(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        task = task_specs(config)[0]
        rows = []
        values = {
            "original": (29.0, 0.0, 27.0, 0.0),
            "legacy": (22.0, 18.0, 25.0, 12.0),
            "target_global_pairwise_residual_subspace": (
                20.0,
                14.0,
                26.0,
                9.0,
            ),
        }
        for model in EXPECTED_METHODS:
            target_clip, non_target_fid, coco_clip, coco_fid = values[model]
            base = {
                "task_id": task["id"],
                "erase_type": "style",
                "target_concepts": "Van Gogh",
                "target_count": 1,
                "model": model,
                "anchor_mode": "" if model == "original" else model,
                "anchor_concepts": "" if model == "original" else "art",
                "subspace_anchor_concepts": "",
                "subspace_anchor_count": "",
                "configured_residual_rank": "",
                "applied_residual_rank": "",
                "target_global_residual_count": "",
                "params": "" if model == "original" else "V",
                "aug_num": "" if model == "original" else (
                    10 if model == "legacy" else 0
                ),
                "threshold": "" if model == "original" else (
                    0.1 if model == "legacy" else 0.3
                ),
                "retain_scale": "" if model == "original" else (
                    1.0 if model == "legacy" else 0.5
                ),
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
        self.assertTrue(all(row["target_n_contents"] == 1 for row in summaries))
        self.assertTrue(all(row["non_target_n_contents"] == 4 for row in summaries))
        comparison = build_comparison_rows(summaries, [task])[0]
        self.assertEqual(comparison["target_clip_score_improvement"], 2.0)
        self.assertEqual(comparison["non_target_fid_improvement"], 4.0)
        self.assertEqual(comparison["mscoco_clip_score_improvement"], 1.0)
        self.assertEqual(comparison["mscoco_fid_improvement"], 3.0)

    def test_scripts_use_style_inputs_and_order_all_stages(self):
        validator = (WORKFLOW_DIR / "00_validate.sh").read_text(encoding="utf-8")
        self.assertIn('data/style.csv', validator)
        self.assertNotIn('data/instance.csv', validator)

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
        self.assertIn("raw-few-style-v1", (
            WORKFLOW_DIR / "common.sh"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
