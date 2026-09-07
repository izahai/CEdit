import copy
import unittest
from pathlib import Path
from unittest import mock

import yaml

from remote_scripts.eval_few.eval_few_style_1_tgprs.evaluate_clip_fid import (
    build_comparison_rows,
    fid_score,
    frechet_distance_from_statistics,
    summarize_detailed_rows,
)
from remote_scripts.eval_few.eval_few_style_1_tgprs.workflow_config import (
    EXPECTED_METHODS,
    expected_image_counts,
    load_config,
    task_specs,
)
from src.template import painting_templates


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = (
    REPO_ROOT / "remote_scripts" / "eval_few" / "eval_few_style_1_tgprs"
)
TGPRS_METHOD = "target_global_pairwise_residual_subspace"


class EvalFewStyleTgprsWorkflowTests(unittest.TestCase):
    def test_torch_fid_matches_scipy_reference(self):
        import numpy as np
        import scipy.linalg

        rng = np.random.default_rng(7)
        samples_1 = rng.normal(size=(96, 16))
        samples_2 = rng.normal(loc=0.2, scale=1.1, size=(96, 16))
        stat_1 = {
            "mu": samples_1.mean(axis=0),
            "sigma": np.cov(samples_1, rowvar=False),
        }
        stat_2 = {
            "mu": samples_2.mean(axis=0),
            "sigma": np.cov(samples_2, rowvar=False),
        }
        diff = stat_1["mu"] - stat_2["mu"]
        covariance_mean = scipy.linalg.sqrtm(
            stat_1["sigma"].dot(stat_2["sigma"])
        ).real
        expected = (
            diff.dot(diff)
            + np.trace(stat_1["sigma"])
            + np.trace(stat_2["sigma"])
            - 2.0 * np.trace(covariance_mean)
        )

        actual = frechet_distance_from_statistics(stat_1, stat_2, "cpu")

        self.assertAlmostEqual(actual, expected, places=9)

    def test_cuda_fid_routes_final_statistics_through_torch(self):
        from torch_fidelity import metric_fid

        stat = {
            "mu": [0.0, 0.0],
            "sigma": [[1.0, 0.0], [0.0, 1.0]],
        }
        original_statistics_to_metric = metric_fid.fid_statistics_to_metric

        def calculate_metrics(**kwargs):
            self.assertTrue(kwargs["cuda"])
            return metric_fid.fid_statistics_to_metric(stat, stat, False)

        with (
            mock.patch(
                "torch_fidelity.calculate_metrics",
                side_effect=calculate_metrics,
            ),
            mock.patch(
                "remote_scripts.eval_few.eval_few_style_1_tgprs."
                "evaluate_clip_fid.frechet_distance_from_statistics",
                return_value=12.5,
            ) as distance,
        ):
            actual = fid_score(
                "edited",
                "original",
                "manifest",
                "cache",
                32,
                "2048",
                True,
            )

        self.assertEqual(actual, 12.5)
        distance.assert_called_once_with(stat, stat, "cuda")
        self.assertIs(
            metric_fid.fid_statistics_to_metric,
            original_statistics_to_metric,
        )

    def test_full_workflow_is_tgprs_only_style_matrix(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        tasks = task_specs(config)

        self.assertEqual(EXPECTED_METHODS, ["original", TGPRS_METHOD])
        self.assertEqual(config["experiment"]["methods"], EXPECTED_METHODS)
        self.assertEqual(
            config["experiment"]["edited_methods"], [TGPRS_METHOD]
        )
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
            "original_few": 1500,
            "edited_few": 4500,
            "original_coco": 1000,
            "edited_coco": 3000,
            "total": 10000,
        })

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
            self.assertTrue(all(
                artist not in normalized for artist in evaluated_artists
            ))

    def test_task_anchor_override_updates_effective_rank(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        config["tasks"][0]["subspace_anchor_concepts"] = ["art", "painting"]

        task = task_specs(config)[0]

        self.assertEqual(task["subspace_anchor_count"], 2)
        self.assertEqual(task["target_global_residual_count"], 2)
        self.assertEqual(task["applied_residual_rank"], 2)

    def test_single_target_without_subspace_anchors_is_invalid(self):
        config = load_config(WORKFLOW_DIR / "workflow.yaml")
        invalid = copy.deepcopy(config)
        invalid["tasks"][0]["subspace_anchor_concepts"] = []

        with self.assertRaisesRegex(ValueError, "no TGPRS residual vectors"):
            task_specs(invalid)

    def test_smoke_workflow_has_exact_reduced_size(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        tasks = task_specs(config)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], "van_gogh")
        self.assertEqual(tasks[0]["contents"], ["Van Gogh", "Caravaggio"])
        self.assertEqual(config["experiment"]["fid_feature_layer"], 64)
        self.assertEqual(expected_image_counts(config), {
            "original_few": 20,
            "edited_few": 20,
            "original_coco": 10,
            "edited_coco": 10,
            "total": 60,
        })

    def test_training_config_matches_style_tgprs_parameters(self):
        with (
            WORKFLOW_DIR
            / "train_config_target_global_pairwise_residual_subspace.yaml"
        ).open() as config_file:
            tgprs = yaml.safe_load(config_file)

        self.assertFalse((WORKFLOW_DIR / "train_config_legacy.yaml").exists())
        self.assertEqual(tgprs["anchor_mode"], TGPRS_METHOD)
        self.assertEqual(tgprs["aug_num"], 0)
        self.assertEqual(tgprs["threshold"], 0.3)
        self.assertEqual(tgprs["retain_scale"], 0.5)
        self.assertEqual(tgprs["residual_rank"], 30)
        self.assertEqual(tgprs["residual_scale"], 1.0)

    def test_comparison_contains_original_and_tgprs_without_legacy(self):
        config = load_config(WORKFLOW_DIR / "workflow_smoke.yaml")
        task = task_specs(config)[0]
        rows = []
        values = {
            "original": (29.0, 0.0, 27.0, 0.0),
            TGPRS_METHOD: (20.0, 14.0, 26.0, 9.0),
        }
        for model in EXPECTED_METHODS:
            target_clip, non_target_fid, coco_clip, coco_fid = values[model]
            edited = model == TGPRS_METHOD
            base = {
                "task_id": task["id"],
                "erase_type": "style",
                "target_concepts": "Van Gogh",
                "target_count": 1,
                "model": model,
                "anchor_mode": TGPRS_METHOD if edited else "",
                "anchor_concepts": "art" if edited else "",
                "subspace_anchor_concepts": "[]" if edited else "",
                "subspace_anchor_count": 30 if edited else "",
                "configured_residual_rank": 30 if edited else "",
                "applied_residual_rank": 30 if edited else "",
                "target_global_residual_count": 30 if edited else "",
                "params": "V" if edited else "",
                "aug_num": 0 if edited else "",
                "threshold": 0.3 if edited else "",
                "retain_scale": 0.5 if edited else "",
                "n_images": 10,
            }
            rows.extend([
                {
                    **base,
                    "content": "Van Gogh",
                    "content_role": "target",
                    "clip_score": target_clip,
                    "fid_vs_original": 0.0,
                },
                {
                    **base,
                    "content": "Caravaggio",
                    "content_role": "non_target",
                    "clip_score": 25.0,
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

        self.assertEqual(comparison["original_target_clip_score"], 29.0)
        self.assertEqual(comparison["tgprs_target_clip_score"], 20.0)
        self.assertEqual(comparison["original_non_target_fid"], 0.0)
        self.assertEqual(comparison["tgprs_non_target_fid"], 14.0)
        self.assertNotIn("legacy_target_clip_score", comparison)

    def test_scripts_use_style_inputs_and_order_all_stages(self):
        validator = (WORKFLOW_DIR / "00_validate.sh").read_text(encoding="utf-8")
        self.assertIn("data/style.csv", validator)
        self.assertNotIn("data/instance.csv", validator)

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
        self.assertIn(
            "raw-few-style-tgprs-v1",
            (WORKFLOW_DIR / "common.sh").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
