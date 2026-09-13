import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from exp_results.val_clip_adv.evaluate_clip_fid import (
    build_comparison_rows,
    load_learned_anchor_metadata,
    summarize_detailed_rows,
    write_report,
)
from exp_results.val_clip_adv.workflow_config import (
    EXPECTED_EDITED_METHODS,
    EXPECTED_METHODS,
    METHOD_LEARNED,
    METHOD_LEGACY,
    METHOD_ORIGINAL,
    METHOD_TGPRS,
    expected_image_counts,
    load_config,
    load_train_configs,
    validate_learn_anchor_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "exp_results" / "val_clip_adv"
WORKFLOW_CONFIG = WORKFLOW_DIR / "workflow.yaml"
SMOKE_WORKFLOW_CONFIG = WORKFLOW_DIR / "workflow_smoke.yaml"


class ValClipAdvWorkflowTests(unittest.TestCase):
    def test_full_workflow_method_order_and_image_count(self):
        config = load_config(WORKFLOW_CONFIG)

        self.assertEqual(config["experiment"]["methods"], EXPECTED_METHODS)
        self.assertEqual(
            config["experiment"]["edited_methods"],
            EXPECTED_EDITED_METHODS,
        )
        self.assertEqual(
            expected_image_counts(config),
            {
                "original_few": 1500,
                "edited_few": 4500,
                "original_coco": 1000,
                "edited_coco": 3000,
                "total": 10000,
            },
        )

    def test_smoke_workflow_image_count(self):
        config = load_config(SMOKE_WORKFLOW_CONFIG)

        self.assertEqual(config["experiment"]["profile"], "smoke")
        self.assertEqual(config["experiment"]["num_samples_per_prompt"], 1)
        self.assertEqual(config["experiment"]["mscoco_num_prompts"], 50)
        self.assertEqual(
            expected_image_counts(config),
            {
                "original_few": 150,
                "edited_few": 450,
                "original_coco": 50,
                "edited_coco": 150,
                "total": 800,
            },
        )

    def test_method_standard_training_settings_are_fixed(self):
        configs = load_train_configs(WORKFLOW_DIR)
        legacy = configs[METHOD_LEGACY]
        tgprs = configs[METHOD_TGPRS]
        learned = configs[METHOD_LEARNED]

        self.assertEqual(legacy["anchor_source"], "text")
        self.assertEqual(legacy["anchor_mode"], "legacy")
        self.assertEqual(legacy["aug_num"], 10)
        self.assertEqual(legacy["threshold"], 0.1)

        self.assertEqual(tgprs["anchor_mode"], METHOD_TGPRS)
        self.assertFalse(tgprs["erase_style"])
        self.assertEqual(tgprs["residual_rank"], 30)
        self.assertEqual(tgprs["aug_num"], 0)
        self.assertEqual(tgprs["threshold"], 1.0)
        self.assertEqual(len(tgprs["subspace_anchor_concepts"]), 100)
        self.assertTrue(
            all(
                "van gogh" not in anchor.casefold()
                for anchor in tgprs["subspace_anchor_concepts"]
            )
        )

        self.assertEqual(learned["anchor_source"], "learned")
        self.assertEqual(learned["anchor_mode"], "legacy")
        self.assertFalse(learned.get("erase_style", False))
        for key in ("params", "aug_num", "threshold", "retain_scale"):
            self.assertEqual(learned[key], legacy[key])

    def test_learned_anchor_search_configuration(self):
        config = validate_learn_anchor_config(WORKFLOW_DIR)

        self.assertEqual(config["target_concepts"], ["Van Gogh"])
        self.assertEqual(config["num_prefix_tokens"], 4)
        self.assertEqual(config["num_reference_images"], 4)
        self.assertEqual(config["num_validation_images"], 1)
        self.assertEqual(config["iterations"], 1000)
        self.assertEqual(config["validation_samples"], 16)
        self.assertEqual(config["clip_model"], "openai/clip-vit-large-patch14")

    def test_k74_configuration_uses_maximum_prefix(self):
        from exp_results.val_clip_adv.workflow_config import load_yaml

        config = load_yaml(WORKFLOW_DIR / "learn_anchor_config_k74.yaml")
        self.assertEqual(config["num_prefix_tokens"], 74)
        self.assertEqual(config["target_concepts"], ["Van Gogh"])

    def test_shell_stages_are_valid_and_ordered(self):
        expected = [
            "00_validate.sh",
            "01_setup_environment.sh",
            "02_learn_anchor.sh",
            "03_generate_original.sh",
            "04_train.sh",
            "05_generate_edits.sh",
            "06_generate_mscoco.sh",
            "07_evaluate.sh",
        ]
        run_all = (WORKFLOW_DIR / "run_all.sh").read_text(encoding="utf-8")
        positions = [run_all.index(stage) for stage in expected]
        self.assertEqual(positions, sorted(positions))
        for script in WORKFLOW_DIR.glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_learned_anchor_metadata_is_validated(self):
        task = load_config(WORKFLOW_CONFIG)["task"]
        manifest = {
            "status": "complete",
            "target_order": ["Van Gogh"],
            "targets": [
                {
                    "prefix_token_ids": [1, 2, 3, 4],
                    "decoded_prefix": "test prefix",
                    "best_step": 150,
                    "best_validation_clip_similarity": 0.12,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "learned_anchors" / "van_gogh"
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            metadata = load_learned_anchor_metadata(temp_dir, task)

        self.assertEqual(metadata["prefix_token_ids"], [1, 2, 3, 4])
        self.assertEqual(metadata["best_step"], 150)
        self.assertEqual(len(metadata["manifest_sha256"]), 64)

    def test_summary_comparison_and_markdown_report(self):
        config = load_config(WORKFLOW_CONFIG)
        task = config["task"]
        scores = {
            METHOD_ORIGINAL: (29.0, 0.0, 30.0, 0.0, 26.0, 0.0),
            METHOD_LEGACY: (25.0, 10.0, 24.0, 12.0, 25.0, 20.0),
            METHOD_TGPRS: (23.0, 12.0, 23.0, 9.0, 25.5, 18.0),
            METHOD_LEARNED: (20.0, 14.0, 22.0, 11.0, 26.0, 19.0),
        }
        metadata = {
            "anchor_source": "",
            "anchor_mode": "",
            "anchor_concepts": "",
            "subspace_anchor_count": "",
            "configured_residual_rank": "",
            "applied_residual_rank": "",
            "params": "",
            "aug_num": "",
            "threshold": "",
            "retain_scale": "",
            "learned_prefix_token_ids": "",
            "learned_decoded_prefix": "",
            "anchor_search_best_step": "",
            "anchor_search_validation_clip_similarity": "",
        }
        rows = []
        for method, values in scores.items():
            target_clip, target_fid, non_clip, non_fid, coco_clip, coco_fid = values
            rows.append({
                "task_id": "van_gogh",
                "model": method,
                "content": "Van Gogh",
                "content_role": "target",
                "n_images": 300,
                "clip_score": target_clip,
                "fid_vs_original": target_fid,
                **metadata,
            })
            for content in ("Picasso", "Monet", "Paul Gauguin", "Caravaggio"):
                rows.append({
                    "task_id": "van_gogh",
                    "model": method,
                    "content": content,
                    "content_role": "non_target",
                    "n_images": 300,
                    "clip_score": non_clip,
                    "fid_vs_original": non_fid,
                    **metadata,
                })
            rows.append({
                "task_id": "van_gogh",
                "model": method,
                "content": "coco",
                "content_role": "coco",
                "n_images": 1000,
                "clip_score": coco_clip,
                "fid_vs_original": coco_fid,
                **metadata,
            })

        summaries = summarize_detailed_rows(rows, task, EXPECTED_METHODS)
        comparisons = build_comparison_rows(summaries, task)
        by_method = {row["comparison_method"]: row for row in comparisons}

        self.assertEqual(len(summaries), 4)
        self.assertEqual(by_method[METHOD_TGPRS]["target_clip_score_improvement"], 2.0)
        self.assertEqual(by_method[METHOD_TGPRS]["non_target_fid_improvement"], 3.0)
        self.assertEqual(by_method[METHOD_LEARNED]["target_clip_score_improvement"], 5.0)
        self.assertEqual(by_method[METHOD_LEARNED]["mscoco_fid_improvement"], 1.0)

        learned_metadata = {
            "prefix_token_ids": [1, 2, 3, 4],
            "decoded_prefix": "test prefix",
            "best_step": 150,
            "best_validation_clip_similarity": 0.12,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.md"
            write_report(report, config, summaries, comparisons, learned_metadata)
            text = report.read_text(encoding="utf-8")

        self.assertIn("CLIP-guided learned anchor", text)
        self.assertIn("Positive improvement values favor", text)
        self.assertIn("test prefix", text)


if __name__ == "__main__":
    unittest.main()
