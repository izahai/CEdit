import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from remote_scripts.eval_paper_comparison_clip_fid.evaluate_mscoco_clip_fid import (
    image_dir_for_run,
    load_prompt_records,
    validate_image_directory,
)
from remote_scripts.eval_paper_comparison_clip_fid.workflow_config import (
    build_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "remote_scripts" / "eval_paper_comparison_clip_fid"


class PaperComparisonClipFidWorkflowTests(unittest.TestCase):
    def test_config_exports_mscoco_settings_and_separate_output_root(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        with tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"WORKSPACE_DIR": workspace}, clear=False):
                environment = build_environment(config, str(WORKFLOW_DIR))

        self.assertEqual(environment["MSCOCO_NUM_PROMPTS"], 1000)
        self.assertEqual(environment["CLIP_BATCH_SIZE"], 32)
        self.assertEqual(
            environment["CLIP_MODEL"], "openai/clip-vit-large-patch14"
        )
        self.assertTrue(
            environment["OUTPUT_ROOT"].endswith(
                "cedit_ce_eval_outputs_paper_comparison_clip_fid"
            )
        )

    def test_loads_first_prompts_and_builds_coco_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "mscoco.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "text"])
                writer.writeheader()
                writer.writerow({"image_id": "42", "text": "first prompt"})
                writer.writerow({"image_id": "7", "text": "second prompt"})
            records = load_prompt_records(csv_path, 1)

        self.assertEqual(records, [{
            "filename": "COCO_val2014_000000000042.png",
            "prompt": "first prompt",
        }])

    def test_validates_exact_image_set(self):
        records = [{"filename": "expected.png", "prompt": "a prompt"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            (image_dir / "unexpected.png").touch()
            with self.assertRaisesRegex(ValueError, "missing 1"):
                validate_image_directory(image_dir, records)

            (image_dir / "unexpected.png").unlink()
            (image_dir / "expected.png").touch()
            validate_image_directory(image_dir, records)

    def test_uses_one_original_directory_and_checkpoint_directories(self):
        root = Path("/outputs/mscoco")
        self.assertEqual(
            image_dir_for_run(root, "original", "10_celebrity"),
            root / "original" / "coco" / "original",
        )
        self.assertEqual(
            image_dir_for_run(root, "legacy", "10_celebrity"),
            root / "legacy" / "10_celebrity" / "coco" / "edit",
        )

    def test_run_all_includes_generation_evaluation_and_summary(self):
        run_all = (WORKFLOW_DIR / "run_all.sh").read_text(encoding="utf-8")
        generation = run_all.index("07_generate_mscoco.sh")
        evaluation = run_all.index("08_eval_clip_fid.sh")
        summary = run_all.index("09_summarize_results.sh")
        self.assertLess(generation, evaluation)
        self.assertLess(evaluation, summary)

    def test_environment_pins_scipy_for_torch_fidelity_sqrtm_api(self):
        setup_script = (WORKFLOW_DIR / "01_setup_environment.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('"scipy<1.18"', setup_script)


if __name__ == "__main__":
    unittest.main()
