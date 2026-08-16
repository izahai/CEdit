import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from remote_scripts.eval_paper_comparison.workflow_config import build_environment


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "remote_scripts" / "eval_paper_comparison"


class PaperComparisonWorkflowTests(unittest.TestCase):
    def test_workflow_defines_three_methods_and_benchmarks(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(
            config["experiment"]["methods"],
            [
                "original",
                "legacy",
                "target_global_pairwise_residual_subspace",
            ],
        )
        self.assertEqual(
            config["experiment"]["benchmark_names"],
            ["10_celebrity", "50_celebrity", "100_celebrity"],
        )

    def test_training_configs_lock_paper_and_new_method_settings(self):
        with (WORKFLOW_DIR / "train_config_legacy.yaml").open(
            encoding="utf-8"
        ) as config_file:
            legacy = yaml.safe_load(config_file)
        with (
            WORKFLOW_DIR
            / "train_config_target_global_pairwise_residual_subspace.yaml"
        ).open(encoding="utf-8") as config_file:
            target_global = yaml.safe_load(config_file)

        self.assertEqual(legacy["anchor_mode"], "legacy")
        self.assertEqual(legacy["aug_num"], 10)
        self.assertEqual(legacy["threshold"], 1e-4)
        self.assertEqual(legacy["retain_scale"], 0.05)
        self.assertEqual(
            target_global["anchor_mode"],
            "target_global_pairwise_residual_subspace",
        )
        self.assertEqual(target_global["aug_num"], 0)
        self.assertEqual(target_global["residual_rank"], 30)

    def test_workflow_loader_exports_method_list_and_output_root(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        with tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"WORKSPACE_DIR": workspace}, clear=False):
                environment = build_environment(config, str(WORKFLOW_DIR))

        self.assertEqual(
            environment["METHODS_RAW"],
            "original legacy target_global_pairwise_residual_subspace",
        )
        self.assertTrue(
            environment["OUTPUT_ROOT"].endswith(
                "cedit_ce_eval_outputs_paper_comparison"
            )
        )


if __name__ == "__main__":
    unittest.main()
