import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from remote_scripts.eval_retain_aware_target_global_pairwise_residual_subspace.workflow_config import (
    build_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = (
    REPO_ROOT
    / "remote_scripts"
    / "eval_retain_aware_target_global_pairwise_residual_subspace"
)
PAPER_CONFIG = (
    REPO_ROOT
    / "remote_scripts"
    / "eval_paper_comparison"
    / "train_config_target_global_pairwise_residual_subspace.yaml"
)
ANCHOR_MODE = "retain_aware_target_global_pairwise_residual_subspace"


class RetainAwareTargetGlobalWorkflowTests(unittest.TestCase):
    def test_runs_only_new_method_on_three_celebrity_benchmarks(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(config["experiment"]["anchor_mode"], ANCHOR_MODE)
        self.assertEqual(
            config["experiment"]["benchmark_names"],
            ["10_celebrity", "50_celebrity", "100_celebrity"],
        )
        self.assertNotIn("methods", config["experiment"])
        self.assertFalse((WORKFLOW_DIR / "02_generate_original.sh").exists())
        self.assertFalse((WORKFLOW_DIR / "train_config_legacy.yaml").exists())

    def test_train_config_matches_paper_except_mode_and_threshold(self):
        with (WORKFLOW_DIR / "train_config.yaml").open(encoding="utf-8") as file:
            retain_aware = yaml.safe_load(file)
        with PAPER_CONFIG.open(encoding="utf-8") as file:
            expected = yaml.safe_load(file)

        expected["anchor_mode"] = ANCHOR_MODE
        expected["threshold"] = 1e-5
        self.assertEqual(retain_aware, expected)

    def test_workflow_loader_exports_new_mode_and_isolated_output_root(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        with tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"WORKSPACE_DIR": workspace}, clear=False):
                environment = build_environment(config, str(WORKFLOW_DIR))

        self.assertEqual(environment["ANCHOR_MODE"], ANCHOR_MODE)
        self.assertTrue(
            environment["OUTPUT_ROOT"].endswith(
                "cedit_ce_eval_outputs_" + ANCHOR_MODE
            )
        )


if __name__ == "__main__":
    unittest.main()
