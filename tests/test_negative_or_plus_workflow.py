import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ablation.negative_or_plus.workflow_config import build_environment


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "ablation" / "negative_or_plus"


class NegativeOrPlusWorkflowTests(unittest.TestCase):
    def test_workflow_compares_two_directions_on_three_celebrity_benchmarks(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(
            config["experiment"]["methods"],
            ["negative_projection", "positive_projection"],
        )
        self.assertEqual(
            config["experiment"]["benchmark_names"],
            ["10_celebrity", "50_celebrity", "100_celebrity"],
        )

    def test_training_configs_differ_only_by_projection_direction(self):
        configs = []
        for direction in ("negative", "positive"):
            with (WORKFLOW_DIR / f"train_config_{direction}.yaml").open(
                encoding="utf-8"
            ) as config_file:
                configs.append(yaml.safe_load(config_file))

        negative, positive = configs
        self.assertEqual(negative["target_projection_direction"], "negative")
        self.assertEqual(positive["target_projection_direction"], "positive")
        self.assertEqual(
            {key: value for key, value in negative.items() if key != "target_projection_direction"},
            {key: value for key, value in positive.items() if key != "target_projection_direction"},
        )
        self.assertEqual(
            negative["anchor_mode"],
            "target_global_pairwise_residual_subspace",
        )
        self.assertEqual(negative["residual_rank"], 30)

    def test_workflow_loader_exports_ablation_output_root(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        with tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"WORKSPACE_DIR": workspace}, clear=False):
                environment = build_environment(config, str(WORKFLOW_DIR))

        self.assertEqual(
            environment["METHODS_RAW"],
            "negative_projection positive_projection",
        )
        self.assertEqual(
            environment["BENCHMARK_NAMES_RAW"],
            "10_celebrity 50_celebrity 100_celebrity",
        )
        self.assertTrue(
            environment["OUTPUT_ROOT"].endswith(
                "cedit_ce_eval_outputs_negative_or_plus"
            )
        )


if __name__ == "__main__":
    unittest.main()
