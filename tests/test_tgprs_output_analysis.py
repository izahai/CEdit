import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "remote_scripts"
    / "toys_cosine_output"
    / "analyze_tgprs_output.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_tgprs_output", SCRIPT_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class TgprsOutputAnalysisTests(unittest.TestCase):
    def test_default_workflow_is_small_and_analysis_only(self):
        workflow_dir = SCRIPT_PATH.parent
        with (workflow_dir / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = ANALYSIS.yaml.safe_load(config_file)

        experiment = config["experiment"]
        target_count = experiment["max_targets"]
        self.assertEqual(target_count, 5)
        self.assertLessEqual(
            experiment["residual_rank"], target_count * (target_count + 1)
        )
        self.assertEqual(experiment["device"], "cuda")
        self.assertFalse(any(workflow_dir.glob("*train*.sh")))
        for stage in (
            "00_validate_repository.sh",
            "01_setup_environment.sh",
            "02_analyze.sh",
            "03_show_results.sh",
            "run_all.sh",
        ):
            self.assertTrue((workflow_dir / stage).is_file())

    def test_identity_weight_preserves_embedding_and_output_cosines(self):
        targets = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
        residuals = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
        rows = ANALYSIS.build_detail_rows(
            ["first", "second"],
            targets,
            residuals,
            [("identity", torch.eye(2))],
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(
                row["embedding_cosine"], row["output_cosine"], places=6
            )
            self.assertAlmostEqual(
                row["embedding_angle_degrees"],
                row["output_angle_degrees"],
                places=5,
            )
            self.assertAlmostEqual(
                row["residual_norm"], row["output_residual_norm"], places=6
            )

    def test_weight_changes_output_cosine_and_relative_residual(self):
        targets = torch.tensor([[[1.0, 1.0]]])
        residuals = torch.tensor([[[1.0, -1.0]]])
        rows = ANALYSIS.build_detail_rows(
            ["target"],
            targets,
            residuals,
            [("weighted", torch.diag(torch.tensor([3.0, 0.25])))],
        )

        row = rows[0]
        self.assertNotAlmostEqual(
            row["embedding_cosine"], row["output_cosine"], places=3
        )
        self.assertGreater(row["output_residual_norm"], 0.0)
        self.assertGreater(row["relative_output_residual_norm"], 0.0)

    def test_summary_groups_target_layer_rows(self):
        targets = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        residuals = torch.tensor([[[0.0, 0.5]], [[0.5, 0.0]]])
        rows = ANALYSIS.build_detail_rows(
            ["first", "second"],
            targets,
            residuals,
            [("one", torch.eye(2)), ("two", 2.0 * torch.eye(2))],
        )

        layer_summary = ANALYSIS.summarize_rows(rows, "layer_name")
        target_summary = ANALYSIS.summarize_rows(rows, "target_concept")
        overall = ANALYSIS.summarize_rows(rows)

        self.assertEqual([entry["count"] for entry in layer_summary], [2, 2])
        self.assertEqual([entry["count"] for entry in target_summary], [2, 2])
        self.assertEqual(overall[0]["count"], 4)

    def test_rejects_incompatible_weight_shape(self):
        with self.assertRaisesRegex(ValueError, "incompatible shape"):
            ANALYSIS.build_detail_rows(
                ["target"],
                torch.ones((1, 1, 2)),
                torch.ones((1, 1, 2)),
                [("bad", torch.ones((3, 4)))],
            )


if __name__ == "__main__":
    unittest.main()
