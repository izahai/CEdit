import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "remote_scripts"
    / "toys_target_pairwise_cosine"
    / "analyze_target_pairwise_cosine.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_target_pairwise_cosine", SCRIPT_PATH
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class TargetPairwiseCosineAnalysisTests(unittest.TestCase):
    def test_pairwise_cosine_returns_expected_matrix(self):
        vectors = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        matrix = ANALYSIS.pairwise_cosine(vectors)

        self.assertEqual(tuple(matrix.shape), (3, 3))
        self.assertTrue(torch.allclose(matrix, matrix.T))
        self.assertTrue(torch.allclose(matrix.diagonal(), torch.ones(3)))
        self.assertAlmostEqual(matrix[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 2 ** -0.5, places=6)

    def test_pair_classes_have_expected_counts(self):
        target_count = 3
        labels = ["a", "b", "c", "person", "<empty>"]
        rows = ANALYSIS.build_pair_rows(
            -1,
            "embedding",
            labels,
            torch.eye(target_count + 2),
            target_count,
        )
        counts = {
            pair_type: sum(row["pair_type"] == pair_type for row in rows)
            for pair_type in ANALYSIS.PAIR_TYPES
        }

        self.assertEqual(counts, {
            "target_target": 3,
            "target_person": 3,
            "target_empty": 3,
            "person_empty": 1,
        })
        self.assertEqual(len(rows), 10)

    def test_summary_separates_anchors_and_embedding_distortion(self):
        target_count = 2
        labels = ["a", "b", "person", "<empty>"]
        vectors = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ])
        matrix = ANALYSIS.pairwise_cosine(vectors)
        rows = ANALYSIS.build_pair_rows(
            -1, "embedding", labels, matrix, target_count
        )
        target_values = [
            row["cosine"]
            for row in rows
            if row["pair_type"] == "target_target"
        ]
        summary = ANALYSIS.summarize_space(
            rows,
            torch.linalg.vector_norm(vectors, dim=1),
            target_count,
            target_values,
        )

        self.assertEqual(summary["target_target_count"], 1)
        self.assertEqual(summary["target_person_count"], 2)
        self.assertEqual(summary["target_empty_count"], 2)
        self.assertEqual(summary["person_empty_count"], 1)
        self.assertAlmostEqual(summary["target_correlation_to_embedding"], 1.0)
        self.assertAlmostEqual(summary["target_rmse_from_embedding"], 0.0)

    def test_workflow_is_analysis_only_and_uses_two_anchors(self):
        workflow_dir = SCRIPT_PATH.parent
        with (workflow_dir / "workflow.yaml").open(encoding="utf-8") as config_file:
            config = ANALYSIS.yaml.safe_load(config_file)

        experiment = config["experiment"]
        self.assertEqual(experiment["target_count"], 50)
        self.assertEqual(experiment["anchor_prompts"], ["person", ""])
        self.assertFalse(any(workflow_dir.glob("*train*.sh")))
        for stage in (
            "00_validate_repository.sh",
            "01_setup_environment.sh",
            "02_analyze.sh",
            "03_show_results.sh",
            "run_all.sh",
        ):
            self.assertTrue((workflow_dir / stage).is_file())


if __name__ == "__main__":
    unittest.main()
