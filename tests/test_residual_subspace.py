import unittest

import torch

from src.residual_subspace import build_smallest_cosine_subspace_residuals


class SmallestCosineSubspaceResidualTests(unittest.TestCase):
    def test_selects_smallest_mean_cosine_residuals(self):
        targets = torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        )
        legacy = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.8, 0.6, 0.0]],
                [[-1.0, 0.0, 0.0]],
            ]
        )

        _, diagnostics = build_smallest_cosine_subspace_residuals(
            targets,
            targets + legacy,
            top_k=1,
        )

        self.assertEqual(diagnostics["subspace_selected_indices"], [2])
        self.assertEqual(diagnostics["subspace_effective_rank"], 1)

    def test_uses_negative_target_projection_and_preserves_norm(self):
        targets = torch.tensor(
            [
                [[3.0, 4.0, 7.0]],
                [[2.0, -1.0, 5.0]],
            ]
        )
        legacy = torch.tensor(
            [
                [[2.0, 0.0, 0.0]],
                [[0.0, 3.0, 0.0]],
            ]
        )

        residuals, diagnostics = build_smallest_cosine_subspace_residuals(
            targets,
            targets + legacy,
            top_k=2,
        )

        expected = torch.tensor(
            [
                [[-1.2, -1.6, 0.0]],
                [[-6.0 / torch.sqrt(torch.tensor(5.0)), 3.0 / torch.sqrt(torch.tensor(5.0)), 0.0]],
            ]
        )
        torch.testing.assert_close(residuals, expected)
        torch.testing.assert_close(
            residuals.flatten(1).norm(dim=1),
            legacy.flatten(1).norm(dim=1),
        )
        self.assertLess(diagnostics["subspace_max_projection_error"], 1e-6)
        self.assertEqual(diagnostics["subspace_legacy_fallback_count"], 0)

    def test_falls_back_to_projected_legacy_residual(self):
        targets = torch.tensor([[[0.0, 0.0, 5.0]], [[0.0, 0.0, 2.0]]])
        legacy = torch.tensor([[[2.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]])

        residuals, diagnostics = build_smallest_cosine_subspace_residuals(
            targets,
            targets + legacy,
            top_k=2,
        )

        torch.testing.assert_close(residuals, legacy)
        self.assertEqual(diagnostics["subspace_target_projection_fallback_count"], 2)
        self.assertEqual(diagnostics["subspace_legacy_fallback_count"], 2)
        self.assertEqual(diagnostics["subspace_basis_fallback_count"], 0)

    def test_falls_back_to_first_basis_vector(self):
        targets = torch.tensor([[[0.0, 2.0, 0.0]], [[0.0, 3.0, 0.0]]])
        legacy = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])

        residuals, diagnostics = build_smallest_cosine_subspace_residuals(
            targets,
            targets + legacy,
            top_k=1,
        )

        self.assertEqual(diagnostics["subspace_selected_indices"], [0])
        self.assertEqual(diagnostics["subspace_target_projection_fallback_count"], 2)
        self.assertEqual(diagnostics["subspace_basis_fallback_count"], 1)
        self.assertEqual(residuals[1].norm().item(), 0.0)

    def test_supports_multi_token_embeddings(self):
        targets = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        )
        legacy = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 2.0]],
            ]
        )

        residuals, _ = build_smallest_cosine_subspace_residuals(
            targets,
            targets + legacy,
            top_k=2,
        )

        self.assertEqual(residuals.shape, targets.shape)
        torch.testing.assert_close(
            residuals.flatten(1).norm(dim=1),
            legacy.flatten(1).norm(dim=1),
        )

    def test_rejects_invalid_top_k_and_zero_rank_subspace(self):
        targets = torch.zeros(2, 1, 3)
        anchors = targets.clone()

        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_smallest_cosine_subspace_residuals(targets, anchors, top_k=0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_smallest_cosine_subspace_residuals(targets, anchors, top_k=3)
        with self.assertRaisesRegex(ValueError, "zero-rank"):
            build_smallest_cosine_subspace_residuals(targets, anchors, top_k=2)


if __name__ == "__main__":
    unittest.main()
