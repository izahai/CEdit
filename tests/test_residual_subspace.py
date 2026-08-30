import unittest
from unittest.mock import patch

import torch

from src.residual_subspace import (
    build_global_pairwise_residual_matrix,
    build_global_pairwise_residual_subspace_residuals,
    build_largest_anchor_cosine_subspace_residuals,
    build_mean_norm_target_global_pairwise_residual_subspace_residuals,
    build_negative_target_normalized_residual_subspace_residuals,
    build_retain_aware_target_global_pairwise_residual_subspace_residuals,
    build_smallest_cosine_subspace_residuals,
    build_target_global_pairwise_residual_subspace_residuals,
)


class GlobalPairwiseResidualSubspaceTests(unittest.TestCase):
    def test_builds_ordered_pairwise_and_extra_anchor_residual_blocks(self):
        concepts = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 2.0]], [[3.0, 3.0]]]
        )
        extra_anchors = torch.tensor([[[0.0, 0.0]], [[2.0, 1.0]]])

        residuals = build_global_pairwise_residual_matrix(
            concepts,
            extra_anchors,
        )

        expected = torch.tensor(
            [
                [-1.0, 2.0],
                [2.0, 3.0],
                [-1.0, 0.0],
                [1.0, 1.0],
                [1.0, -2.0],
                [3.0, 1.0],
                [0.0, -2.0],
                [2.0, -1.0],
                [-2.0, -3.0],
                [-3.0, -1.0],
                [-3.0, -3.0],
                [-1.0, -2.0],
            ]
        )
        self.assertEqual(residuals.shape, (3 * (3 + 1), 2))
        torch.testing.assert_close(residuals, expected)

    def test_builds_target_pair_residuals_without_extra_anchors(self):
        concepts = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
        extra_anchors = torch.empty(0, 1, 2)

        residuals = build_global_pairwise_residual_matrix(
            concepts,
            extra_anchors,
        )

        expected = torch.tensor([[-1.0, 2.0], [1.0, -2.0]])
        self.assertEqual(residuals.shape, (2, 2))
        torch.testing.assert_close(residuals, expected)

    def test_rejects_one_concept_without_extra_anchors(self):
        concepts = torch.tensor([[[1.0, 0.0]]])
        extra_anchors = torch.empty(0, 1, 2)

        with self.assertRaisesRegex(ValueError, "no vectors"):
            build_global_pairwise_residual_matrix(concepts, extra_anchors)

    def test_normalizes_every_global_residual_before_exact_svd(self):
        concepts = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
        extra_anchors = torch.tensor([[[4.0, 0.0]], [[0.0, 8.0]]])
        targets = torch.tensor([[[1.0, 1.0]]])
        anchors = torch.tensor([[[2.0, 1.0]]])
        captured = {}
        exact_svd = torch.linalg.svd

        def capture_svd(matrix, *args, **kwargs):
            captured["matrix"] = matrix.detach().clone()
            return exact_svd(matrix, *args, **kwargs)

        with patch("src.residual_subspace.torch.linalg.svd", side_effect=capture_svd):
            _, diagnostics = build_global_pairwise_residual_subspace_residuals(
                targets,
                anchors,
                concepts,
                extra_anchors,
                rank=2,
            )

        torch.testing.assert_close(
            captured["matrix"].norm(dim=1),
            torch.ones(6),
        )
        self.assertTrue(diagnostics["subspace_normalized_before_svd"])
        self.assertEqual(diagnostics["global_subspace_residual_count"], 6)
        self.assertEqual(diagnostics["global_subspace_residual_shape"], (6, 2))

    def test_uses_negative_target_projection_and_preserves_legacy_norm(self):
        concepts = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        extra_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]]
        )
        targets = torch.tensor([[[3.0, 4.0, 0.0]]])
        legacy = torch.tensor([[[0.0, 2.0, 0.0]]])

        residuals, diagnostics = build_global_pairwise_residual_subspace_residuals(
            targets,
            targets + legacy,
            concepts,
            extra_anchors,
            rank=1,
        )

        torch.testing.assert_close(residuals, torch.tensor([[[-2.0, 0.0, 0.0]]]))
        torch.testing.assert_close(residuals.norm(), legacy.norm())
        self.assertLess(diagnostics["subspace_max_projection_error"], 1e-6)
        self.assertEqual(diagnostics["subspace_basis_rank"], 1)

    def test_preserves_both_negative_target_fallbacks(self):
        concepts = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        extra_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]]
        )
        targets = torch.tensor(
            [[[0.0, 2.0, 0.0]], [[0.0, 3.0, 0.0]]]
        )
        legacy = torch.tensor(
            [[[2.0, 0.0, 0.0]], [[0.0, 0.0, 4.0]]]
        )

        residuals, diagnostics = build_global_pairwise_residual_subspace_residuals(
            targets,
            targets + legacy,
            concepts,
            extra_anchors,
            rank=1,
        )

        torch.testing.assert_close(residuals[0], legacy[0])
        self.assertEqual(residuals[1, :, 1:].abs().max().item(), 0.0)
        self.assertAlmostEqual(residuals[1].norm().item(), 4.0)
        self.assertEqual(diagnostics["subspace_target_projection_fallback_count"], 2)
        self.assertEqual(diagnostics["subspace_legacy_fallback_count"], 1)
        self.assertEqual(diagnostics["subspace_basis_fallback_count"], 1)

    def test_rejects_invalid_rank_and_zero_rank_global_matrix(self):
        concepts = torch.zeros(1, 1, 2)
        extra_anchors = torch.zeros(2, 1, 2)
        targets = torch.ones(1, 1, 2)

        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_global_pairwise_residual_subspace_residuals(
                targets,
                targets,
                concepts,
                extra_anchors,
                rank=3,
            )
        with self.assertRaisesRegex(ValueError, "zero-rank"):
            build_global_pairwise_residual_subspace_residuals(
                targets,
                targets,
                concepts,
                extra_anchors,
                rank=1,
            )


class TargetGlobalPairwiseResidualSubspaceTests(unittest.TestCase):
    def test_retain_aware_mode_projects_raw_residuals_before_normalization(self):
        targets = torch.tensor(
            [[[1.0, 0.0, 1.0]], [[0.0, 2.0, 2.0]]]
        )
        anchors = targets + torch.tensor(
            [[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]]
        )
        extra_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[2.0, 1.0, 3.0]]]
        )
        retain_projection = torch.diag(torch.tensor([1.0, 1.0, 0.0]))
        captured = {}
        exact_svd = torch.linalg.svd

        def capture_svd(matrix, *args, **kwargs):
            captured["matrix"] = matrix.detach().clone()
            return exact_svd(matrix, *args, **kwargs)

        with patch("src.residual_subspace.torch.linalg.svd", side_effect=capture_svd):
            residuals, diagnostics = (
                build_retain_aware_target_global_pairwise_residual_subspace_residuals(
                    targets,
                    anchors,
                    extra_anchor_embeddings=extra_anchors,
                    retain_projection=retain_projection,
                    rank=2,
                )
            )

        expected = build_global_pairwise_residual_matrix(targets, extra_anchors)
        expected = expected @ retain_projection
        expected = expected / expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(captured["matrix"], expected)
        torch.testing.assert_close(
            residuals[..., 2],
            torch.zeros_like(residuals[..., 2]),
        )
        self.assertTrue(diagnostics["subspace_input_projected"])
        self.assertTrue(diagnostics["retain_aware_subspace"])

    def test_retain_aware_mode_requires_a_compatible_projection(self):
        targets = torch.tensor([[[1.0, 0.0, 1.0]]])
        extra_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[2.0, 1.0, 3.0]]]
        )

        with self.assertRaisesRegex(ValueError, "retain-low projection"):
            build_retain_aware_target_global_pairwise_residual_subspace_residuals(
                targets,
                targets + 1.0,
                extra_anchor_embeddings=extra_anchors,
                retain_projection=None,
                rank=1,
            )
        with self.assertRaisesRegex(ValueError, "projection shape"):
            build_retain_aware_target_global_pairwise_residual_subspace_residuals(
                targets,
                targets + 1.0,
                extra_anchor_embeddings=extra_anchors,
                retain_projection=torch.eye(2),
                rank=1,
            )

    def test_uses_only_targets_and_fixed_anchors_for_the_basis(self):
        targets = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 2.0]], [[3.0, 3.0]]]
        )
        anchors = targets + torch.tensor(
            [[[0.5, 0.0]], [[0.0, 0.5]], [[0.5, 0.5]]]
        )
        extra_anchors = torch.tensor([[[0.0, 0.0]], [[2.0, 1.0]]])
        captured = {}
        exact_svd = torch.linalg.svd

        def capture_svd(matrix, *args, **kwargs):
            captured["matrix"] = matrix.detach().clone()
            return exact_svd(matrix, *args, **kwargs)

        with patch("src.residual_subspace.torch.linalg.svd", side_effect=capture_svd):
            _, diagnostics = (
                build_target_global_pairwise_residual_subspace_residuals(
                    targets,
                    anchors,
                    extra_anchor_embeddings=extra_anchors,
                    rank=2,
                )
            )

        expected = build_global_pairwise_residual_matrix(targets, extra_anchors)
        expected = expected / expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(captured["matrix"], expected)
        self.assertFalse(diagnostics["subspace_input_projected"])
        self.assertEqual(
            diagnostics["target_global_subspace_target_count"],
            3,
        )
        self.assertEqual(
            diagnostics["target_global_subspace_residual_count"],
            3 * (3 + 1),
        )
        self.assertEqual(
            diagnostics["target_global_subspace_residual_shape"],
            (12, 2),
        )

    def test_preserves_negative_target_projection_and_legacy_norm(self):
        targets = torch.tensor(
            [[[3.0, 4.0, 0.0]], [[2.0, 1.0, 0.0]]]
        )
        legacy = torch.tensor(
            [[[0.0, 2.0, 0.0]], [[0.0, 3.0, 0.0]]]
        )
        extra_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]
        )

        residuals, diagnostics = (
            build_target_global_pairwise_residual_subspace_residuals(
                targets,
                targets + legacy,
                extra_anchor_embeddings=extra_anchors,
                rank=1,
            )
        )

        torch.testing.assert_close(
            residuals.norm(dim=2),
            legacy.norm(dim=2),
        )
        self.assertLess(diagnostics["subspace_max_projection_error"], 1e-6)
        self.assertEqual(diagnostics["subspace_basis_rank"], 1)

    def test_mean_norm_mode_uses_each_targets_mean_outgoing_residual_norm(self):
        targets = torch.tensor(
            [[[1.0, 1.0]], [[4.0, 1.0]]]
        )
        legacy_anchors = targets + torch.tensor(
            [[[7.0, 0.0]], [[0.0, 8.0]]]
        )
        extra_anchors = torch.tensor(
            [[[1.0, 5.0]], [[5.0, 4.0]]]
        )

        residuals, diagnostics = (
            build_mean_norm_target_global_pairwise_residual_subspace_residuals(
                targets,
                legacy_anchors,
                extra_anchor_embeddings=extra_anchors,
                rank=2,
            )
        )

        expected_norms = torch.tensor([
            (3.0 + 4.0 + 5.0) / 3.0,
            (3.0 + 5.0 + torch.sqrt(torch.tensor(10.0))) / 3.0,
        ])
        torch.testing.assert_close(residuals.norm(dim=2), expected_norms[:, None])
        torch.testing.assert_close(
            torch.tensor(diagnostics["target_global_mean_residual_norms"]),
            expected_norms,
        )
        self.assertEqual(diagnostics["subspace_magnitude_mode"], "source_mean")
        self.assertLess(diagnostics["subspace_max_norm_error"], 1e-6)

    def test_mean_norm_mode_supports_target_only_pairwise_residuals(self):
        targets = torch.tensor([[[1.0, 1.0]], [[4.0, 1.0]]])
        legacy_anchors = targets + 1.0
        extra_anchors = torch.empty(0, 1, 2)

        residuals, diagnostics = (
            build_mean_norm_target_global_pairwise_residual_subspace_residuals(
                targets,
                legacy_anchors,
                extra_anchor_embeddings=extra_anchors,
                rank=1,
            )
        )

        torch.testing.assert_close(
            residuals.norm(dim=2),
            torch.tensor([[3.0], [3.0]]),
        )
        self.assertEqual(diagnostics["global_subspace_extra_anchor_count"], 0)
        self.assertEqual(diagnostics["global_subspace_residual_count"], 2)


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


class NormalizedResidualSubspaceTests(unittest.TestCase):
    def test_normalizes_residuals_before_svd_and_uses_requested_rank(self):
        targets = torch.tensor(
            [[[3.0, 4.0, 0.0]], [[0.0, 0.0, 5.0]], [[1.0, 0.0, 0.0]]]
        )
        legacy = torch.tensor(
            [[[10.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]], [[0.0, 0.0, 1.0]]]
        )

        residuals, diagnostics = build_negative_target_normalized_residual_subspace_residuals(
            targets,
            targets + legacy,
            rank=2,
        )

        self.assertEqual(diagnostics["subspace_requested_rank"], 2)
        self.assertEqual(diagnostics["subspace_effective_rank"], 3)
        self.assertTrue(diagnostics["subspace_normalized_before_svd"])
        torch.testing.assert_close(
            residuals.flatten(1).norm(dim=1), legacy.flatten(1).norm(dim=1)
        )


class LargestAnchorCosineSubspaceResidualTests(unittest.TestCase):
    def test_uses_positive_anchor_projection_and_preserves_legacy_norm(self):
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
        anchors = targets + legacy

        residuals, diagnostics = (
            build_largest_anchor_cosine_subspace_residuals(
                targets,
                anchors,
                top_k=2,
            )
        )

        expected = torch.tensor(
            [
                [[10.0 / torch.sqrt(torch.tensor(41.0)), 8.0 / torch.sqrt(torch.tensor(41.0)), 0.0]],
                [[3.0 / torch.sqrt(torch.tensor(2.0)), 3.0 / torch.sqrt(torch.tensor(2.0)), 0.0]],
            ]
        )
        torch.testing.assert_close(residuals, expected)
        torch.testing.assert_close(
            residuals.flatten(1).norm(dim=1),
            legacy.flatten(1).norm(dim=1),
        )
        self.assertLess(diagnostics["subspace_max_projection_error"], 1e-6)
        self.assertEqual(diagnostics["subspace_anchor_projection_fallback_count"], 0)

    def test_falls_back_to_projected_legacy_when_anchor_projection_is_zero(self):
        target = torch.tensor([[[-2.0, 0.0, 3.0]]])
        legacy = torch.tensor([[[2.0, 0.0, 0.0]]])
        anchor = target + legacy

        residuals, diagnostics = (
            build_largest_anchor_cosine_subspace_residuals(
                target,
                anchor,
                top_k=1,
            )
        )

        torch.testing.assert_close(residuals, legacy)
        self.assertEqual(diagnostics["subspace_anchor_projection_fallback_count"], 1)
        self.assertEqual(diagnostics["subspace_legacy_fallback_count"], 1)
        self.assertEqual(diagnostics["subspace_basis_fallback_count"], 0)


if __name__ == "__main__":
    unittest.main()
