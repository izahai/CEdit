import unittest

import torch

from src.rank_analysis import (
    aggregate_layer_metrics,
    build_tgprs_basis,
    edit_statistic,
    layer_edit_metrics,
    low_rank_delta_spectrum,
    projected_target_residuals,
    random_negative_target_residuals,
    random_rank_residuals,
    retain_eigensystem,
    retain_projection_from_eigensystem,
    spectral_metrics,
    speed_delta_weight,
    speed_delta_weight_from_target_factor,
    speed_delta_weight_low_rank,
    speed_retain_construction,
    speed_right_factor,
    tgprs_residuals_from_basis,
    truncated_svd_residuals,
)


class SpectralMetricsTests(unittest.TestCase):
    def test_reports_known_rank_and_energy_ranks(self):
        matrix = torch.diag(torch.tensor([4.0, 2.0, 0.0]))
        metrics, singular_values = spectral_metrics(matrix, rtol=1e-6)

        self.assertEqual(metrics["numerical_rank"], 2)
        self.assertAlmostEqual(metrics["stable_rank"], 1.25, places=6)
        self.assertGreater(metrics["effective_rank"], 1.0)
        self.assertLess(metrics["effective_rank"], 2.0)
        torch.testing.assert_close(
            singular_values, torch.tensor([4.0, 2.0, 0.0])
        )

    def test_zero_matrix_has_zero_rank_metrics(self):
        metrics, _ = spectral_metrics(torch.zeros(3, 5))
        self.assertEqual(metrics["numerical_rank"], 0)
        self.assertEqual(metrics["stable_rank"], 0.0)
        self.assertEqual(metrics["effective_rank"], 0.0)


class ResidualControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.legacy = torch.randn(8, 12)

    def assert_norm_matched(self, actual):
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual, dim=1),
            torch.linalg.vector_norm(self.legacy, dim=1),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_truncated_svd_is_rank_bounded_and_norm_matched(self):
        residuals = truncated_svd_residuals(self.legacy, rank=3)
        self.assertLessEqual(int(torch.linalg.matrix_rank(residuals)), 3)
        self.assert_norm_matched(residuals)

    def test_random_subspace_is_seeded_rank_bounded_and_norm_matched(self):
        first = random_rank_residuals(self.legacy, rank=4, seed=11)
        second = random_rank_residuals(self.legacy, rank=4, seed=11)
        different = random_rank_residuals(self.legacy, rank=4, seed=12)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.allclose(first, different))
        self.assertLessEqual(int(torch.linalg.matrix_rank(first)), 4)
        self.assert_norm_matched(first)

    def test_tgprs_uses_shared_basis_and_legacy_norms(self):
        targets = torch.randn(6, 12)
        anchors = torch.randn(2, 12)
        common_anchor = torch.randn(1, 12)
        legacy = common_anchor - targets
        basis, _, diagnostics = build_tgprs_basis(
            targets, anchors, max_rank=4
        )
        residuals, direction_diagnostics = tgprs_residuals_from_basis(
            targets, legacy, basis, rank=3
        )

        self.assertEqual(diagnostics["pairwise_residual_count"], 42)
        self.assertEqual(direction_diagnostics["requested_rank"], 3)
        self.assertLessEqual(int(torch.linalg.matrix_rank(residuals)), 3)
        torch.testing.assert_close(
            residuals,
            (residuals @ basis[:3].T) @ basis[:3],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_signed_projection_and_complement_are_orthogonal(self):
        targets = torch.randn(6, 12)
        basis = torch.linalg.qr(torch.randn(12, 4), mode="reduced").Q.T
        negative, _ = projected_target_residuals(
            targets, self.legacy[:6], basis, rank=4, sign=-1.0
        )
        positive, _ = projected_target_residuals(
            targets, self.legacy[:6], basis, rank=4, sign=1.0
        )
        complement, _ = projected_target_residuals(
            targets,
            self.legacy[:6],
            basis,
            rank=4,
            sign=-1.0,
            complement=True,
        )
        torch.testing.assert_close(negative, -positive)
        self.assertLessEqual(int(torch.linalg.matrix_rank(negative)), 4)
        torch.testing.assert_close(
            complement @ basis.T,
            torch.zeros(6, 4),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_random_negative_target_is_seeded_and_rank_bounded(self):
        targets = torch.randn(8, 12)
        first, _ = random_negative_target_residuals(
            targets, self.legacy, rank=3, seed=19
        )
        second, _ = random_negative_target_residuals(
            targets, self.legacy, rank=3, seed=19
        )
        torch.testing.assert_close(first, second)
        self.assertLessEqual(int(torch.linalg.matrix_rank(first)), 3)
        self.assert_norm_matched(first)
        torch.testing.assert_close(
            torch.linalg.vector_norm(residuals, dim=1),
            torch.linalg.vector_norm(legacy, dim=1),
            rtol=1e-5,
            atol=1e-5,
        )


class SpeedAnalysisTests(unittest.TestCase):
    def test_edit_statistic_matches_outer_product_mean(self):
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        residuals = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        statistic = edit_statistic(residuals, targets)
        torch.testing.assert_close(statistic, torch.diag(torch.tensor([1.0, 2.0])))

    def test_speed_factor_and_delta_have_expected_shapes(self):
        target_covariance = torch.eye(4)
        retain_projection = torch.eye(4)
        k2 = torch.eye(4)[:, :2]
        factor = speed_right_factor(
            target_covariance,
            retain_projection,
            k2,
            retain_scale=0.5,
            lamb=0.1,
        )
        delta = speed_delta_weight(torch.randn(3, 4), torch.eye(4), factor)
        self.assertEqual(factor.shape, (4, 4))
        self.assertEqual(delta.shape, (3, 4))
        self.assertTrue(torch.isfinite(delta).all())

    def test_zero_delta_has_zero_effect_and_leakage(self):
        weight = torch.eye(3)
        targets = torch.eye(3)
        residuals = -torch.eye(3)
        retain = torch.tensor([[1.0, 1.0, 1.0]])
        metrics = layer_edit_metrics(
            weight, torch.zeros_like(weight), targets, residuals, retain
        )
        self.assertAlmostEqual(metrics["target_effect"], 0.0)
        self.assertAlmostEqual(metrics["target_rotation_deg"], 0.0)
        self.assertAlmostEqual(metrics["retain_leakage"], 0.0)

    def test_low_rank_delta_spectrum_matches_full_svd(self):
        torch.manual_seed(23)
        weight = torch.randn(7, 9)
        targets = torch.randn(5, 9)
        residuals = torch.randn(5, 9)
        factor = torch.randn(9, 9)
        statistic = edit_statistic(residuals, targets)
        delta = speed_delta_weight(weight, statistic, factor)
        actual = low_rank_delta_spectrum(weight, residuals, targets, factor)
        expected = torch.linalg.svdvals(delta)[:actual.numel()]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            speed_delta_weight_low_rank(weight, residuals, targets, factor),
            delta,
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            speed_delta_weight_from_target_factor(
                weight, residuals, targets @ factor / targets.shape[0]
            ),
            delta,
            rtol=1e-4,
            atol=1e-5,
        )
        residual_rank = int(torch.linalg.matrix_rank(residuals))
        statistic_rank = int(torch.linalg.matrix_rank(statistic))
        delta_rank = int(torch.linalg.matrix_rank(delta))
        self.assertLessEqual(statistic_rank, residual_rank)
        self.assertLessEqual(delta_rank, statistic_rank)

    def test_retain_projector_rank_is_monotonic(self):
        retain = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.1]))
        u, singular_values = retain_eigensystem(retain)
        low, low_rank = retain_projection_from_eigensystem(
            u, singular_values, 0.5
        )
        high, high_rank = retain_projection_from_eigensystem(
            u, singular_values, 5.0
        )
        self.assertLessEqual(low_rank, high_rank)
        torch.testing.assert_close(low, low.T)
        torch.testing.assert_close(high, high.T)

    def test_speed_retain_construction_is_deterministic(self):
        torch.manual_seed(29)
        retain = torch.randn(10, 6)
        weight = torch.randn(4, 6)
        statistic = torch.randn(6, 6)
        covariance = torch.eye(6)
        first, first_info = speed_retain_construction(
            retain, weight, statistic, covariance, aug_num=2, seed=31
        )
        second, second_info = speed_retain_construction(
            retain, weight, statistic, covariance, aug_num=2, seed=31
        )
        torch.testing.assert_close(first, second)
        self.assertEqual(first_info, second_info)
        self.assertEqual(first.shape[0], first_info["retain_constructed_count"])

    def test_zero_augmentation_without_filter_preserves_retain_set(self):
        retain = torch.randn(10, 6)
        constructed, info = speed_retain_construction(
            retain,
            torch.randn(4, 6),
            torch.randn(6, 6),
            torch.eye(6),
            aug_num=0,
            filter_enabled=False,
        )
        torch.testing.assert_close(constructed, retain)
        self.assertEqual(info["retain_augmented_count"], 0)

    def test_aggregate_layer_metrics(self):
        names = (
            "target_effect",
            "target_rotation_deg",
            "retain_leakage",
            "directional_realization_cosine",
            "directional_relative_error",
            "delta_frobenius_norm",
        )
        rows = [
            {name: 1.0 for name in names},
            {name: 3.0 for name in names},
        ]
        aggregate = aggregate_layer_metrics(rows)
        for name in names:
            self.assertEqual(aggregate[f"{name}_mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
