import sys
import types
import unittest

import torch


import importlib.machinery


def _install_optional_dependency_stubs():
    for name in ("pandas", "tqdm", "kmeans_pytorch", "diffusers"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
            if name == "tqdm":
                stub.tqdm = lambda values, **_: values
            elif name == "kmeans_pytorch":
                stub.kmeans = None
            elif name == "diffusers":
                stub.StableDiffusionPipeline = object
            sys.modules[name] = stub


_install_optional_dependency_stubs()

from train_erase_null import build_target_anchor_statistics


class TargetAnchorStatisticsTests(unittest.TestCase):
    def test_norm_matched_truncated_svd_caps_edit_statistic_rank(self):
        targets = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        ]
        legacy = [
            torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 2.0, 3.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 3.0, 4.0]]),
            torch.tensor([[5.0, 0.0, 0.0, 4.0]]),
        ]
        anchors = [target + residual for target, residual in zip(targets, legacy)]

        _, target_anchor_delta, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="norm_matched_truncated_svd_residual",
            residual_rank=2,
        )

        self.assertLessEqual(diagnostics["residual_rank"], 2)
        self.assertLessEqual(diagnostics["edit_statistic_rank"], 2)
        self.assertLessEqual(torch.linalg.matrix_rank(target_anchor_delta), 2)
        self.assertTrue(diagnostics["truncated_svd_norm_matched"])

    def test_mean_norm_target_global_mode_reports_per_target_magnitudes(self):
        targets = [
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[4.0, 1.0]]),
        ]
        anchors = [target + 1.0 for target in targets]
        fixed_anchors = torch.tensor(
            [[[1.0, 5.0]], [[5.0, 4.0]]]
        )

        _, _, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="mean_norm_target_global_pairwise_residual_subspace",
            residual_rank=2,
            subspace_anchor_embeddings=fixed_anchors,
        )

        self.assertEqual(diagnostics["subspace_magnitude_mode"], "source_mean")
        self.assertEqual(len(diagnostics["target_global_mean_residual_norms"]), 2)

    def test_retain_aware_mode_uses_projected_target_global_basis(self):
        targets = [
            torch.tensor([[1.0, 0.0, 1.0]]),
            torch.tensor([[0.0, 2.0, 2.0]]),
        ]
        anchors = [target + 1.0 for target in targets]
        fixed_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[2.0, 1.0, 3.0]]]
        )
        retain_projection = torch.diag(torch.tensor([1.0, 1.0, 0.0]))

        _, target_anchor_delta, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="retain_aware_target_global_pairwise_residual_subspace",
            residual_rank=2,
            subspace_anchor_embeddings=fixed_anchors,
            retain_projection=retain_projection,
        )

        torch.testing.assert_close(
            target_anchor_delta[2],
            torch.zeros_like(target_anchor_delta[2]),
        )
        self.assertTrue(diagnostics["retain_aware_subspace"])

    def test_target_global_mode_builds_the_basis_from_targets(self):
        targets = [
            torch.tensor([[3.0, 4.0, 0.0]]),
            torch.tensor([[2.0, 1.0, 0.0]]),
        ]
        legacy = [
            torch.tensor([[0.0, 2.0, 0.0]]),
            torch.tensor([[0.0, 3.0, 0.0]]),
        ]
        anchors = [
            target + residual
            for target, residual in zip(targets, legacy)
        ]
        fixed_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]
        )

        _, _, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="target_global_pairwise_residual_subspace",
            residual_rank=1,
            subspace_anchor_embeddings=fixed_anchors,
        )

        self.assertEqual(diagnostics["target_global_subspace_target_count"], 2)
        self.assertEqual(diagnostics["target_global_subspace_residual_count"], 6)
        self.assertEqual(diagnostics["target_global_subspace_residual_shape"], (6, 3))

    def test_target_global_mode_propagates_positive_projection_direction(self):
        targets = [
            torch.tensor([[3.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
        ]
        anchors = [
            targets[0] + torch.tensor([[0.0, 2.0, 0.0]]),
            targets[1] + torch.tensor([[0.0, 0.0, 3.0]]),
        ]
        fixed_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]
        )

        _, negative_delta, _ = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="target_global_pairwise_residual_subspace",
            residual_rank=1,
            subspace_anchor_embeddings=fixed_anchors,
        )
        _, positive_delta, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="target_global_pairwise_residual_subspace",
            residual_rank=1,
            subspace_anchor_embeddings=fixed_anchors,
            target_projection_direction="positive",
        )

        torch.testing.assert_close(positive_delta, -negative_delta)
        self.assertEqual(
            diagnostics["subspace_target_projection_direction"],
            "positive",
        )

    def test_global_pairwise_mode_uses_external_concept_embeddings(self):
        targets = [torch.tensor([[3.0, 4.0, 0.0]])]
        anchors = [torch.tensor([[3.0, 6.0, 0.0]])]
        global_concepts = torch.tensor(
            [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]
        )
        global_anchors = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]]
        )

        _, target_anchor_delta, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="global_pairwise_residual_subspace",
            residual_rank=1,
            subspace_concept_embeddings=global_concepts,
            subspace_anchor_embeddings=global_anchors,
        )

        expected_residual = torch.tensor([[-2.0, 0.0, 0.0]])
        torch.testing.assert_close(
            target_anchor_delta,
            expected_residual.T @ targets[0],
        )
        self.assertEqual(diagnostics["global_subspace_concept_count"], 2)
        self.assertEqual(diagnostics["global_subspace_residual_count"], 6)

    def test_legacy_statistics_remain_the_direct_residual_formula(self):
        targets = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 1.0]]),
        ]
        anchors = [
            torch.tensor([[2.0, 2.0]]),
            torch.tensor([[3.0, 5.0]]),
        ]

        sum_target_target, target_anchor_delta, diagnostics = (
            build_target_anchor_statistics(
                targets,
                anchors,
                anchor_mode="legacy",
            )
        )

        stacked_targets = torch.stack(targets)
        stacked_residuals = torch.stack(anchors) - stacked_targets
        expected_sum = torch.stack(
            [target.T @ target for target in stacked_targets]
        ).mean(0)
        expected_delta = torch.stack(
            [
                residual.T @ target
                for residual, target in zip(stacked_residuals, stacked_targets)
            ]
        ).mean(0)
        torch.testing.assert_close(sum_target_target, expected_sum)
        torch.testing.assert_close(target_anchor_delta, expected_delta)
        self.assertEqual(diagnostics["residual_scale"], 1.0)

    def test_residual_scale_scales_new_mode_edit_statistic(self):
        targets = [
            torch.tensor([[3.0, 4.0, 1.0]]),
            torch.tensor([[2.0, -1.0, 2.0]]),
        ]
        legacy = [
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 3.0, 0.0]]),
        ]
        anchors = [target + residual for target, residual in zip(targets, legacy)]

        _, unscaled_delta, _ = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="smallest_cosine_subspace",
            residual_top_k=2,
        )
        _, scaled_delta, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="smallest_cosine_subspace",
            residual_top_k=2,
            residual_scale=2.5,
        )

        torch.testing.assert_close(scaled_delta, 2.5 * unscaled_delta)
        self.assertEqual(diagnostics["residual_scale"], 2.5)
        self.assertEqual(diagnostics["subspace_requested_top_k"], 2)

    def test_largest_anchor_mode_exposes_anchor_cosine_diagnostics(self):
        targets = [
            torch.tensor([[3.0, 4.0, 1.0]]),
            torch.tensor([[2.0, -1.0, 2.0]]),
        ]
        legacy = [
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 3.0, 0.0]]),
        ]
        anchors = [target + residual for target, residual in zip(targets, legacy)]

        _, _, diagnostics = build_target_anchor_statistics(
            targets,
            anchors,
            anchor_mode="largest_anchor_cosine_subspace",
            residual_top_k=2,
        )

        self.assertEqual(diagnostics["subspace_requested_top_k"], 2)
        self.assertIn("subspace_new_anchor_cosine_mean", diagnostics)
        self.assertEqual(diagnostics["subspace_anchor_projection_fallback_count"], 0)

    def test_zero_anchor_produces_negative_target_outer_product(self):
        targets = [
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[4.0, -1.0, 2.0]]),
        ]
        zero_anchors = [torch.zeros_like(t) for t in targets]

        sum_target_target, target_anchor_delta, diagnostics = (
            build_target_anchor_statistics(
                targets,
                zero_anchors,
                anchor_mode="legacy",
            )
        )

        torch.testing.assert_close(target_anchor_delta, -sum_target_target)
        self.assertEqual(diagnostics["residual_rank"], 2)


if __name__ == "__main__":
    unittest.main()
