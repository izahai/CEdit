import sys
import types
import unittest

import torch


def _install_optional_dependency_stubs():
    pandas = types.ModuleType("pandas")
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda values, **_: values
    kmeans_module = types.ModuleType("kmeans_pytorch")
    kmeans_module.kmeans = None
    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusionPipeline = object
    sys.modules.setdefault("pandas", pandas)
    sys.modules.setdefault("tqdm", tqdm_module)
    sys.modules.setdefault("kmeans_pytorch", kmeans_module)
    sys.modules.setdefault("diffusers", diffusers)


_install_optional_dependency_stubs()

from train_erase_null import build_target_anchor_statistics


class TargetAnchorStatisticsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
