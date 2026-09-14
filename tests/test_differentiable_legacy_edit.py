import unittest

import torch

from src.differentiable_legacy_edit import (
    DifferentiableLegacyEditConfig,
    build_legacy_target_anchor_statistics,
    build_retain_projector,
    prepare_differentiable_legacy_edit,
    select_value_parameter_names,
)
try:
    from train_erase_null import build_target_anchor_statistics
except ModuleNotFoundError:
    build_target_anchor_statistics = None


class _Attention(torch.nn.Module):
    def __init__(self, dimension, output_dimension=None):
        super().__init__()
        self.to_v = torch.nn.Linear(
            dimension,
            output_dimension or dimension,
            bias=False,
        )


class _Block(torch.nn.Module):
    def __init__(self, dimension, output_dimension=None):
        super().__init__()
        self.attn2 = _Attention(dimension, output_dimension)


class _ToyUNet(torch.nn.Module):
    def __init__(self, dimension=4, output_dimension=None):
        super().__init__()
        self.block = _Block(dimension, output_dimension)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
    ):
        output = self.block.attn2.to_v(sample)
        return (output,)


def _prepared_state(dtype=torch.float64):
    dimension = 6
    torch.manual_seed(7)
    unet = _ToyUNet(dimension).to(dtype=dtype)
    targets = torch.randn(2, 1, dimension, dtype=dtype)
    retains = torch.zeros(5, 1, dimension, dtype=dtype)
    null_hidden = torch.zeros(5, dimension, dtype=dtype)
    k2 = torch.eye(dimension, dtype=dtype)[:, :4]
    config = DifferentiableLegacyEditConfig(
        residual_scale=0.7,
        retain_scale=1.3,
        threshold=0.1,
        lamb=0.0,
        chunk_size=2,
    )
    state = prepare_differentiable_legacy_edit(
        unet,
        targets,
        retains,
        null_hidden,
        config,
        prepared_k2=k2,
    )
    return unet, targets, state


class DifferentiableLegacyEditTests(unittest.TestCase):
    def test_fixed_retain_projection_rank_overrides_threshold(self):
        covariance = torch.diag(
            torch.tensor([5.0, 3.0, 1.0, 0.1], dtype=torch.float64)
        )

        threshold_projector, _ = build_retain_projector(covariance, threshold=2.0)
        fixed_projector, _ = build_retain_projector(
            covariance,
            threshold=10.0,
            retain_projection_rank=1,
        )

        torch.testing.assert_close(
            threshold_projector,
            torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)),
        )
        torch.testing.assert_close(
            fixed_projector,
            torch.diag(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)),
        )

    def test_zero_retain_projection_rank_builds_zero_projector(self):
        covariance = torch.eye(4, dtype=torch.float64)
        projector, _ = build_retain_projector(
            covariance,
            threshold=2.0,
            retain_projection_rank=0,
        )
        torch.testing.assert_close(projector, torch.zeros_like(covariance))

    def test_retain_projection_rank_rejects_invalid_values(self):
        covariance = torch.eye(4, dtype=torch.float64)
        for rank in (-1, True):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    build_retain_projector(
                        covariance,
                        threshold=0.1,
                        retain_projection_rank=rank,
                    )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_retain_projector(
                covariance,
                threshold=0.1,
                retain_projection_rank=5,
            )

    def test_fixed_rank_is_reported_by_diagnostics_and_geometry(self):
        dimension = 4
        torch.manual_seed(17)
        unet = _ToyUNet(dimension).double()
        targets = torch.randn(1, 1, dimension, dtype=torch.float64)
        retains = torch.randn(8, 1, dimension, dtype=torch.float64)
        config = DifferentiableLegacyEditConfig(
            retain_projection_rank=2,
            threshold=-1.0,
            use_k2=False,
        )
        state = prepare_differentiable_legacy_edit(
            unet,
            targets,
            retains,
            config=config,
        )

        _, diagnostics = state.effective_parameters(targets + 0.1)
        self.assertEqual(state.projector_rank, 2)
        self.assertEqual(diagnostics["projector_rank"], 2)
        self.assertEqual(state.geometry_metadata()["projector_rank"], 2)

    def test_statistics_match_existing_legacy_helper(self):
        if build_target_anchor_statistics is None:
            self.skipTest("train_erase_null dependencies (kmeans_pytorch) not installed")
        torch.manual_seed(2)
        targets = torch.randn(3, 1, 5, dtype=torch.float64)
        anchors = torch.randn(3, 1, 5, dtype=torch.float64)

        expected_c, expected_d, _ = build_target_anchor_statistics(
            list(targets),
            list(anchors),
            anchor_mode="legacy",
            residual_scale=0.6,
        )
        actual_c, actual_d = build_legacy_target_anchor_statistics(
            targets,
            anchors,
            residual_scale=0.6,
        )

        torch.testing.assert_close(actual_c, expected_c, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(actual_d, expected_d, rtol=1e-8, atol=1e-10)

    def test_effective_weight_preserves_dense_legacy_multiplication_order(self):
        unet, targets, state = _prepared_state()
        anchors = targets + torch.randn_like(targets) * 0.2

        overrides, diagnostics = state.effective_parameters(anchors)
        _, target_anchor_delta = build_legacy_target_anchor_statistics(
            targets,
            anchors,
            state.config.residual_scale,
        )
        identity = torch.eye(targets.shape[-1], dtype=targets.dtype)
        base_weight = dict(unet.named_parameters())["block.attn2.to_v.weight"]
        expected_delta = (
            base_weight
            @ target_anchor_delta
            @ state.retain_projector
            @ (
                identity
                - state.matrix_m
                @ state.k2
                @ state.inner_inverse
                @ state.k2.T
                @ state.retain_projector
            )
            @ state.matrix_m
        )

        torch.testing.assert_close(
            overrides["block.attn2.to_v.weight"],
            base_weight + expected_delta,
            rtol=1e-8,
            atol=1e-10,
        )
        self.assertEqual(diagnostics["projector_rank"], 6)

    def test_identity_anchor_produces_zero_update(self):
        unet, targets, state = _prepared_state()
        overrides, _ = state.effective_parameters(targets)
        torch.testing.assert_close(
            overrides["block.attn2.to_v.weight"],
            dict(unet.named_parameters())["block.attn2.to_v.weight"],
            rtol=0,
            atol=0,
        )

    def test_effective_weight_without_k2(self):
        dimension = 6
        torch.manual_seed(7)
        unet = _ToyUNet(dimension).to(dtype=torch.float64)
        targets = torch.randn(2, 1, dimension, dtype=torch.float64)
        retains = torch.zeros(5, 1, dimension, dtype=torch.float64)
        config = DifferentiableLegacyEditConfig(
            residual_scale=0.7,
            retain_scale=1.3,
            threshold=0.1,
            lamb=0.0,
            chunk_size=2,
            use_k2=False,
        )
        state = prepare_differentiable_legacy_edit(
            unet,
            targets,
            retains,
            null_hidden_states=None,
            config=config,
        )
        self.assertIsNone(state.k2)
        self.assertIsNone(state.inner_inverse)
        self.assertFalse(state.geometry_metadata()["use_k2"])
        self.assertIsNone(state.geometry_metadata()["k2_sha256"])

        anchors = targets + torch.randn_like(targets) * 0.2
        overrides, diagnostics = state.effective_parameters(anchors)
        _, target_anchor_delta = build_legacy_target_anchor_statistics(
            targets,
            anchors,
            state.config.residual_scale,
        )
        base_weight = dict(unet.named_parameters())["block.attn2.to_v.weight"]
        expected_delta = (
            base_weight
            @ target_anchor_delta
            @ state.retain_projector
            @ state.matrix_m
        )
        torch.testing.assert_close(
            overrides["block.attn2.to_v.weight"],
            base_weight + expected_delta,
            rtol=1e-8,
            atol=1e-10,
        )

    def test_dense_equation_passes_gradcheck(self):
        _, targets, state = _prepared_state()
        anchors = (targets + 0.1 * torch.randn_like(targets)).requires_grad_(True)

        def edited_weight(value):
            return state.effective_parameters(value)[0]["block.attn2.to_v.weight"]

        self.assertTrue(
            torch.autograd.gradcheck(
                edited_weight,
                (anchors,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_functional_call_reaches_anchor_without_mutating_base(self):
        unet, targets, state = _prepared_state()
        self.assertTrue(unet.training)
        self.assertTrue(
            dict(unet.named_parameters())["block.attn2.to_v.weight"].requires_grad
        )
        anchors = (targets + 0.1 * torch.randn_like(targets)).requires_grad_(True)
        before = {
            name: value.detach().clone() for name, value in unet.named_parameters()
        }
        overrides, _ = state.effective_parameters(anchors)
        sample = torch.randn(3, 6, dtype=torch.float64)
        output = torch.func.functional_call(
            unet,
            overrides,
            args=(sample, torch.tensor(1)),
            kwargs={"return_dict": False},
            strict=False,
        )[0]
        output.square().sum().backward()

        self.assertIsNotNone(anchors.grad)
        self.assertTrue(torch.isfinite(anchors.grad).all())
        self.assertGreater(anchors.grad.abs().sum().item(), 0)
        self.assertTrue(
            torch.all(torch.linalg.vector_norm(anchors.grad.flatten(1), dim=1) > 0)
        )
        for name, parameter in unet.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_failed_functional_call_leaves_base_weights_untouched(self):
        class FailingUNet(_ToyUNet):
            def forward(
                self,
                sample,
                timestep,
                encoder_hidden_states=None,
                return_dict=False,
            ):
                self.block.attn2.to_v(sample)
                raise RuntimeError("deliberate forward failure")

        _, targets, state = _prepared_state()
        failing_unet = FailingUNet(6).double()
        with torch.no_grad():
            failing_unet.block.attn2.to_v.weight.copy_(
                state.base_weights["block.attn2.to_v.weight"]
            )
        original = failing_unet.block.attn2.to_v.weight.detach().clone()
        overrides, _ = state.effective_parameters(targets + 0.1)
        with self.assertRaisesRegex(RuntimeError, "deliberate forward failure"):
            torch.func.functional_call(
                failing_unet,
                overrides,
                args=(torch.randn(1, 6, dtype=torch.float64), torch.tensor(1)),
                strict=False,
            )
        torch.testing.assert_close(
            failing_unet.block.attn2.to_v.weight,
            original,
            rtol=0,
            atol=0,
        )

    def test_parameter_selection_is_exact_and_rejects_empty_match(self):
        unet = _ToyUNet()
        self.assertEqual(
            select_value_parameter_names(unet),
            ["block.attn2.to_v.weight"],
        )
        with self.assertRaisesRegex(ValueError, "No attn2"):
            select_value_parameter_names(torch.nn.Linear(4, 4))

    def test_shape_nonfinite_and_singular_inputs_fail_explicitly(self):
        _, targets, state = _prepared_state()
        with self.assertRaisesRegex(ValueError, "shape"):
            state.effective_parameters(torch.zeros(1, 1, 4, dtype=torch.float64))
        invalid = targets.clone()
        invalid[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            state.effective_parameters(invalid)

        retains = torch.zeros(3, 1, 6, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "singular"):
            prepare_differentiable_legacy_edit(
                _ToyUNet(6).double(),
                targets,
                retains,
                torch.zeros(5, 6, dtype=torch.float64),
                DifferentiableLegacyEditConfig(),
                prepared_k2=torch.zeros(6, 4, dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
