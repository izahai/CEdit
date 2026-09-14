import unittest
from unittest import mock

import torch

from src.closed_form_anchor_training import (
    AnchorTrainingConfig,
    BoundedAnchor,
    DiffusionState,
    optimize_anchor,
    paired_predictions,
    prediction_objective,
    predicted_noise_cosine,
    sample_prefix_diffusion_state,
    state_prediction_metrics,
)


class _Attention(torch.nn.Module):
    def __init__(self, dimension=3, output_dimension=2):
        super().__init__()
        self.to_v = torch.nn.Linear(dimension, output_dimension, bias=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = _Attention()


class _ToyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.calls = []

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
    ):
        self.calls.append((sample, timestep, encoder_hidden_states))
        return (self.block.attn2.to_v(sample),)


class _NoOpEditState:
    def __init__(self, unet):
        self.base_weight = dict(unet.named_parameters())[
            "block.attn2.to_v.weight"
        ].detach().clone()

    def effective_parameters(self, anchors):
        graph_zero = anchors.sum() * 0.0
        return {
            "block.attn2.to_v.weight": self.base_weight + graph_zero
        }, {}


class _Scheduler:
    def __init__(self):
        self.config = {"name": "fake"}
        self.timesteps = None

    @classmethod
    def from_config(cls, config):
        return cls()

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.arange(steps, 0, -1, device=device)

    def scale_model_input(self, sample, timestep):
        return sample * (1.0 + timestep.float() / 10.0)

    def step(self, prediction, timestep, latents, return_dict=False, **kwargs):
        return (latents - prediction * 0.01,)


class _PrefixUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"in_channels": 4})()
        self.calls = 0

    @property
    def device(self):
        return torch.device("cpu")

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
    ):
        self.calls += 1
        conditioning = encoder_hidden_states.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return (sample * 0.1 + conditioning,)


class _FakePipe:
    def __init__(self):
        self.unet = _PrefixUNet()
        self.scheduler = _Scheduler()

    def prepare_latents(
        self,
        batch_size,
        channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        return torch.randn(
            batch_size,
            channels,
            height // 8,
            width // 8,
            dtype=dtype,
            device=device,
            generator=generator,
        )

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}


class ClosedFormAnchorTrainingTests(unittest.TestCase):
    def test_bounded_anchor_trains_norm_and_direction_under_target_norm(self):
        targets = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
        initial = torch.zeros_like(targets)
        initial[1, 0, 0] = 10.0
        model = BoundedAnchor(targets, initial, seed=9)

        anchors = model()
        residuals = model.residuals()
        norms = torch.linalg.vector_norm(anchors.flatten(1), dim=1)
        self.assertTrue(torch.all(norms > 0))
        self.assertTrue(torch.all(norms < torch.tensor([1.0, 2.0])))
        torch.testing.assert_close(residuals, anchors - targets)
        torch.testing.assert_close(model.max_anchor_norm.flatten(), torch.tensor([1.0, 2.0]))
        self.assertEqual(model.initial_was_zero.tolist(), [True, False])
        self.assertEqual(model.initial_was_clipped.tolist(), [False, True])

        model().square().sum().backward()
        self.assertTrue(torch.isfinite(model.raw_direction.grad).all())
        self.assertTrue(torch.isfinite(model.raw_magnitude.grad).all())
        self.assertGreater(model.raw_magnitude.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        model()[1, 0, 1].backward()
        self.assertGreater(model.raw_direction.grad.abs().sum().item(), 0)

    def test_subcap_initial_anchor_is_preserved_and_zero_target_fails(self):
        target = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float64)
        initial = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = BoundedAnchor(target, initial)
        torch.testing.assert_close(model(), initial, rtol=1e-12, atol=1e-12)
        self.assertFalse(model.initial_was_clipped.item())
        self.assertEqual(model.diagnostics()["anchor_norms"], [1.0])
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            BoundedAnchor(torch.zeros_like(target), initial)

    def test_anchor_initialization_clips_norm_but_preserves_direction(self):
        target = torch.tensor([[[2.0, 0.0, 0.0]]])
        empty_anchor = torch.tensor([[[0.0, 4.0, 0.0]]])
        model = BoundedAnchor(target, empty_anchor)
        self.assertTrue(model.initial_was_clipped.item())
        self.assertAlmostEqual(
            torch.linalg.vector_norm(model()).item(), 1.9, places=5
        )
        torch.testing.assert_close(
            model() / torch.linalg.vector_norm(model()),
            empty_anchor / torch.linalg.vector_norm(empty_anchor),
        )
        torch.testing.assert_close(model.residuals(), model() - target)

    def test_cosine_reference_cases_and_positive_rescaling(self):
        base = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(predicted_noise_cosine(base, base).item(), 1.0)
        self.assertAlmostEqual(
            predicted_noise_cosine(torch.tensor([[0.0, 1.0]]), base).item(),
            0.0,
        )
        self.assertAlmostEqual(
            predicted_noise_cosine(torch.tensor([[-1.0, 0.0]]), base).item(),
            -1.0,
        )
        self.assertAlmostEqual(predicted_noise_cosine(7.0 * base, base).item(), 1.0)

    def test_cosine_is_per_example_and_zero_inputs_have_finite_gradients(self):
        base = torch.tensor([[1.0, 0.0], [100.0, 0.0]])
        edited = torch.tensor([[1.0, 0.0], [-100.0, 0.0]])
        self.assertAlmostEqual(predicted_noise_cosine(edited, base).item(), 0.0)

        zero = torch.zeros_like(base, requires_grad=True)
        loss = predicted_noise_cosine(zero, base)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_paired_calls_share_exact_inputs_and_leave_base_frozen(self):
        unet = _ToyUNet()
        unet.requires_grad_(False)
        state = DiffusionState(
            model_input=torch.randn(2, 3),
            timestep=torch.tensor(4),
            target_hidden_states=torch.randn(2, 2, 3),
        )
        base_weight = dict(unet.named_parameters())["block.attn2.to_v.weight"]
        override = base_weight.detach().clone().requires_grad_(True)
        base_prediction, edited_prediction = paired_predictions(
            unet,
            {"block.attn2.to_v.weight": override},
            state,
        )

        self.assertEqual(len(unet.calls), 2)
        self.assertEqual(unet.calls[0][0].data_ptr(), unet.calls[1][0].data_ptr())
        self.assertEqual(unet.calls[0][1].data_ptr(), unet.calls[1][1].data_ptr())
        self.assertEqual(unet.calls[0][2].data_ptr(), unet.calls[1][2].data_ptr())
        edited_prediction.square().sum().backward()
        self.assertIsNotNone(override.grad)
        self.assertTrue(all(parameter.grad is None for parameter in unet.parameters()))
        torch.testing.assert_close(base_prediction, edited_prediction)

    def test_null_retain_objective_reuses_state_and_is_optional(self):
        unet = _ToyUNet().eval()
        unet.requires_grad_(False)
        target_hidden = torch.ones(2, 2, 3)
        null_hidden = torch.zeros(2, 2, 3)
        state = DiffusionState(
            model_input=torch.randn(2, 3),
            timestep=torch.tensor(4),
            target_hidden_states=target_hidden,
            null_hidden_states=null_hidden,
        )
        base_weight = dict(unet.named_parameters())["block.attn2.to_v.weight"]
        override = base_weight.detach().clone().requires_grad_(True)

        metrics = state_prediction_metrics(
            unet,
            {"block.attn2.to_v.weight": override},
            state,
            1e-8,
            use_null_retain_loss=True,
        )

        self.assertEqual(len(unet.calls), 4)
        for sample, timestep, _ in unet.calls:
            self.assertEqual(sample.data_ptr(), state.model_input.data_ptr())
            self.assertEqual(timestep.data_ptr(), state.timestep.data_ptr())
        self.assertIs(unet.calls[0][2], target_hidden)
        self.assertIs(unet.calls[1][2], target_hidden)
        self.assertIs(unet.calls[2][2], null_hidden)
        self.assertIs(unet.calls[3][2], null_hidden)
        torch.testing.assert_close(
            metrics["loss"], metrics["cosine"] - metrics["null_cosine"]
        )
        torch.testing.assert_close(
            prediction_objective(torch.tensor(0.25), torch.tensor(0.75)),
            torch.tensor(-0.5),
        )

        unet.calls.clear()
        disabled_metrics = state_prediction_metrics(
            unet,
            {"block.attn2.to_v.weight": override},
            state,
            1e-8,
            use_null_retain_loss=False,
        )
        self.assertEqual(len(unet.calls), 2)
        self.assertNotIn("null_cosine", disabled_metrics)
        torch.testing.assert_close(
            disabled_metrics["loss"], disabled_metrics["cosine"]
        )

    def test_null_retain_loss_selects_checkpoint_by_combined_validation(self):
        torch.manual_seed(3)
        unet = _ToyUNet().eval()
        unet.requires_grad_(False)
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        anchor = BoundedAnchor(target, torch.tensor([[[0.3, 0.1, 0.0]]]))
        state = DiffusionState(
            model_input=torch.tensor([[1.0, 2.0, 3.0]]),
            timestep=torch.tensor(1),
            target_hidden_states=target,
            null_hidden_states=torch.zeros_like(target),
        )
        initial_validation = {
            "cosine": 0.2,
            "prediction_norm_ratio": 1.0,
            "mse": 0.0,
            "null_cosine": 0.9,
            "null_prediction_norm_ratio": 1.0,
            "null_mse": 0.0,
            "loss": -0.7,
        }
        final_validation = {
            "cosine": 0.1,
            "prediction_norm_ratio": 1.0,
            "mse": 0.0,
            "null_cosine": 0.1,
            "null_prediction_norm_ratio": 1.0,
            "null_mse": 0.0,
            "loss": 0.0,
        }

        with mock.patch(
            "src.closed_form_anchor_training.evaluate_anchor",
            side_effect=[initial_validation, final_validation],
        ):
            result = optimize_anchor(
                unet,
                _NoOpEditState(unet),
                anchor,
                lambda: state,
                [state],
                AnchorTrainingConfig(
                    steps=1,
                    learning_rate=0.1,
                    validation_interval=1,
                    use_null_retain_loss=True,
                ),
            )

        self.assertEqual(result.best_step, 0)
        self.assertEqual(result.best_validation_cosine, 0.2)
        self.assertEqual(result.best_validation_null_cosine, 0.9)
        self.assertEqual(result.best_validation_loss, -0.7)

    def test_optimizer_keeps_earliest_candidate_on_validation_tie(self):
        torch.manual_seed(3)
        unet = _ToyUNet().eval()
        unet.requires_grad_(False)
        targets = torch.tensor([[[1.0, 0.0, 0.0]]])
        initial = torch.tensor([[[0.3, 0.1, 0.0]]])
        anchor = BoundedAnchor(targets, initial)
        initial_raw = {
            name: value.detach().clone()
            for name, value in anchor.named_parameters()
        }
        state = DiffusionState(
            model_input=torch.tensor([[1.0, 2.0, 3.0]]),
            timestep=torch.tensor(1),
            target_hidden_states=torch.zeros(1, 1, 3),
        )

        result = optimize_anchor(
            unet,
            _NoOpEditState(unet),
            anchor,
            lambda: state,
            [state],
            AnchorTrainingConfig(
                steps=2,
                learning_rate=0.1,
                validation_interval=1,
            ),
        )

        self.assertEqual(result.best_step, 0)
        for name, value in anchor.named_parameters():
            torch.testing.assert_close(value, initial_raw[name], rtol=0, atol=0)

    def test_optimizer_requires_pre_frozen_eval_base_without_mutating_it(self):
        unet = _ToyUNet()
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        anchor = BoundedAnchor(target, target)
        state = DiffusionState(
            model_input=torch.zeros(1, 3),
            timestep=torch.tensor(1),
            target_hidden_states=target,
        )
        with self.assertRaisesRegex(ValueError, "frozen"):
            optimize_anchor(
                unet,
                _NoOpEditState(unet),
                anchor,
                lambda: state,
                [state],
                AnchorTrainingConfig(1, 0.1, 1),
            )
        self.assertTrue(unet.training)
        self.assertTrue(all(parameter.requires_grad for parameter in unet.parameters()))

    def test_prefix_sampling_is_deterministic_and_covers_endpoints(self):
        target_hidden = torch.ones(1, 3, 5)
        null_hidden = torch.zeros(1, 3, 5)

        first_pipe = _FakePipe()
        first = sample_prefix_diffusion_state(
            first_pipe,
            target_hidden,
            null_hidden,
            prompt="target",
            seed=11,
            prefix_index=0,
            num_inference_steps=4,
            guidance_scale=3.0,
            resolution=16,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(first_pipe.unet.calls, 0)
        self.assertEqual(first.prefix_index, 0)
        torch.testing.assert_close(first.null_hidden_states, null_hidden)

        second_pipe = _FakePipe()
        second = sample_prefix_diffusion_state(
            second_pipe,
            target_hidden,
            null_hidden,
            prompt="target",
            seed=11,
            prefix_index=3,
            num_inference_steps=4,
            guidance_scale=3.0,
            resolution=16,
            generator=torch.Generator().manual_seed(11),
        )
        repeated_pipe = _FakePipe()
        repeated = sample_prefix_diffusion_state(
            repeated_pipe,
            target_hidden,
            null_hidden,
            prompt="target",
            seed=11,
            prefix_index=3,
            num_inference_steps=4,
            guidance_scale=3.0,
            resolution=16,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(second_pipe.unet.calls, 3)
        torch.testing.assert_close(second.model_input, repeated.model_input)
        torch.testing.assert_close(second.timestep, repeated.timestep)


if __name__ == "__main__":
    unittest.main()
