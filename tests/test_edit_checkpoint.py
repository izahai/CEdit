import tempfile
import unittest
from pathlib import Path

import torch

from src.closed_form_anchor_training import DiffusionState
from src.differentiable_legacy_edit import SPEED_CHECKPOINT_FORMAT
from src.edit_checkpoint import (
    apply_edit_checkpoint,
    load_edit_checkpoint,
    save_speed_checkpoint,
)
from train_closed_form_backprop import verify_materialized_prediction


class _ToyPredictionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = torch.nn.Module()
        self.attn2.to_v = torch.nn.Linear(3, 2, bias=False)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
    ):
        return (self.attn2.to_v(sample),)


class EditCheckpointTests(unittest.TestCase):
    def test_fresh_loaded_prediction_matches_functional_override(self):
        module = _ToyPredictionModule().eval()
        base_weight = module.attn2.to_v.weight.detach()
        changed_weight = base_weight + 0.2
        state = DiffusionState(
            model_input=torch.randn(2, 3),
            timestep=torch.tensor(1),
            target_hidden_states=torch.zeros(2, 1, 3),
        )
        weights = {"attn2.to_v.weight": changed_weight}
        diagnostics = verify_materialized_prediction(module, weights, weights, state)
        self.assertTrue(diagnostics["prediction_allclose"])
        self.assertEqual(diagnostics["prediction_max_abs_error"], 0.0)
        torch.testing.assert_close(module.attn2.to_v.weight, base_weight)

        with self.assertRaisesRegex(RuntimeError, "differs"):
            verify_materialized_prediction(
                module,
                weights,
                {"attn2.to_v.weight": base_weight},
                state,
            )

    def test_safetensors_round_trip_preserves_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weight.safetensors"
            tensors = {"weight": torch.randn(2, 3)}
            save_speed_checkpoint(
                tensors,
                path,
                {"component": "unet", "format": "incorrect"},
            )
            loaded, metadata = load_edit_checkpoint(
                path,
                reference_state={"weight": torch.zeros(2, 3)},
            )

        torch.testing.assert_close(loaded["weight"], tensors["weight"])
        self.assertEqual(metadata["format"], SPEED_CHECKPOINT_FORMAT)
        self.assertEqual(metadata["component"], "unet")

    def test_unknown_safetensors_format_is_rejected(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unknown.safetensors"
            save_file({"weight": torch.zeros(2, 3)}, str(path))
            with self.assertRaisesRegex(ValueError, "Unsupported.*format"):
                load_edit_checkpoint(path)

    def test_legacy_pt_checkpoint_loads_and_applies(self):
        module = torch.nn.Linear(3, 2, bias=False)
        replacement = torch.full_like(module.weight, 2.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            torch.save({"weight": replacement}, path)
            metadata, result = apply_edit_checkpoint(module, path)

        self.assertEqual(metadata, {})
        self.assertEqual(result.unexpected_keys, [])
        torch.testing.assert_close(module.weight, replacement)

    def test_invalid_keys_shapes_and_nonfinite_values_fail_before_loading(self):
        reference = {"weight": torch.zeros(2, 3)}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            unexpected = temp_dir / "unexpected.pt"
            wrong_shape = temp_dir / "wrong_shape.pt"
            nonfinite = temp_dir / "nonfinite.pt"
            torch.save({"other": torch.zeros(2, 3)}, unexpected)
            torch.save({"weight": torch.zeros(3, 2)}, wrong_shape)
            torch.save({"weight": torch.tensor([[float("nan")]])}, nonfinite)

            with self.assertRaisesRegex(KeyError, "unexpected"):
                load_edit_checkpoint(unexpected, reference_state=reference)
            with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                load_edit_checkpoint(wrong_shape, reference_state=reference)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_edit_checkpoint(nonfinite)


if __name__ == "__main__":
    unittest.main()
