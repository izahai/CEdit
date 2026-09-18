"""CPU-only unit tests for ReSAM sparse mixture and numerical optimization."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.closed_form_anchor_training import DiffusionState
from src.resam_training import (
    ReSAMTrainingConfig,
    ReSAMTrainingResult,
    SparseAnchorMixture,
    load_candidate_concepts,
    optimize_resam,
    retain_noise_mse,
)


class _ToyLinear(torch.nn.Module):
    def __init__(self, in_dim=4, out_dim=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(out_dim, in_dim))

    def forward(self, x):
        return x @ self.weight.T


class _ToyUNet(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.attn2_to_v = _ToyLinear(dim, dim)
        self.calls = []

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states=None,
        return_dict=False,
    ):
        self.calls.append((sample, timestep, encoder_hidden_states))
        # Simulated prediction combines sample and conditioning
        cond = encoder_hidden_states.mean(dim=1, keepdim=True)
        return (self.attn2_to_v(sample) + cond,)


class _LinearEditState:
    """Mock edit state where effective weights depend on anchor embedding."""

    def __init__(self, unet: _ToyUNet, target_embeddings: torch.Tensor):
        self.base_weight = dict(unet.named_parameters())["attn2_to_v.weight"].detach().clone()
        self.target_embeddings = target_embeddings

    def effective_parameters(self, anchors: torch.Tensor):
        # Weight edit perturbation proportional to anchor residual
        # anchors has shape [1, 1, d]
        residual = (anchors - self.target_embeddings).flatten()
        # Rank-1 outer product perturbation
        perturbation = torch.outer(residual, residual) * 0.1
        return {
            "attn2_to_v.weight": self.base_weight + perturbation
        }, {"weight_norm": float(torch.linalg.matrix_norm(perturbation).item())}


class ReSAMTrainingTests(unittest.TestCase):
    def test_load_candidate_concepts_from_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "candidates.csv"
            csv_path.write_text(
                "id,concept\n1, dog \n2,cartoon dog\n3,beagle\n",
                encoding="utf-8",
            )
            candidates, file_hash = load_candidate_concepts(csv_path)
            self.assertEqual(candidates, ["dog", "cartoon dog", "beagle"])
            self.assertEqual(len(file_hash), 64)

    def test_load_candidate_concepts_from_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_path = Path(temp_dir) / "candidates.txt"
            txt_path.write_text("dog\ncartoon dog\nbeagle\n", encoding="utf-8")
            candidates, file_hash = load_candidate_concepts(txt_path)
            self.assertEqual(candidates, ["dog", "cartoon dog", "beagle"])
            self.assertEqual(len(file_hash), 64)

    def test_load_candidate_concepts_rejects_duplicates_and_blanks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dup_csv = Path(temp_dir) / "dup.csv"
            dup_csv.write_text("id,concept\n1,dog\n2,dog\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate concept"):
                load_candidate_concepts(dup_csv)

            blank_csv = Path(temp_dir) / "blank.csv"
            blank_csv.write_text("id,concept\n1,dog\n2,  \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank concept"):
                load_candidate_concepts(blank_csv)

            empty_csv = Path(temp_dir) / "empty.csv"
            empty_csv.write_text("id,concept\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_candidate_concepts(empty_csv)

            bad_header = Path(temp_dir) / "bad.csv"
            bad_header.write_text("id,name\n1,dog\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a 'concept' or 'prompt' column"):
                load_candidate_concepts(bad_header)

    def test_sparse_anchor_mixture_forward_and_sparsity(self):
        torch.manual_seed(42)
        candidates = torch.randn(5, 1, 4)
        target = torch.randn(1, 1, 4)
        mixture = SparseAnchorMixture(candidates, target, k=2, temperature=1.0)

        # Set specific scores: highest at indices 1 and 3
        with torch.no_grad():
            mixture.raw_scores.copy_(torch.tensor([0.1, 2.5, -0.5, 3.0, -1.0]))

        weights, hard, topk_indices = mixture.compute_weights()

        # Check top-k selection
        self.assertEqual(sorted(topk_indices.tolist()), [1, 3])
        # Check hard weights sum to 1.0
        self.assertAlmostEqual(float(hard.sum().item()), 1.0, places=5)
        # Check sparsity: only indices 1 and 3 are non-zero
        self.assertEqual(int((hard > 0).sum().item()), 2)
        self.assertEqual(float(hard[0].item()), 0.0)
        self.assertEqual(float(hard[2].item()), 0.0)
        self.assertEqual(float(hard[4].item()), 0.0)

        # Check forward output shape and finiteness
        anchor = mixture()
        self.assertEqual(anchor.shape, (1, 1, 4))
        self.assertTrue(torch.isfinite(anchor).all().item())

    def test_deterministic_bank_order_ties(self):
        candidates = torch.randn(4, 1, 4)
        target = torch.randn(1, 1, 4)
        # All scores equal to 0.0
        mixture = SparseAnchorMixture(candidates, target, k=2)
        _, _, topk_indices = mixture.compute_weights()
        # Stable tie breaking must select index 0 and 1 in bank order
        self.assertEqual(topk_indices.tolist(), [0, 1])

    def test_boundary_k_values(self):
        candidates = torch.randn(3, 1, 4)
        target = torch.randn(1, 1, 4)

        # k = 1
        m1 = SparseAnchorMixture(candidates, target, k=1)
        with torch.no_grad():
            m1.raw_scores.copy_(torch.tensor([1.0, 5.0, 2.0]))
        _, hard1, idx1 = m1.compute_weights()
        self.assertEqual(idx1.tolist(), [1])
        self.assertAlmostEqual(float(hard1[1].item()), 1.0, places=5)

        # k = V (3)
        m3 = SparseAnchorMixture(candidates, target, k=3)
        _, hard3, idx3 = m3.compute_weights()
        self.assertEqual(len(idx3), 3)
        self.assertAlmostEqual(float(hard3.sum().item()), 1.0, places=5)

        # Invalid k
        with self.assertRaisesRegex(ValueError, "k must be between 1 and bank size"):
            SparseAnchorMixture(candidates, target, k=0)
        with self.assertRaisesRegex(ValueError, "k must be between 1 and bank size"):
            SparseAnchorMixture(candidates, target, k=4)

    def test_straight_through_estimator_gradients(self):
        candidates = torch.randn(4, 1, 4)
        target = torch.randn(1, 1, 4)

        # STE enabled: unselected candidates MUST receive gradients from soft surrogate
        ste_mixture = SparseAnchorMixture(candidates, target, k=2, temperature=1.0, use_ste=True)
        with torch.no_grad():
            ste_mixture.raw_scores.copy_(torch.tensor([3.0, 2.0, 0.5, 0.1]))
        anchor_ste = ste_mixture()
        loss_ste = (anchor_ste * torch.ones_like(anchor_ste)).sum()
        loss_ste.backward()

        self.assertIsNotNone(ste_mixture.raw_scores.grad)
        # All 4 candidates must receive a non-zero gradient under STE
        for i in range(4):
            self.assertNotEqual(float(ste_mixture.raw_scores.grad[i].item()), 0.0)

        # STE disabled: unselected candidates MUST have 0.0 gradient
        non_ste_mixture = SparseAnchorMixture(candidates, target, k=2, temperature=1.0, use_ste=False)
        with torch.no_grad():
            non_ste_mixture.raw_scores.copy_(torch.tensor([3.0, 2.0, 0.5, 0.1]))
        anchor_non_ste = non_ste_mixture()
        loss_non_ste = (anchor_non_ste * torch.ones_like(anchor_non_ste)).sum()
        loss_non_ste.backward()

        self.assertIsNotNone(non_ste_mixture.raw_scores.grad)
        # Indices 2 and 3 were unselected, so their gradient must be exactly 0.0
        self.assertEqual(float(non_ste_mixture.raw_scores.grad[2].item()), 0.0)
        self.assertEqual(float(non_ste_mixture.raw_scores.grad[3].item()), 0.0)

    def test_retain_noise_mse(self):
        pred1 = torch.tensor([1.0, 2.0, 3.0])
        pred2 = torch.tensor([2.0, 2.0, 5.0])
        # (1^2 + 0^2 + 2^2) / 3 = 5/3
        mse = retain_noise_mse(pred1, pred2)
        self.assertAlmostEqual(float(mse.item()), 5.0 / 3.0, places=5)

    def test_optimize_resam_runs_and_records_history(self):
        dim = 4
        unet = _ToyUNet(dim=dim)
        unet.eval()
        unet.requires_grad_(False)
        target = torch.zeros(1, 1, dim)
        edit_state = _LinearEditState(unet, target)
        candidates = torch.randn(3, 1, dim)
        names = ["c0", "c1", "c2"]

        mixture = SparseAnchorMixture(candidates, target, k=2, temperature=1.0, candidate_names=names)
        state = DiffusionState(
            model_input=torch.randn(1, 1, dim),
            timestep=torch.tensor([20]),
            target_hidden_states=torch.randn(1, 1, dim),
            null_hidden_states=torch.randn(1, 1, dim),
        )

        config = ReSAMTrainingConfig(
            steps=3,
            learning_rate=0.05,
            use_null_retain_loss=False,
        )

        result = optimize_resam(
            unet,
            edit_state,
            mixture,
            training_state_factory=lambda: state,
            config=config,
        )

        self.assertEqual(result.final_step, 3)
        self.assertEqual(len(result.history), 3)
        self.assertIsInstance(result.final_loss, float)

        for rec in result.history:
            self.assertIn("step", rec)
            self.assertIn("train_loss", rec)
            self.assertIn("retain_loss", rec)
            self.assertNotIn("null_loss", rec)
            self.assertEqual(rec["train_loss"], rec["retain_loss"])
            self.assertIn("grad_norm", rec)
            self.assertIn("selected_indices", rec)
            self.assertEqual(len(rec["selected_indices"]), 2)

    def test_optimize_resam_with_null_retain_loss(self):
        dim = 4
        unet = _ToyUNet(dim=dim)
        unet.eval()
        unet.requires_grad_(False)
        target = torch.zeros(1, 1, dim)
        edit_state = _LinearEditState(unet, target)
        candidates = torch.randn(3, 1, dim)
        names = ["c0", "c1", "c2"]

        mixture = SparseAnchorMixture(candidates, target, k=2, temperature=1.0, candidate_names=names)
        state = DiffusionState(
            model_input=torch.randn(1, 1, dim),
            timestep=torch.tensor([20]),
            target_hidden_states=torch.randn(1, 1, dim),
            null_hidden_states=torch.randn(1, 1, dim),
        )

        config = ReSAMTrainingConfig(
            steps=3,
            learning_rate=0.05,
            use_null_retain_loss=True,
        )

        result = optimize_resam(
            unet,
            edit_state,
            mixture,
            training_state_factory=lambda: state,
            config=config,
        )

        self.assertEqual(result.final_step, 3)
        self.assertEqual(len(result.history), 3)

        for rec in result.history:
            self.assertIn("null_loss", rec)
            expected_loss = 0.5 * (rec["retain_loss"] + rec["null_loss"])
            self.assertAlmostEqual(rec["train_loss"], expected_loss, places=5)

        # Rejects missing null_hidden_states when use_null_retain_loss=True
        state_no_null = DiffusionState(
            model_input=torch.randn(1, 1, dim),
            timestep=torch.tensor([20]),
            target_hidden_states=torch.randn(1, 1, dim),
            null_hidden_states=None,
        )
        with self.assertRaisesRegex(ValueError, "null_hidden_states is required"):
            optimize_resam(
                unet,
                edit_state,
                mixture,
                training_state_factory=lambda: state_no_null,
                config=config,
            )

    def test_base_unet_parameters_not_mutated_during_training(self):
        dim = 4
        unet = _ToyUNet(dim=dim)
        unet.eval()
        unet.requires_grad_(False)
        initial_param = unet.attn2_to_v.weight.detach().clone()

        target = torch.zeros(1, 1, dim)
        edit_state = _LinearEditState(unet, target)
        candidates = torch.randn(3, 1, dim)
        mixture = SparseAnchorMixture(candidates, target, k=2)

        state = DiffusionState(
            model_input=torch.randn(1, 1, dim),
            timestep=torch.tensor([20]),
            target_hidden_states=torch.randn(1, 1, dim),
        )
        config = ReSAMTrainingConfig(
            steps=3,
            learning_rate=0.05,
        )
        optimize_resam(
            unet,
            edit_state,
            mixture,
            training_state_factory=lambda: state,
            config=config,
        )
        # Check base unet parameters were never mutated
        self.assertTrue(torch.equal(unet.attn2_to_v.weight.detach(), initial_param))


if __name__ == "__main__":
    unittest.main()
