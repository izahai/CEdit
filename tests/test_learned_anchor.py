import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import CLIPTextConfig, CLIPTextModel

from src.learned_anchor import (
    ANCHOR_REPRESENTATION,
    ARTIFACT_SCHEMA_VERSION,
    LearnedAnchorBundle,
    LearnedAnchorConfig,
    StraightThroughCategorical,
    build_prompt_layout,
    differentiable_clip_preprocess,
    encode_soft_prompt,
    initialize_token_distributions,
    load_learned_anchors,
    normalize_target_concepts,
    project_rows_to_simplex,
    reconstruct_clean_latent,
)


class FakeTokenizer:
    model_max_length = 8
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self):
        return 12

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        padding=None,
        max_length=None,
        return_tensors=None,
    ):
        content = [3, 4] if text == "Snoopy" else [7, 8, 9, 10, 11]
        if not add_special_tokens:
            return {"input_ids": content}
        ids = [self.bos_token_id] + content + [self.eos_token_id]
        attention = [1] * len(ids)
        if truncation and max_length is not None:
            ids = ids[:max_length]
            attention = attention[:max_length]
        if padding == "max_length":
            ids.extend([self.pad_token_id] * (max_length - len(ids)))
            attention.extend([0] * (max_length - len(attention)))
        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.tensor([attention], dtype=torch.long),
            )
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


def tiny_text_encoder():
    torch.manual_seed(7)
    config = CLIPTextConfig(
        vocab_size=12,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=2,
    )
    return CLIPTextModel(config).eval()


class LearnedAnchorMathTests(unittest.TestCase):
    def test_uniform_initialization_and_simplex_projection(self):
        initialized = initialize_token_distributions(3, 5, "cpu")
        projected = project_rows_to_simplex(
            torch.tensor(
                [[-2.0, 0.2, 0.4, 1.7, 0.3], [4.0, -1.0, 0.5, 0.5, 0.5]]
            )
        )

        self.assertTrue(torch.allclose(initialized, torch.full((3, 5), 0.2)))
        self.assertTrue((projected >= 0).all())
        self.assertTrue((projected <= 1).all())
        self.assertTrue(torch.allclose(projected.sum(-1), torch.ones(2), atol=1e-5))

    def test_straight_through_sample_is_hard_and_has_gradient(self):
        probabilities = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            requires_grad=True,
        )
        sample = StraightThroughCategorical.apply(
            probabilities,
            torch.Generator().manual_seed(1),
        )
        sample.sum().backward()

        self.assertTrue(torch.equal(sample, probabilities.detach()))
        self.assertIsNotNone(probabilities.grad)
        self.assertTrue(torch.equal(probabilities.grad, torch.ones_like(probabilities)))

    def test_prompt_layout_inserts_prefix_and_tracks_last_target_token(self):
        tokenizer = FakeTokenizer()
        layout = build_prompt_layout(tokenizer, "Snoopy", [5, 6])

        self.assertEqual(layout.input_ids.tolist()[0], [0, 5, 6, 3, 4, 2, 2, 2])
        self.assertEqual(layout.target_token_index, 4)
        self.assertEqual(layout.target_token_ids, (3, 4))
        self.assertFalse(layout.target_was_truncated)

    def test_prompt_layout_truncates_target_but_keeps_one_token(self):
        layout = build_prompt_layout(FakeTokenizer(), "long target", [3, 4, 5, 6])

        self.assertEqual(layout.target_token_ids, (7, 8))
        self.assertEqual(layout.input_ids.shape, (1, 8))
        self.assertTrue(layout.target_was_truncated)

    def test_soft_encoder_matches_normal_encoder_for_hard_embeddings(self):
        text_encoder = tiny_text_encoder()
        input_ids = torch.tensor([[0, 5, 6, 3, 4, 2, 2, 2]])
        input_embeddings = text_encoder.get_input_embeddings()(input_ids)

        expected = text_encoder(input_ids=input_ids).last_hidden_state
        actual = encode_soft_prompt(text_encoder, input_ids, input_embeddings)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_clean_latent_reconstruction_recovers_original_for_exact_noise(self):
        clean = torch.randn(1, 4, 2, 2)
        noise = torch.randn_like(clean)
        alphas = torch.tensor([0.9, 0.6, 0.2])
        timestep = torch.tensor([1])
        noised = alphas[1].sqrt() * clean + (1 - alphas[1]).sqrt() * noise

        recovered = reconstruct_clean_latent(noised, noise, alphas, timestep)

        self.assertTrue(torch.allclose(recovered, clean, atol=1e-6))

    def test_clip_preprocessing_preserves_image_gradient(self):
        image = torch.rand(1, 3, 16, 16, requires_grad=True)
        processor = SimpleNamespace(
            crop_size={"height": 8, "width": 8},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.25, 0.25, 0.25],
        )

        output = differentiable_clip_preprocess(image, processor)
        output.sum().backward()

        self.assertEqual(output.shape, (1, 3, 8, 8))
        self.assertIsNotNone(image.grad)

    def test_target_normalization_rejects_duplicates_and_nudity(self):
        self.assertEqual(
            normalize_target_concepts("Snoopy, Mickey"),
            ["Snoopy", "Mickey"],
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            normalize_target_concepts("Snoopy,Snoopy")
        with self.assertRaisesRegex(ValueError, "nudity"):
            normalize_target_concepts("Nudity")

class LearnedAnchorArtifactTests(unittest.TestCase):
    def make_bundle(self):
        tokenizer = FakeTokenizer()
        text_encoder = tiny_text_encoder()
        prefix_ids = torch.tensor([5, 6])
        layout = build_prompt_layout(tokenizer, "Snoopy", prefix_ids.tolist())
        with torch.no_grad():
            anchor_states = text_encoder(layout.input_ids).last_hidden_state
            anchor_hidden = anchor_states[0, layout.target_token_index].unsqueeze(0)
            target_inputs = tokenizer(
                "Snoopy",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            target_states = text_encoder(target_inputs.input_ids).last_hidden_state
            target_hidden = target_states[0, 2].unsqueeze(0)
        distributions = torch.nn.functional.one_hot(
            prefix_ids,
            num_classes=len(tokenizer),
        ).float()
        config = LearnedAnchorConfig(
            num_prefix_tokens=2,
            device="cpu",
            dtype="float32",
        )
        target_id = "target_0000"
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "complete",
            "sd_ckpt": config.sd_ckpt,
            "clip_model": config.clip_model,
            "anchor_representation": ANCHOR_REPRESENTATION,
            "scheduler_prediction_type": "epsilon",
            "target_order": ["Snoopy"],
            "tokenizer": {
                "vocab_size": len(tokenizer),
                "model_max_length": tokenizer.model_max_length,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
            "text_hidden_size": text_encoder.config.hidden_size,
            "config": {"num_prefix_tokens": 2},
            "targets": [
                {
                    "target_id": target_id,
                    "target": "Snoopy",
                    "prefix_token_ids": prefix_ids.tolist(),
                }
            ],
        }
        tensors = {
            f"targets.{target_id}.token_distributions": distributions,
            f"targets.{target_id}.prefix_token_ids": prefix_ids,
            f"targets.{target_id}.anchor_hidden_state": anchor_hidden,
            f"targets.{target_id}.target_hidden_state": target_hidden,
        }
        pipeline = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
        return LearnedAnchorBundle(manifest, tensors, [], {}), pipeline

    def test_artifact_round_trip_recomputes_anchor(self):
        bundle, pipeline = self.make_bundle()
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle.save(temp_dir)
            anchors = load_learned_anchors(
                temp_dir,
                ["Snoopy"],
                pipeline,
                expected_sd_ckpt="CompVis/stable-diffusion-v1-4",
            )

        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].shape, (1, 8))

    def test_loader_rejects_reordered_or_changed_targets(self):
        bundle, pipeline = self.make_bundle()
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle.save(temp_dir)
            manifest_path = Path(temp_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["target_order"] = ["Mickey"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "target order"):
                load_learned_anchors(temp_dir, ["Snoopy"], pipeline)


if __name__ == "__main__":
    unittest.main()
