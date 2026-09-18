import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remote_scripts.eval_ReSAM.erase_van_gogh.compute_clip_similarity import (
    compute_similarity_rankings,
    encode_texts,
    load_artists,
    write_csv,
)


class TestComputeClipSimilarity(unittest.TestCase):
    def test_load_artists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "style.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "concept"])
                writer.writerow(["1", "Claude Monet"])
                writer.writerow(["2", "Van Gogh"])
                writer.writerow(["3", ""])  # blank should be skipped

            artists = load_artists(csv_path)
            self.assertEqual(len(artists), 2)
            self.assertEqual(artists[0], {"id": "1", "artist": "Claude Monet"})
            self.assertEqual(artists[1], {"id": "2", "artist": "Van Gogh"})

    def test_compute_similarity_rankings(self):
        # 3 mock artists with known 2D embeddings
        artists = [
            {"id": "1", "artist": "Claude Monet"},
            {"id": "2", "artist": "Van Gogh"},
            {"id": "3", "artist": "Orthogonal Artist"},
        ]

        # Target is along [1, 0]
        target_emb = torch.tensor([[1.0, 0.0]])

        # Artist embeddings:
        # Monet: [0.8, 0.6] -> cosine 0.8
        # Van Gogh: [1.0, 0.0] -> cosine 1.0
        # Orthogonal: [0.0, 1.0] -> cosine 0.0
        artist_embs = torch.tensor([
            [0.8, 0.6],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        ranked = compute_similarity_rankings(target_emb, artist_embs, artists)

        self.assertEqual(len(ranked), 3)
        # Rank 1 must be Van Gogh with 1.000000
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["artist"], "Van Gogh")
        self.assertEqual(ranked[0]["similarity"], "1.000000")

        # Rank 2 must be Monet with 0.800000
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[1]["artist"], "Claude Monet")
        self.assertEqual(ranked[1]["similarity"], "0.800000")

        # Rank 3 must be Orthogonal with 0.000000
        self.assertEqual(ranked[2]["rank"], 3)
        self.assertEqual(ranked[2]["artist"], "Orthogonal Artist")
        self.assertEqual(ranked[2]["similarity"], "0.000000")

    def test_write_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "similarity.csv"
            sample_rows = [
                {"rank": 1, "id": "10", "artist": "Van Gogh", "similarity": "1.000000"},
                {"rank": 2, "id": "5", "artist": "Paul Gauguin", "similarity": "0.852300"},
            ]
            write_csv(out_path, sample_rows)

            self.assertTrue(out_path.is_file())
            with out_path.open("r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["artist"], "Van Gogh")
                self.assertEqual(rows[1]["similarity"], "0.852300")

    def test_encode_texts_mock(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 77

        # Mock tokenization returning dict-like tensor batch
        mock_tokens = MagicMock()
        mock_tokens.to.return_value = mock_tokens
        mock_tokens.attention_mask = torch.ones((2, 77))
        mock_tokenizer.return_value = mock_tokens

        # Mock model output having text_embeds
        # Return arbitrary 4D vectors for 2 items
        raw_embeds = torch.tensor([
            [3.0, 4.0, 0.0, 0.0],  # norm = 5 -> [0.6, 0.8, 0, 0]
            [0.0, 0.0, 1.0, 0.0],  # norm = 1 -> [0, 0, 1, 0]
        ])
        mock_output = MagicMock()
        mock_output.text_embeds = raw_embeds
        mock_model.return_value = mock_output

        result = encode_texts(mock_model, mock_tokenizer, ["artist1", "artist2"], device="cpu", batch_size=2)
        self.assertEqual(result.shape, (2, 4))
        # Verify L2 normalization
        norms = torch.norm(result, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(result[0], torch.tensor([0.6, 0.8, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

