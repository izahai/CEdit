import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_rank_analysis import (
    aggregate_edits,
    plot_common_anchor,
    plot_direction_ablation,
    plot_pareto,
    plot_retain_profile_robustness,
    plot_threshold_sensitivity,
)


def aggregate_row(method, rank, profile="fixed", threshold=0.1, scale=0.1):
    return {
        "layer_index": "aggregate",
        "method": method,
        "direction_variant": method,
        "requested_rank": rank,
        "retain_profile": profile,
        "retain_threshold": threshold,
        "retain_scale": scale,
        "target_effect_mean": 1.0,
        "target_rotation_deg_mean": 50.0,
        "retain_leakage_mean": 0.01,
        "directional_realization_cosine_mean": 0.99,
        "directional_relative_error_mean": 0.1,
        "edited_output_norm_ratio_mean": 0.8,
        "canonical_erasure_alignment_mean": 0.9,
        "anchor_distance_ratio_mean": 0.7,
        "retain_low_rank_mean": 10.0,
    }


class PlotCompatibilityTests(unittest.TestCase):
    def test_old_schema_receives_compatibility_defaults(self):
        row = aggregate_row("legacy_full", None)
        row.pop("retain_profile")
        row.pop("direction_variant")
        frame = aggregate_edits([row])
        self.assertEqual(frame.iloc[0]["retain_profile"], "fixed")
        self.assertAlmostEqual(frame.iloc[0]["retain_threshold"], 0.1)

    def test_new_figures_smoke(self):
        rows = []
        for threshold in (0.1, 1.5):
            for method, rank in (
                ("legacy_full", None),
                ("legacy_svd", 30),
                ("random_subspace", 30),
                ("tgprs", 5),
                ("tgprs", 10),
                ("tgprs", 30),
                ("tgprs_positive", 30),
                ("tgprs_complement", 30),
                ("random_negative_target", 30),
            ):
                rows.append(aggregate_row(method, rank, threshold=threshold))
        for method, rank in (
            ("legacy_full", None),
            ("tgprs", 5),
            ("tgprs", 10),
            ("tgprs", 30),
        ):
            rows.append(aggregate_row(method, rank, profile="full_speed"))
        grouped = aggregate_edits(rows)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_pareto(grouped, output, 50)
            plot_threshold_sensitivity(grouped, output, 50)
            plot_retain_profile_robustness(grouped, output, 50)
            plot_direction_ablation(grouped, output, 50)
            self.assertTrue((output / "figure_c_rank_pareto.png").exists())
            self.assertTrue((output / "figure_g_direction_ablation.pdf").exists())

    def test_common_anchor_figure_smoke(self):
        payload = {
            "target_coordinates": [[0.0, 0.0], [1.0, 0.0]],
            "anchor_coordinate": [0.5, 1.0],
            "target_names": ["one", "two"],
            "anchor_name": "person",
            "pca_explained_variance_ratio": [0.6, 0.3],
            "residual_cosine_similarity": [[1.0, 0.2], [0.2, 1.0]],
            "residual": {"numerical_rank": 2, "effective_rank": 1.5},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "common.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            plot_common_anchor(source, output, 50)
            self.assertTrue(
                (output / "figure_0_common_anchor_directions.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
