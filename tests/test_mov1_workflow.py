import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from motivation.mov1_draft3.summarize_results import (
    build_rows,
    summarize_gcd,
    wilson_interval,
)
from motivation.mov1_draft3.workflow_config import build_environment


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "motivation" / "mov1"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Mov1WorkflowTests(unittest.TestCase):
    def test_analyzer_entrypoint_resolves_repository_imports_from_any_cwd(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1"})
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(WORKFLOW_DIR / "analyze_residuals.py"),
                    "--help",
                ],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_workflow_locks_paper_strength_matrix(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.assertEqual(config["experiment"]["benchmark_name"], "100_celebrity")
        self.assertEqual(
            config["experiment"]["methods"],
            ["legacy_full", "legacy_svd_rank30", "tgprs_rank30"],
        )
        self.assertEqual(
            config["experiment"]["residual_scales"],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(config["experiment"]["inference_timesteps"], 50)
        self.assertEqual(config["experiment"]["expected_images_per_split"], 500)

    def test_environment_keeps_outputs_inside_mov1(self):
        with (WORKFLOW_DIR / "workflow.yaml").open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        with tempfile.TemporaryDirectory() as workspace:
            repo_root = Path(workspace) / "CEdit"
            with patch.dict(
                os.environ,
                {"WORKSPACE_DIR": workspace, "REPO_ROOT": str(repo_root)},
                clear=False,
            ):
                environment = build_environment(config, str(WORKFLOW_DIR))

        self.assertEqual(
            environment["OUTPUT_ROOT"],
            str(repo_root / "motivation" / "mov1" / "outputs"),
        )
        self.assertEqual(
            environment["METHODS_RAW"],
            "legacy_full legacy_svd_rank30 tgprs_rank30",
        )

    def test_run_all_orders_analysis_training_evaluation_and_plotting(self):
        run_all = (WORKFLOW_DIR / "run_all.sh").read_text(encoding="utf-8")
        stages = [
            "02_analyze_residuals.sh",
            "03_train.sh",
            "04_infer.sh",
            "05_evaluate.sh",
            "06_summarize.sh",
            "07_plot.sh",
        ]
        positions = [run_all.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("mscoco", run_all.lower())

    def test_wilson_interval_and_gcd_summary(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "gcd.csv"
            write_csv(
                csv_path,
                [
                    {"p_celebrity_correct": "0.8"},
                    {"p_celebrity_correct": "0"},
                    {"p_celebrity_correct": "N"},
                    {"p_celebrity_correct": "0.2"},
                ],
            )
            summary = summarize_gcd(csv_path, expected_count=4)

        self.assertEqual(summary["n_faces_detected"], 3)
        self.assertEqual(summary["n_identity_hits"], 2)
        self.assertEqual(summary["identity_hit_rate"], 0.5)

    def test_builds_one_wide_metric_row_per_method_and_scale(self):
        config = {
            "experiment": {
                "methods": ["legacy_full"],
                "residual_scales": [0.2],
                "residual_rank": 30,
                "expected_images_per_split": 2,
            }
        }
        ranks = {
            "legacy_full": {
                "residual_numerical_rank": "2",
                "residual_stable_rank": "1.5",
                "residual_spectral_effective_rank": "1.7",
                "edit_statistic_numerical_rank": "2",
                "edit_statistic_stable_rank": "1.4",
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "gcd" / "legacy_full" / "scale_0p2"
            write_csv(
                run_dir / "erase.csv",
                [{"p_celebrity_correct": "0"}, {"p_celebrity_correct": "N"}],
            )
            write_csv(
                run_dir / "retain.csv",
                [{"p_celebrity_correct": "0.7"}, {"p_celebrity_correct": "0.9"}],
            )
            checkpoint_path = (
                root / "checkpoints" / "legacy_full" / "scale_0p2" / "weight.pt"
            )
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.touch()
            rows = build_rows(config, ranks, root / "gcd", root / "checkpoints")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["erasure_success"], 1.0)
        self.assertEqual(rows[0]["retain_identity_hit_rate"], 1.0)
        self.assertTrue(rows[0]["checkpoint_path"].endswith("weight.pt"))

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is not installed",
    )
    def test_plotter_writes_pdf_and_png(self):
        from motivation.mov1_draft3.plot_figure import render_figure

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spectrum_rows = []
            rank_rows = []
            metric_rows = []
            labels = {
                "legacy_full": "Legacy full-rank",
                "legacy_svd_rank30": "Legacy-SVD rank 30",
                "tgprs_rank30": "TGPRS rank 30",
            }
            for method_index, (method, label) in enumerate(labels.items()):
                rank_rows.append({
                    "method": method,
                    "residual_numerical_rank": 100 if method_index == 0 else 30,
                    "residual_stable_rank": 4.0 + method_index,
                    "edit_statistic_numerical_rank": 90 if method_index == 0 else 30,
                })
                for component in range(1, 101):
                    spectrum_rows.append({
                        "method": method,
                        "method_label": label,
                        "component": component,
                        "normalized_energy": 1.0 / 100,
                    })
                metric_rows.append({
                    "method": method,
                    "method_label": label,
                    "residual_scale": 0.2,
                    "retain_identity_hit_rate": 0.9,
                    "retain_identity_hit_ci_low": 0.87,
                    "retain_identity_hit_ci_high": 0.92,
                    "erasure_success": 0.95,
                    "erasure_success_ci_low": 0.92,
                    "erasure_success_ci_high": 0.97,
                })
            spectrum_path = root / "spectrum.csv"
            rank_path = root / "ranks.csv"
            metrics_path = root / "metrics.csv"
            write_csv(spectrum_path, spectrum_rows)
            write_csv(rank_path, rank_rows)
            write_csv(metrics_path, metric_rows)

            pdf_path, png_path = render_figure(
                spectrum_path, rank_path, metrics_path, root / "figures"
            )

            self.assertTrue(pdf_path.is_file())
            self.assertTrue(png_path.is_file())


if __name__ == "__main__":
    unittest.main()
