import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "remote_scripts/eval_100_style/workflow.py"
SPEC = importlib.util.spec_from_file_location("eval_100_style_workflow", SCRIPT)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class Style100WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = workflow.load_config(SCRIPT.parent / "workflow.yaml")
        cls.smoke = workflow.load_config(SCRIPT.parent / "workflow_smoke.yaml")

    def test_committed_split_membership_and_source_validation(self):
        erase, retain_eval, retain_train = workflow.validate_committed_splits()
        self.assertEqual((len(erase), len(retain_eval), len(retain_train)), (100, 100, 1634))
        ids = {int(row["id"]) for row in erase}
        self.assertEqual(ids, (set(range(401, 501)) - {437}) | {1266})
        self.assertEqual([int(row["id"]) for row in retain_eval], list(range(1, 101)))
        self.assertTrue({r["id"] for r in retain_eval} <= {r["id"] for r in retain_train})
        self.assertEqual(next(r["concept"] for r in erase if r["id"] == "1266"), "Monet")

    def test_full_manifest_counts_and_paired_seeds(self):
        rows = workflow.build_manifest(self.config, workflow.selected_splits(self.config),
                                       workflow.read_csv(ROOT / "data/mscoco.csv"))
        self.assertEqual({group: sum(row["group"] == group for row in rows)
                          for group in ("erase", "retain", "coco")},
                         {"erase": 200, "retain": 200, "coco": 100})
        self.assertEqual(len(rows) * 3, 1500)
        first = [r for r in rows if r["artist_id"] == 401]
        self.assertEqual(len(first), 2)
        self.assertEqual({r["seed"] for r in first}, {0})
        self.assertEqual({r["filename"].split("_")[1] for r in first}, {"0", "1"})
        for group in ("erase", "retain"):
            group_rows = [r for r in rows if r["group"] == group]
            self.assertEqual(sum(r["filename"].split("_")[1] == "0" for r in group_rows), 100)
            self.assertEqual({index: sum(r["filename"].split("_")[1] == str(index) for r in group_rows)
                              for index in range(1, 5)}, {1: 25, 2: 25, 3: 25, 4: 25})
        self.assertEqual(len({(r["group"], r["filename"]) for r in rows}), len(rows))

    def test_training_parity_and_smoke_rank(self):
        self.assertEqual(set(self.config["training"]), {"legacy", workflow.METHODS[2]})
        self.assertNotEqual(self.config["training"]["legacy"], self.config["training"][workflow.METHODS[2]])
        targets = workflow.selected_splits(self.config)["erase"]
        legacy = workflow.training_config(self.config, "legacy", targets)
        tgprs = workflow.training_config(self.config, workflow.METHODS[2], targets)
        for key in ("target_concepts", "anchor_concepts", "retain_path", "erase_style", "params",
                    "threshold", "retain_scale", "seed"):
            self.assertEqual(legacy[key], tgprs[key])
        self.assertEqual((legacy["aug_num"], tgprs["aug_num"]), (10, 0))
        self.assertEqual(tgprs["residual_rank"], 30)
        self.assertEqual(tgprs["subspace_anchor_concepts"], ["art"])
        smoke_targets = workflow.selected_splits(self.smoke)["erase"]
        self.assertEqual(workflow.training_config(self.smoke, workflow.METHODS[2], smoke_targets)["residual_rank"], 3)

    def test_method_yaml_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "workflow.yaml").write_text((SCRIPT.parent / "workflow.yaml").read_text())
            for method, filename in self.config["training"].items():
                train = dict(self.config["_training_configs"][method])
                if method == "legacy":
                    train["threshold"] = 0.2
                (directory / filename).write_text(yaml.safe_dump(train))
            with self.assertRaisesRegex(ValueError, "Training settings differ"):
                workflow.load_config(directory / "workflow.yaml")

    def test_config_change_changes_run_identity(self):
        changed = workflow.load_config(SCRIPT.parent / "workflow.yaml")
        original = workflow.run_id(changed)
        changed["experiment"]["mscoco_prompts"] = 999
        self.assertNotEqual(original, workflow.run_id(changed))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "style.csv"
            path.write_text("id,concept\n1,A\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.style_splits(path)


    def test_resume_requires_exact_filenames_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            manifest.write_text("group,artist_id,artist,prompt,seed,variant,filename\nerase,1,A,p,0,0,one.png\n")
            identity = workflow.sampling_identity(root, self.config, "original")
            manifest.write_text(manifest.read_text() + "retain,2,B,p,0,0,two.png\n")
            self.assertNotEqual(identity, workflow.sampling_identity(root, self.config, "original"))
            (root / "erase").mkdir()
            (root / "erase" / "wrong.png").write_bytes(b"x")
            with self.assertRaises(ValueError):
                workflow.validate_image_set(root, [{"group": "erase", "filename": "one.png"}], "erase")

    def test_artist_macro_aggregation_and_delta(self):
        records = []
        for model, erase, retain in (("original", [10, 30], [40, 60]),
                                     ("legacy", [5, 15], [35, 55]),
                                     (workflow.METHODS[2], [4, 12], [38, 58])):
            records.extend({"model": model, "group": group, "score": score}
                           for group, scores in (("erase", erase), ("retain", retain)) for score in scores)
        result = workflow.aggregate_scores(records, 100)
        self.assertEqual((result[0]["clip_e"], result[0]["clip_s"], result[0]["h_a"]), (20, 50, 30))
        self.assertEqual(result[1]["delta_clip_e_vs_original"], -10)
        self.assertEqual(result[2]["h_a"], 40)
        self.assertLessEqual(result[0]["h_a_ci_low"], result[0]["h_a_ci_high"])


if __name__ == "__main__":
    unittest.main()
