import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "remote_scripts/eval_100_ins/workflow.py"
SPEC = importlib.util.spec_from_file_location("eval_100_ins_workflow", SCRIPT)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class Instance100WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = workflow.load_config(SCRIPT.parent / "workflow.yaml")
        cls.smoke = workflow.load_config(SCRIPT.parent / "workflow_smoke.yaml")

    def test_committed_splits_normalize_names_and_exclude_targets(self):
        erase, retain_eval, retain_train = workflow.validate_committed_splits()
        self.assertEqual(tuple(map(len, (erase, retain_eval, retain_train))), (100, 100, 1234))
        self.assertEqual([int(row["id"]) for row in erase], list(range(1, 101)))
        self.assertEqual([int(row["id"]) for row in retain_eval], list(range(101, 201)))
        self.assertEqual(erase[5]["concept"], "Pink Panther")
        names = [row["concept"].casefold() for row in retain_train]
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(set(names) & {row["concept"].casefold() for row in erase})
        self.assertTrue({row["concept"].casefold() for row in retain_eval} <= set(names))
        self.assertEqual([row["id"] for row in retain_train if row["concept"] == "Groot"], ["145"])

    def test_source_id_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "instance.csv"
            path.write_text("id,concept\n1,Something\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.instance_splits(path)

    def test_full_and_smoke_manifests_have_paired_prompts(self):
        coco = workflow.read_csv(ROOT / "data/mscoco.csv")
        rows = workflow.build_manifest(self.config, workflow.selected_splits(self.config), coco)
        self.assertEqual({group: sum(row["group"] == group for row in rows)
                          for group in ("erase", "retain", "coco")},
                         {"erase": 200, "retain": 200, "coco": 100})
        self.assertEqual(len(rows) * len(workflow.METHODS), 1500)
        first = [row for row in rows if row["instance_id"] == 1]
        self.assertEqual([row["prompt"] for row in first],
                         ["An image of Crazy Frog.", "An illustration of Crazy Frog."])
        self.assertEqual({row["seed"] for row in rows}, {0})
        self.assertEqual(len({(row["group"], row["filename"]) for row in rows}), len(rows))
        smoke_rows = workflow.build_manifest(self.smoke, workflow.selected_splits(self.smoke), coco)
        self.assertEqual(len(smoke_rows), 20)

    def test_training_settings_and_smoke_rank(self):
        targets = workflow.selected_splits(self.config)["erase"]
        speed = workflow.training_config(self.config, "legacy", targets)
        ours = workflow.training_config(self.config, workflow.METHODS[2], targets)
        for key in ("target_concepts", "anchor_concepts", "retain_path", "erase_style", "params",
                    "threshold", "retain_scale", "seed"):
            self.assertEqual(speed[key], ours[key])
        self.assertEqual(speed["anchor_concepts"], "")
        self.assertEqual((speed["aug_num"], ours["aug_num"]), (10, 0))
        self.assertEqual(ours["subspace_anchor_concepts"], [""])
        self.assertEqual(ours["residual_rank"], 30)
        smoke_targets = workflow.selected_splits(self.smoke)["erase"]
        self.assertEqual(workflow.training_config(self.smoke, workflow.METHODS[2], smoke_targets)["residual_rank"], 3)

    def test_config_and_filename_changes_invalidate_resume(self):
        changed = workflow.load_config(SCRIPT.parent / "workflow.yaml")
        old_id = workflow.run_id(changed)
        changed["experiment"]["mscoco_prompts"] = 99
        self.assertNotEqual(old_id, workflow.run_id(changed))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "erase").mkdir()
            (root / "erase" / "unexpected.png").write_bytes(b"x")
            with self.assertRaises(ValueError):
                workflow.validate_image_set(root, [{"group": "erase", "filename": "expected.png"}], "erase")

    def test_macro_average_bootstraps_instances_and_labels_models(self):
        rows = []
        for method, erase, retain in (("original", (10, 30), (40, 60)),
                                      ("legacy", (5, 15), (35, 55)),
                                      (workflow.METHODS[2], (4, 12), (38, 58))):
            rows.extend({"method": method, "group": group, "clip_score": score}
                        for group, scores in (("erase", erase), ("retain", retain)) for score in scores)
        result = workflow.aggregate_scores(rows, 200)
        self.assertEqual([row["model"] for row in result], ["Original SD v1.4", "SPEED", "Ours"])
        self.assertEqual((result[0]["erase_clip"], result[0]["retain_clip"]), (20, 50))
        self.assertEqual(result[1]["delta_erase_clip_vs_original"], -10)
        self.assertEqual(result[2]["erase_clip"], 8)
        self.assertLessEqual(result[0]["erase_ci_low"], result[0]["erase_ci_high"])


if __name__ == "__main__":
    unittest.main()
