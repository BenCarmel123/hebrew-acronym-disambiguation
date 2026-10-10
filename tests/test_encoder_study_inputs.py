"""Checkpoint input matching on invented CSVs; never load a real encoder."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.run_experimental_study import disable_guard, enable_guard, predictions


class EncoderInputTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        disable_guard()

    @classmethod
    def setUpClass(cls):
        enable_guard()
        from hebrew_acronyms import experimental_study
        from hebrew_acronyms.models.dictabert_cross_encoder import model, eval
        cls.study, cls.model, cls.eval = experimental_study, model, eval

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="encoder-input-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rows = self.study.fixture_rows()
        rows += [dict(row, item_id=row["item_id"] + "-copy") for row in rows]
        self.dev = self.write_csv("dev.csv", rows)
        self.train = self.write_csv("train.csv", [dict(row, item_id="train-" + row["item_id"]) for row in rows])
        self.expected = {split: self.study.input_identity(self.study.load_rows(path))
                         for split, path in (("train", self.train), ("dev", self.dev))}
        self.metadata = {"inputs": deepcopy(self.expected), "training_config": {"epochs": 2},
                         "selection": {"epoch": 1}}
        self.checkpoint = self.folder / "mock.pt"
        self.checkpoint.write_text("mock, never loaded")
        self.sidecar = Path(str(self.checkpoint) + ".json")
        self.sidecar.write_text(json.dumps(self.metadata))
        self.settings = {"mode": "run", "run_kind": "validation", "validation_items": 3,
                         "train_path": str(self.train), "input_path": str(self.dev), "expected_dev_items": 4,
                         "output_root": str(self.folder)}
        self.artifact, self.rows = self.fresh("current")

    def write_csv(self, name, rows):
        path = self.folder / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def fresh(self, run_id):
        rows, cohort = self.study.read_study_inputs(self.settings)
        artifact = self.study.new_artifact(rows, run_id, self.settings, {})
        artifact["cohort"] = cohort
        return artifact, rows

    def predict(self, artifact=None, rows=None, **kwargs):
        with (patch.object(self.model, "load_finetuned", return_value=(
                object(), SimpleNamespace(checkpoint_metadata=deepcopy(self.metadata)), 1, 2)) as load,
              patch.object(self.eval, "evaluate", side_effect=lambda rows, *a: predictions(rows)) as evaluate,
              patch.object(self.study, "save_study")):
            result = self.study.predict_encoder(artifact or self.artifact, rows or self.rows,
                                               checkpoint=self.checkpoint, **kwargs)
        return result, load, evaluate

    def test_validation_passes_train_and_full_dev_to_strict_loader(self):
        _, load, evaluate = self.predict()
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(load.call_args.kwargs["expected_inputs"], self.expected)
        self.assertEqual(len(load.call_args.kwargs["expected_inputs"]["dev"]["item_ids"]), 4)
        evaluate.assert_called_once()
        metadata, summary = self.study.checkpoint_report(self.artifact)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(summary["train_items"], 4)
        self.assertEqual(summary["dev_items"], 4)
        self.assertEqual(summary["selected_epoch"], 1)
        self.assertIn("matched", summary["input_match"])
        self.assertNotIn("inputs", summary)

    def test_checkpoint_train_or_dev_mismatch_stops_before_loading(self):
        for split in ("train", "dev"):
            bad = deepcopy(self.metadata)
            bad["inputs"][split]["sha256"] = "different"
            self.sidecar.write_text(json.dumps(bad))
            with (self.subTest(split=split), patch.object(self.model, "load_finetuned") as load,
                  patch.object(self.eval, "evaluate") as evaluate,
                  self.assertRaisesRegex(ValueError, f"Checkpoint {split} input differs")):
                self.study.predict_encoder(self.artifact, self.rows, checkpoint=self.checkpoint)
            load.assert_not_called()
            evaluate.assert_not_called()
            self.assertEqual(json.loads(self.sidecar.read_text()), bad)

    def test_dev_outside_validation_subset_cannot_change_after_prepare(self):
        changed = self.study.load_rows(self.dev)
        changed[-1]["sentence"] += " altered"
        self.write_csv("dev.csv", changed)
        with patch.object(self.model, "load_finetuned") as load, self.assertRaisesRegex(ValueError, "dev changed"):
            self.study.predict_encoder(self.artifact, self.rows, checkpoint=self.checkpoint)
        load.assert_not_called()

    def test_changed_current_sources_are_compared_with_original_manifest(self):
        for split, path in (("train", self.train), ("dev", self.dev)):
            original = path.read_text()
            changed = self.study.load_rows(path)
            changed[-1]["sentence"] += " changed"
            self.write_csv(path.name, changed)
            artifact, rows = self.fresh("changed-" + split)
            with patch.object(self.model, "load_finetuned") as load, self.assertRaisesRegex(ValueError, f"Checkpoint {split} input differs"):
                self.study.predict_encoder(artifact, rows, checkpoint=self.checkpoint)
            load.assert_not_called()
            path.write_text(original)

    def test_missing_original_input_identity_is_not_inferred(self):
        for bad in ({}, {"inputs": {"train": self.expected["train"]}}):
            self.sidecar.write_text(json.dumps(bad))
            with patch.object(self.model, "load_finetuned") as load, self.assertRaisesRegex(ValueError, "lacks"):
                self.study.predict_encoder(self.artifact, self.rows, checkpoint=self.checkpoint)
            load.assert_not_called()

    def saved_source(self):
        origin = {"weights_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                  "checkpoint_metadata": deepcopy(self.metadata)}
        source, rows = self.fresh("source")
        self.study.attach_encoder(source, predictions(rows), origin=origin)
        return self.study.save_artifact(source, self.folder / "source", create=True)

    def test_saved_predictions_match_current_full_inputs_without_model_load(self):
        path = self.saved_source()
        _, load, evaluate = self.predict(saved_path=path, saved_run_id="source")
        load.assert_not_called()
        evaluate.assert_not_called()
        origin = self.artifact["records"][0]["encoder_origin"]
        self.assertEqual(origin["expected_inputs"], self.expected)
        self.assertEqual(origin["reused_from"]["source_run_id"], "source")

    def test_saved_predictions_reject_changed_train_or_unselected_dev(self):
        path = self.saved_source()
        for split, source_path in (("train", self.train), ("dev", self.dev)):
            original = source_path.read_text()
            changed = self.study.load_rows(source_path)
            changed[-1]["sentence"] += " changed"
            self.write_csv(source_path.name, changed)
            artifact, rows = self.fresh("reuse-" + split)
            with patch.object(self.model, "load_finetuned") as load, self.assertRaisesRegex(ValueError, f"Checkpoint {split} input differs"):
                self.study.predict_encoder(artifact, rows, checkpoint=None, saved_path=path, saved_run_id="source")
            load.assert_not_called()
            self.assertTrue(all(r["status"] == "not_run" for r in artifact["records"]))
            source_path.write_text(original)

    def test_reload_and_summary_need_no_research_inputs(self):
        path = self.saved_source()
        before = path.read_bytes()
        self.train.unlink()
        self.dev.unlink()
        with patch.object(self.study, "load_rows", side_effect=AssertionError("research read")):
            loaded = self.study.load_artifact(path, expected_run_id="source")
            metadata, summary = self.study.checkpoint_report(loaded)
            self.assertEqual(metadata, self.metadata)
            self.assertEqual(summary["input_match"], "not checked for this historical run")
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
