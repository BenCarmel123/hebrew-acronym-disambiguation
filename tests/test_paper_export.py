"""Paper export contracts on invented inputs inside dedicated temporary roots."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import disable_guard, enable_guard, predictions


class PaperExportTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        disable_guard()

    @classmethod
    def setUpClass(cls):
        enable_guard()
        from hebrew_acronyms import experimental_study, paper_export
        cls.study, cls.exporter = experimental_study, paper_export

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "generated"
        rows = self.study.fixture_rows()
        self.artifact = self.study.new_artifact(rows, "fixture-first",
            {"mode": "run", "run_kind": "full_dev", "qwen_model": "fixture-qwen", "qwen_revision": "fixture-digest"},
            {"fixture": True, "git_head": "invented-revision"})
        self.artifact["llm_runtimes"] = {"qwen": {"model": "fixture-qwen", "digest": "fixture-digest"}}
        self.artifact["cohort"] = {"kind": "full_dev", "full_dev": True,
            "available_items": len(rows), "requested_items": len(rows),
            "full_input_identity": deepcopy(self.artifact["input_identity"])}
        self.study.attach_encoder(self.artifact, predictions(rows),
            origin={"weights_sha256": "a" * 64, "manifest_sha256": "b" * 64})

    def export(self, artifact=None, **kwargs):
        return self.exporter.export_paper(artifact or self.artifact,
            expected_run_id=self.artifact["run_id"], paper_source_run_id=self.artifact["run_id"],
            output_dir=self.output, fixture_root=self.root, **kwargs)

    def bundle(self):
        manifest = self.exporter.verify_paper_bundle(self.output)
        digest = self.exporter._sha(self.exporter._json_bytes(manifest))
        return self.output / "bundles" / digest

    def test_shared_scores_unjudged_generation_and_failure_denominators(self):
        self.study.collect_responses(self.artifact, "select", Mock(side_effect=RuntimeError("invented outage")))
        manifest = self.export()
        shared = self.study.inspect_results(self.artifact)["metrics"]
        summaries = {s["condition"]: s for s in manifest["summaries"]}
        for score in shared:
            for key in ("micro_accuracy", "macro_accuracy", "n_failures", "status_counts", "execution_status"):
                self.assertEqual(summaries[score["condition"]][key], score[key])
        self.assertEqual(summaries["qwen_select"]["micro_accuracy"], 0)
        self.assertEqual(summaries["qwen_select"]["status_counts"], {"service_error": 2})
        self.assertEqual(summaries["qwen_select"]["execution_status"], "completed")
        self.assertIsNone(summaries["gemini_select"]["micro_accuracy"])
        self.assertIsNone(summaries["qwen_generate"]["micro_accuracy"])
        self.assertEqual(manifest["execution_status"], "partial")
        table = (self.bundle() / "table.tex").read_text()
        self.assertIn("Qwen generation & 0/2 & 0 & -- & -- & not run", table)
        self.assertIn("Generation is unjudged", table)
        self.assertTrue((self.bundle() / "selection_accuracy.pdf").read_bytes().startswith(b"%PDF"))

    def test_replacement_updates_table_figure_and_macros_from_one_snapshot(self):
        first = self.export()
        previous = self.bundle()
        changed = deepcopy(self.artifact)
        changed["run_id"] = "fixture-second"
        for record in changed["records"]:
            record["run_id"] = changed["run_id"]
        record = changed["records"][0]
        record["selected_index"] = 1
        record["selected_candidate"] = changed["items"][0]["candidates"].split("|")[1]
        self.artifact = changed
        second = self.export()
        current = self.bundle()
        self.assertNotEqual(previous, current)
        self.assertEqual(first["input_identity"], second["input_identity"])
        self.assertNotEqual(first["source_artifact_sha256"], second["source_artifact_sha256"])
        for name in self.exporter.OUTPUT_FILES:
            self.assertNotEqual((previous / name).read_bytes(), (current / name).read_bytes(), name)
        self.assertIn("50.0", (current / "table.tex").read_text())
        self.assertIn("fixture-second", (current / "macros.tex").read_text())
        pointer = (self.output / "current.tex").read_text()
        self.assertNotIn(previous.name, pointer)
        self.assertEqual(pointer.count(current.name), 3)

    def test_saved_reload_has_same_scores_and_exact_source_file_hash(self):
        saved = self.study.save_artifact(self.artifact, self.root / "run", create=True)
        original_bytes = saved.read_bytes()
        manifest = self.export(saved)
        self.assertEqual(manifest["source_file_sha256"], self.exporter._sha(original_bytes))
        self.assertEqual(manifest["summaries"], self.exporter._summaries(self.artifact))
        self.assertEqual(saved.read_bytes(), original_bytes)

    def test_preview_validation_fixture_and_wrong_source_cannot_replace(self):
        self.export()
        before = (self.output / "current.tex").read_bytes()
        for key, value in (("mode", "preview"), ("run_kind", "validation")):
            changed = deepcopy(self.artifact)
            changed["settings"][key] = value
            with self.assertRaises(ValueError):
                self.export(changed)
        with self.assertRaisesRegex(ValueError, "fixtures cannot"):
            self.exporter.export_paper(self.artifact, expected_run_id=self.artifact["run_id"],
                paper_source_run_id=self.artifact["run_id"], output_dir=self.output)
        with self.assertRaisesRegex(ValueError, "explicit paper-source"):
            self.exporter.export_paper(self.artifact, expected_run_id=self.artifact["run_id"],
                paper_source_run_id="other", output_dir=self.output)
        self.assertEqual(before, (self.output / "current.tex").read_bytes())

    def test_fixture_permission_cannot_write_outside_temporary_root(self):
        with self.assertRaisesRegex(ValueError, "temporary root"):
            self.exporter.export_paper(self.artifact, expected_run_id=self.artifact["run_id"],
                paper_source_run_id=self.artifact["run_id"], fixture_root=self.root,
                output_dir=Path(__file__).resolve().parents[1] / "paper/generated")
        self.assertFalse(self.output.exists())

    def test_render_failure_keeps_previous_complete_bundle(self):
        self.export()
        before = (self.output / "current.tex").read_bytes()
        with patch.object(self.exporter, "_render_figure", side_effect=RuntimeError("invented render failure")):
            with self.assertRaisesRegex(RuntimeError, "render failure"):
                self.export()
        self.assertEqual(before, (self.output / "current.tex").read_bytes())
        self.exporter.verify_paper_bundle(self.output)
        self.assertFalse(list((self.output / "bundles").glob(".staging-*")))

    def test_hash_verification_detects_tampering(self):
        self.export()
        (self.bundle() / "table.tex").write_text("stale table")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.exporter.verify_paper_bundle(self.output)

    def test_rejects_changed_input_identity_before_writing(self):
        self.artifact["input_identity"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input identity"):
            self.export()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
