"""Focused contracts on invented inputs; guards exist only in the test process."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import ROOT, guard, predictions


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.addaudithook(guard)
        from hebrew_acronyms import experimental_study
        cls.study = experimental_study

    def setUp(self):
        self.rows = self.study.fixture_rows()
        self.settings = {"qwen_model": "invented-qwen-tag", "qwen_revision": "invented-immutable-version"}
        self.origin = {"weights_sha256": "a" * 64, "manifest_sha256": "b" * 64}
        self.artifact = self.study.new_artifact(self.rows, "run-one", self.settings, {"fixture": True})

    def test_all_notebook_cells_preview_mocked_run_reload_and_saved_encoder(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.run_experimental_study"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-3000:])

    def test_generation_target_and_sentence_match_selection_without_candidate_leakage(self):
        generated, selected = [], []
        self.study.collect_responses(self.artifact, "generate", lambda prompt: generated.append(prompt) or "raw")
        self.study.collect_responses(self.artifact, "select", lambda prompt: selected.append(prompt) or "A or B")
        for row, generate, select in zip(self.rows, generated, selected):
            for prompt in (generate, select):
                self.assertIn(row["sentence"], prompt)
                self.assertIn('"' + row["target_raw"] + '"', prompt)
            for candidate in row["candidates"].split("|"):
                self.assertNotIn(candidate, generate)
                self.assertIn(candidate, select)
        for record in self.artifact["records"]:
            self.assertEqual(record["score_status"], "unscored")
            self.assertNotIn("correct", record)
        self.assertEqual(len(selected), 2)  # singleton retained
        self.assertIsNone(self.artifact["records"][2]["shown_order"])

    def test_generation_needs_no_candidates_or_gold(self):
        for row in self.rows:
            row.pop("candidates"); row.pop("gold_expansion")
        artifact = self.study.new_artifact(self.rows, "no-gold", self.settings, {})
        backend = Mock(return_value="unscored expansion")
        self.study.collect_responses(artifact, "generate", backend)
        self.assertEqual(backend.call_count, 2)

    def test_failures_and_invalid_inputs_retain_every_record(self):
        self.rows[0]["span_start"] = 0
        self.rows[1]["candidates"] = "|".join(f"candidate-{i}" for i in range(27))
        artifact = self.study.new_artifact(self.rows, "bad-inputs", self.settings, {})
        backend = Mock(return_value="never")
        self.study.collect_responses(artifact, "select", backend)
        backend.assert_not_called()
        self.assertEqual([r["status"] for r in artifact["records"][-2:]], ["invalid_input", "invalid_input"])
        self.assertEqual(len(artifact["records"]), 6)
        self.study.collect_responses(self.artifact, "generate", Mock(side_effect=[RuntimeError("fixture failure"), "retained raw"]))
        records = self.artifact["records"][2:4]
        self.assertEqual(records[0]["status"], "service_error")
        self.assertIn("fixture failure", records[0]["error"])
        self.assertEqual(records[1]["raw_response"], "retained raw")

    def test_interrupt_saves_current_and_unattempted_status(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            path = self.study.save_artifact(self.artifact, folder, create=True)
            with self.assertRaises(KeyboardInterrupt):
                self.study.collect_responses(self.artifact, "generate", Mock(side_effect=KeyboardInterrupt),
                    on_record=lambda artifact: self.study.save_artifact(artifact, folder))
            saved = self.study.load_artifact(path, expected_run_id="run-one")
            self.assertEqual([r["status"] for r in saved["records"][2:4]], ["interrupted", "not_run"])

    def test_encoder_joins_by_id_and_rejects_dropped_duplicate_and_extra_ids(self):
        source = predictions(self.rows)[::-1]
        self.study.attach_encoder(self.artifact, source, origin=self.origin)
        self.assertEqual([r["item_id"] for r in self.artifact["records"][:2]], [row["item_id"] for row in self.rows])
        invalid = [source[:1], [source[0], source[0]], source + [dict(source[0], item_id="foreign")]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                fresh = self.study.new_artifact(self.rows, "fresh", self.settings, {})
                self.study.attach_encoder(fresh, records, origin=self.origin)

    def test_encoder_failures_and_inventory_validation(self):
        source = predictions(self.rows)
        source[1].update(status="model_error", selected_candidate=None, selected_index=None,
                         candidate_scores=[], error="invented forward failure")
        self.study.attach_encoder(self.artifact, source, origin=self.origin)
        self.assertEqual(self.artifact["records"][1]["status"], "model_error")
        for key, value in (("selected_candidate", "foreign"), ("selected_index", 99), ("candidate_scores", [])):
            bad = deepcopy(self.artifact); bad["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.study.validate_artifact(bad, expected_run_id="run-one")

    def test_artifact_roundtrip_and_strict_identity(self):
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        self.study.collect_responses(self.artifact, "select", lambda prompt: "A and B")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            path = self.study.save_artifact(self.artifact, folder, create=True)
            self.assertEqual(self.study.load_artifact(path, expected_run_id="run-one", rows=self.rows), self.artifact)
            for field, value in (("sentence", "different sentence"), ("candidates", "בית מדרש|בית מלאכה")):
                changed = deepcopy(self.rows); changed[0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.study.load_artifact(path, expected_run_id="run-one", rows=changed)
            with self.assertRaises(ValueError):
                self.study.load_artifact(path, expected_run_id="another-run")
            with self.assertRaises(FileExistsError):
                self.study.save_artifact(self.artifact, folder, create=True)
            changed = deepcopy(self.artifact); changed["settings"]["device"] = "changed"
            with self.assertRaises(ValueError):
                self.study.save_artifact(changed, folder)

    def test_mixed_records_model_settings_provenance_and_prompt_rejected(self):
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        self.study.collect_responses(self.artifact, "generate", lambda prompt: "raw")
        changes = [(0, "run_id", "another"), (0, "input_sha256", "changed"),
                   (0, "encoder_origin", None), (0, "encoder_origin", dict(self.origin, weights_sha256="c"*64)),
                   (2, "model", "other-model"), (2, "model_revision", "other-version"),
                   (2, "prompt", "foreign prompt"), (2, "shown_order", ["foreign candidate"])]
        for index, field, value in changes:
            bad = deepcopy(self.artifact); bad["records"][index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.validate_artifact(bad, expected_run_id="run-one")
        for records in (self.artifact["records"][:-1], self.artifact["records"] + [self.artifact["records"][0]]):
            with self.assertRaises(ValueError):
                self.study.validate_artifact(dict(self.artifact, records=records), expected_run_id="run-one")

    def test_saved_encoder_requires_bound_identity_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "run"
            path = self.study.save_artifact(self.artifact, folder, create=True)
            with self.assertRaises(ValueError):
                self.study.saved_encoder_predictions(path, expected_run_id="run-one", rows=self.rows)
            self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
            self.study.save_artifact(self.artifact, folder)
            saved, origin = self.study.saved_encoder_predictions(path, expected_run_id="run-one", rows=self.rows)
            new = self.study.new_artifact(self.rows, "run-two", self.settings, {})
            self.study.attach_encoder(new, saved, origin=origin)
            self.assertEqual(new["records"][0]["encoder_origin"]["reused_from"]["source_run_id"], "run-one")
            path.write_text(json.dumps(predictions(self.rows)))
            with self.assertRaises(ValueError):
                self.study.load_artifact(path, expected_run_id="run-one")

    def test_missing_checkpoint_json_and_unsafe_output_fail_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); input_path = folder / "invented.csv"; input_path.write_text("fixture")
            arguments = dict(root=ROOT, input_path=input_path, checkpoint=None, output_dir=folder / "new",
                             run=True, encoder=True, llm=False, model=None, revision=None)
            with self.assertRaises(FileNotFoundError):
                self.study.check_readiness(**arguments)
            checkpoint = folder / "fixture.pt";checkpoint.write_text("fixture")
            arguments["checkpoint"] = checkpoint
            with self.assertRaises(FileNotFoundError):
                self.study.check_readiness(**arguments)
            Path(str(checkpoint)+".json").write_text("{}")
            self.study.check_readiness(**arguments)
            for output in (ROOT / "outputs", ROOT.parent / "hebrew-acronym-disambiguation" / "ignored-output"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    self.study.check_readiness(**dict(arguments, output_dir=output))
            with self.assertRaises(ValueError):
                self.study.check_readiness(**dict(arguments, llm=True))

    def test_guards_forbid_network_and_research_reads_only_in_test_process(self):
        import socket
        with self.assertRaisesRegex(RuntimeError, "Network forbidden"):
            socket.socket()
        with self.assertRaisesRegex(RuntimeError, "Research I/O forbidden"):
            (ROOT / "data" / "never-open.csv").read_text()
        with patch("sys.addaudithook") as hook:
            self.study.fixture_rows()
            self.study.new_artifact(self.rows, "id", self.settings, {})
        hook.assert_not_called()

    def test_notebook_binds_qwen_model_across_interactive_configuration_changes(self):
        notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
        backend = Mock(return_value="raw fixture")
        namespace = {"MODE": "run", "ENABLE_LLM": True, "artifact": self.artifact,
                     "QWEN_MODEL": "mutable-global", "ollama_generate": backend, "study": self.study,
                     "OUTPUT_DIR": "unused-test-path"}
        with patch.object(self.study, "save_artifact"):
            exec("".join(notebook["cells"][9]["source"]), namespace)
            namespace["QWEN_MODEL"] = "different-global"
            exec("".join(notebook["cells"][11]["source"]), namespace)
        self.assertEqual(backend.call_count, 4)
        self.assertEqual({call.kwargs["model"] for call in backend.call_args_list}, {"invented-qwen-tag"})

    def test_changed_intermediate_rows_rejected_before_encoder_load(self):
        changed_rows = deepcopy(self.rows)
        changed_rows[0]["sentence"] = "different sentence with the same item ID"
        notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
        loader = Mock()
        namespace = {"MODE": "run", "ENABLE_ENCODER": True, "SAVED_ENCODER": None,
                     "artifact": self.artifact, "load_finetuned": loader,
                     "study": self.study, "RUN_ID": "run-one", "rows": changed_rows}
        with self.assertRaisesRegex(ValueError, "input identity differs"):
            exec("".join(notebook["cells"][7]["source"]), namespace)
        loader.assert_not_called()

    def test_notebook_rejects_repeated_encoder_before_inference(self):
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        notebook = json.loads((ROOT / "notebooks/experimental_study.ipynb").read_text())
        loader = Mock()
        namespace = {"MODE": "run", "ENABLE_ENCODER": True, "SAVED_ENCODER": None,
                     "artifact": self.artifact, "load_finetuned": loader,
                     "study": self.study, "RUN_ID": "run-one", "rows": self.rows}
        with self.assertRaisesRegex(ValueError, "already exist"):
            exec("".join(notebook["cells"][7]["source"]), namespace)
        loader.assert_not_called()

    def test_no_shared_historical_scorer_is_called_or_repeated_responses_overwritten(self):
        with (patch.object(self.study.prompts, "evaluate", side_effect=AssertionError("historical scorer")),
              patch.object(self.study.prompts, "parse_letter_choice", side_effect=AssertionError("historical parser"))):
            self.study.collect_responses(self.artifact, "select", lambda prompt: "Ambiguous A or B")
        backend = Mock()
        with self.assertRaises(ValueError):
            self.study.collect_responses(self.artifact, "select", backend)
        backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
