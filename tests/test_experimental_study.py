"""Focused contracts on invented inputs; guards exist only in the test process."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import ROOT, disable_guard, enable_guard, predictions, legacy_artifact


class StudyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        disable_guard()

    @classmethod
    def setUpClass(cls):
        enable_guard()
        from hebrew_acronyms import experimental_study
        cls.study = experimental_study

    def setUp(self):
        self.rows = self.study.fixture_rows()
        self.settings = {"qwen_model": "invented-qwen-tag", "qwen_revision": "invented-immutable-version"}
        self.origin = {"weights_sha256": "a" * 64, "manifest_sha256": "b" * 64}
        self.artifact = legacy_artifact(self.rows, "run-one", self.settings, {"fixture": True})

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
                self.assertIn(self.study.mark_span(row["sentence"], self.study.explicit_span(row)), prompt)
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
        artifact = legacy_artifact(self.rows, "no-gold", self.settings, {})
        backend = Mock(return_value="unscored expansion")
        self.study.collect_responses(artifact, "generate", backend)
        self.assertEqual(backend.call_count, 2)

    def test_failures_and_invalid_inputs_retain_every_record(self):
        self.rows[0]["span_start"] = 0
        self.rows[1]["candidates"] = "|".join(f"candidate-{i}" for i in range(27))
        artifact = legacy_artifact(self.rows, "bad-inputs", self.settings, {})
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
                fresh = legacy_artifact(self.rows, "fresh", self.settings, {})
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
            new = legacy_artifact(self.rows, "run-two", self.settings, {})
            self.study.attach_encoder(new, saved, origin=origin)
            self.assertEqual(new["records"][0]["encoder_origin"]["reused_from"]["source_run_id"], "run-one")
            path.write_text(json.dumps(predictions(self.rows)))
            with self.assertRaises(ValueError):
                self.study.load_artifact(path, expected_run_id="run-one")

    def test_missing_checkpoint_json_and_unsafe_output_fail_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); input_path = folder / "invented.csv"; input_path.write_text("fixture")
            arguments = dict(root=ROOT, input_path=input_path, checkpoint=None, output_dir=folder / "new",
                             run=True, encoder=True, llm=False, model=None)
            with self.assertRaises(FileNotFoundError):
                self.study.check_readiness(**arguments)
            checkpoint = folder / "fixture.pt";checkpoint.write_text("fixture")
            arguments["checkpoint"] = checkpoint
            with self.assertRaises(FileNotFoundError):
                self.study.check_readiness(**arguments)
            Path(str(checkpoint)+".json").write_text("{}")
            self.study.check_readiness(**arguments)
            # Another Git working tree, created here instead of assuming a sibling checkout.
            (folder / "other-checkout" / ".git").mkdir(parents=True)
            for output in (ROOT / "outputs", folder / "other-checkout" / "ignored-output"):
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
            legacy_artifact(self.rows, "id", self.settings, {})
        hook.assert_not_called()

    def test_qwen_callable_binds_resolved_model_digest_and_request_settings(self):
        from tests.run_experimental_study import runtime, response
        from hebrew_acronyms.models.qwen import eval as backend
        self.artifact["settings"].update(ollama_url="http://localhost:11434", request_timeout=90, qwen_options={"temperature": 0})
        with (patch.object(backend, "inspect_ollama", side_effect=runtime),
              patch.object(backend, "ollama_response", side_effect=response) as generate,
              patch.object(self.study, "save_study")):
            bound = self.study.connect_qwen(self.artifact)
            unrelated_settings = dict(self.artifact["settings"], qwen_model="different-tag")
            bound("first prompt"); bound("second prompt")
        self.assertEqual(generate.call_count, 2)
        self.assertEqual({call.kwargs["model"] for call in generate.call_args_list}, {"invented-qwen-tag"})
        self.assertEqual({call.kwargs["expected_digest"] for call in generate.call_args_list}, {"fixture-digest"})

    def test_changed_intermediate_rows_and_settings_rejected_before_encoder_load(self):
        from hebrew_acronyms.models.dictabert_cross_encoder import model
        changed_rows = deepcopy(self.rows)
        changed_rows[0]["sentence"] = "different sentence with the same item ID"
        with patch.object(model, "load_finetuned") as loader:
            with self.assertRaisesRegex(ValueError, "input identity differs"):
                self.study.predict_encoder(self.artifact, changed_rows, checkpoint=None)
            self.artifact["settings"]["checkpoint"] = "original.pt"
            with self.assertRaisesRegex(ValueError, "setting checkpoint"):
                self.study.predict_encoder(self.artifact, self.rows, checkpoint="other.pt")
        loader.assert_not_called()

    def test_repeated_encoder_rejected_before_inference(self):
        from hebrew_acronyms.models.dictabert_cross_encoder import model
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.study.attach_encoder(self.artifact, predictions(self.rows), origin=self.origin)
        with patch.object(model, "load_finetuned") as loader:
            with self.assertRaisesRegex(ValueError, "already exist"):
                self.study.predict_encoder(self.artifact, self.rows, checkpoint=None)
        loader.assert_not_called()

    def test_two_identical_occurrences_produce_different_marked_prompts(self):
        sentence = "א״ב וגם א״ב"
        rows = [dict(self.rows[0], item_id="first", sentence=sentence, target_raw="א״ב", span_start=0, span_end=3),
                dict(self.rows[0], item_id="second", sentence=sentence, target_raw="א״ב", span_start=8, span_end=11)]
        artifact = legacy_artifact(rows, "two-occurrences", self.settings, {})
        self.study.collect_responses(artifact, "generate", lambda prompt: "raw")
        self.study.collect_responses(artifact, "select", lambda prompt: "A")
        generated = [record for record in artifact["records"] if record["condition"] == "generate"]
        selected = [record for record in artifact["records"] if record["condition"] == "select"]
        self.assertNotEqual(generated[0]["prompt_sentence"], generated[1]["prompt_sentence"])
        for generate, select in zip(generated, selected):
            self.assertEqual(generate["prompt_sentence"], select["prompt_sentence"])
            self.assertEqual(generate["sentence"], sentence)
            self.assertIn(generate["prompt_sentence"], generate["prompt"])

    def test_historical_prompt_reload_independent_of_current_builder(self):
        self.study.collect_responses(self.artifact, "generate", lambda prompt: "raw")
        # Simulate a format_version=1 artifact predating marked-prompt/hash fields.
        for record in self.artifact["records"]:
            if record["condition"] == "generate":
                record.pop("prompt_sentence"); record.pop("prompt_sha256")
                record["prompt"] = "historical unmarked wording"
        with tempfile.TemporaryDirectory() as directory:
            path = self.study.save_artifact(self.artifact, Path(directory)/"run", create=True)
            with (patch.object(self.study.prompts, "build_generate_prompt", side_effect=AssertionError("current builder")),
                  patch.object(self.study.prompts, "build_select_prompt", side_effect=AssertionError("current builder"))):
                loaded = self.study.load_artifact(path, expected_run_id="run-one")
                table, results = self.study.display_tables(loaded)
            self.assertEqual(loaded, self.artifact)
            self.assertIn("manual review", table)
            self.assertEqual(results["generation_score_status"], "manual_review_unscored")

    def test_backend_evidence_validation_and_identity_error_retains_raw(self):
        from tests.run_experimental_study import response
        self.artifact["llm_runtime"] = {"model": self.settings["qwen_model"], "digest": "digest", "options": {}}
        result = response("prompt", model=self.settings["qwen_model"], expected_digest="digest",
                          base_url="unused", timeout=10, options={})
        self.study.collect_responses(self.artifact, "generate", lambda prompt: deepcopy(result))
        bad = deepcopy(self.artifact)
        bad["records"][2]["backend_metadata"]["digest_after"] = "other-digest"
        with self.assertRaisesRegex(ValueError, "Backend response evidence"):
            self.study.validate_artifact(bad, expected_run_id="run-one")
        new = legacy_artifact(self.rows, "identity-failed", self.settings, {})
        new["llm_runtime"] = deepcopy(self.artifact["llm_runtime"])
        failed = dict(result, identity_status="unverified", digest_after="changed", error="changed digest")
        self.study.collect_responses(new, "select", lambda prompt: failed)
        self.assertEqual(new["records"][-1]["status"], "model_identity_error")
        self.assertEqual(new["records"][-1]["raw_response"], result["response"])
        self.study.validate_artifact(new, expected_run_id="identity-failed")

    def test_attempted_selection_mapping_and_full_dev_scope_cannot_be_dropped_or_faked(self):
        self.study.collect_responses(self.artifact, "select", lambda prompt: "A")
        bad = deepcopy(self.artifact)
        bad["records"][-1]["shown_order"] = None
        with self.assertRaisesRegex(ValueError, "letter mapping"):
            self.study.validate_artifact(bad, expected_run_id="run-one")
        identity = deepcopy(self.artifact["input_identity"])
        self.artifact["cohort"] = {"kind": "full_dev", "requested_items": 2, "available_items": 2,
                                   "full_dev": True, "full_input_identity": identity}
        self.study.validate_artifact(self.artifact, expected_run_id="run-one")
        for field, value in (("requested_items", 62), ("available_items", 62), ("full_dev", False),
                             ("full_input_identity", dict(identity, sha256="foreign"))):
            bad = deepcopy(self.artifact);bad["cohort"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.validate_artifact(bad, expected_run_id="run-one")

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
