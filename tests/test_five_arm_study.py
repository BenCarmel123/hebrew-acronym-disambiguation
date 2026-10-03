"""Five-arm integration on invented inputs and mocked provider replies only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import guard, predictions, response as qwen_reply


class FiveArmStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.addaudithook(guard)
        from hebrew_acronyms import experimental_study
        cls.study = experimental_study

    def setUp(self):
        self.rows = self.study.fixture_rows()
        self.settings = {"qwen_model": "qwen2.5:7b", "gemini_model": "gemini-3.8-flash",
                         "request_timeout": 120, "gemini_generation_config": {"thinkingConfig": {"thinkingLevel": "low"}}}
        self.artifact = self.study.new_artifact(self.rows, "five", self.settings, {})
        self.request = {"generationConfig": self.settings["gemini_generation_config"], "timeout_seconds": 120, "max_attempts": 1}
        self.artifact["llm_runtimes"] = {
            "qwen": {"model": "qwen2.5:7b", "digest": "fixture-digest", "options": {}, "timeout_seconds": 120},
            "gemini": {"requested_model": "gemini-3.8-flash", "model_version": None, "request_settings": self.request}}

    def qwen(self, prompt):
        return qwen_reply(prompt, model="qwen2.5:7b", expected_digest="fixture-digest", base_url="unused", timeout=120, options={})

    def gemini(self, prompt, *, version="gemini-fixture-v1", status="response_received", answer="A"):
        return {"response": answer, "status": status, "requested_model": "gemini-3.8-flash",
                "model_version": version, "request_settings": deepcopy(self.request), "finish_reason": "STOP",
                "usage_metadata": {"totalTokenCount": 3}, "error": None}

    def test_five_arms_same_prompts_and_orders_no_gold_leakage_and_roundtrip(self):
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin={"weights_sha256": "a"*64, "manifest_sha256": "b"*64})
        for system, backend in (("qwen", self.qwen), ("gemini", self.gemini)):
            for task in ("generate", "select"):
                self.study.collect_responses(self.artifact, task, backend, system=system)
        self.assertEqual(len(self.artifact["records"]), len(self.rows)*5)
        keyed = {(r["item_id"], r["system"], r["task"]): r for r in self.artifact["records"]}
        for row in self.rows:
            for task in ("generate", "select"):
                q = keyed[(row["item_id"], "qwen", task)]
                g = keyed[(row["item_id"], "gemini", task)]
                self.assertEqual(q["prompt"], g["prompt"])
                self.assertEqual(q["shown_order"], g["shown_order"])
                self.assertEqual(q["sentence"], row["sentence"])
                self.assertIn(q["prompt_sentence"], q["prompt"])
                self.assertNotIn("gold", q["prompt"])
                if task == "generate":
                    for candidate in row["candidates"].split("|"):
                        self.assertNotIn(candidate, q["prompt"])
        with tempfile.TemporaryDirectory() as temp:
            path = self.study.save_artifact(self.artifact, Path(temp)/"five", create=True)
            before = path.read_bytes()
            loaded = self.study.load_artifact(path, expected_run_id="five")
            self.assertEqual(loaded, self.artifact)
            result = self.study.inspect_results(loaded)
            self.assertEqual([metric["system"] for metric in result["metrics"]], ["dictabert", "qwen", "gemini"])
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result["generation_score_status"], "manual_review_unscored")

    def test_missing_changed_version_between_modes_retains_raw_failure(self):
        self.study.collect_responses(self.artifact, "generate", self.gemini, system="gemini")
        self.study.collect_responses(self.artifact, "select", lambda prompt: self.gemini(prompt, version="different"), system="gemini")
        self.study.validate_artifact(self.artifact, expected_run_id="five")
        records = [r for r in self.artifact["records"] if r["condition"] == "gemini_select"]
        self.assertTrue(all(r["status"] == "model_identity_error" and r["raw_response"] == "A" for r in records))
        self.assertEqual(self.study.inspect_results(self.artifact)["metrics"][2]["micro_accuracy"], 0)
        fresh = self.study.new_artifact(self.rows, "missing", self.settings, {})
        fresh["llm_runtimes"] = deepcopy(self.artifact["llm_runtimes"])
        self.study.collect_responses(fresh, "select", lambda prompt: self.gemini(prompt, version=None), system="gemini")
        self.assertTrue(all(r["status"] == "model_identity_error" for r in fresh["records"] if r["condition"] == "gemini_select"))

    def test_known_gemini_micro_macro_include_service_parse_and_unrun_failures(self):
        from hebrew_acronyms.models.common.eval import summarize_selection
        rows = [{"item_id": str(i), "type_id": "large" if i < 4 else "small", "gold_expansion": "gold"} for i in range(5)]
        answers = [("response_received", "B"), ("response_received", "A or B"),
                   ("service_error", None), ("not_run", None), ("response_received", " B ")]
        records = [{"condition": "gemini_select", "item_id": str(i), "status": status,
                    "raw_response": answer, "shown_order": ["other", "gold"]} for i, (status, answer) in enumerate(answers)]
        score = summarize_selection(rows, records, "gemini_select")
        self.assertEqual(score["micro_accuracy"], 2/5)
        self.assertEqual(score["macro_accuracy"], (1/4+1)/2)
        self.assertEqual(score["n_attempted"], 4)
        self.assertEqual(score["n_failures"], 2)
        self.assertEqual(score["execution_status"], "partial")
        for record in records:
            record["status"] = "not_run"
        unrun = summarize_selection(rows, records, "gemini_select")
        self.assertEqual(unrun["execution_status"], "not_run")
        self.assertIsNone(unrun["micro_accuracy"])
        self.assertIsNone(unrun["macro_accuracy"])
        self.assertTrue(all(group["accuracy"] is None for group in unrun["by_type"]))
        self.assertEqual(unrun["n_failures"], 0)

    def test_system_task_input_model_and_runtime_mixing_rejected(self):
        self.study.collect_responses(self.artifact, "select", self.gemini, system="gemini")
        for field, value in (("system", "qwen"), ("task", "generate"), ("input_sha256", "foreign"),
                             ("model", "foreign"), ("model_revision", "foreign")):
            bad = deepcopy(self.artifact)
            bad["records"][-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.study.validate_artifact(bad, expected_run_id="five")
        bad = deepcopy(self.artifact)
        bad["records"][-1]["backend_metadata"]["request_settings"]["generationConfig"] = {}
        with self.assertRaisesRegex(ValueError, "settings"):
            self.study.validate_artifact(bad, expected_run_id="five")
        for system in ("qwen", "gemini"):
            bad = deepcopy(self.artifact)
            if system == "gemini":
                bad["settings"]["gemini_generation_config"] = {}
            else:
                bad["llm_runtimes"]["qwen"]["timeout_seconds"] = 1
            with self.subTest(system=system), self.assertRaisesRegex(ValueError, "runtime settings"):
                self.study.validate_artifact(bad, expected_run_id="five")
        for records in (self.artifact["records"][:-1], self.artifact["records"] + [self.artifact["records"][-1]]):
            with self.assertRaises(ValueError):
                self.study.validate_artifact(dict(self.artifact, records=records), expected_run_id="five")

    def test_cross_provider_order_change_rejected_even_if_inventory_unchanged(self):
        for system, backend in (("qwen", self.qwen), ("gemini", self.gemini)):
            self.study.collect_responses(self.artifact, "select", backend, system=system)
        bad = deepcopy(self.artifact)
        record = next(r for r in bad["records"] if r["condition"] == "gemini_select")
        record["shown_order"].reverse()
        with self.assertRaisesRegex(ValueError, "Providers must share"):
            self.study.validate_artifact(bad, expected_run_id="five")

    def test_unexpected_gemini_exception_never_leaks_key_into_save_or_display(self):
        key = "invented-sensitive-token-ONLY"
        with patch.dict(os.environ, {"GEMINI_API_KEY": key}):
            self.study.collect_responses(self.artifact, "generate", Mock(side_effect=RuntimeError(key)), system="gemini")
            # Even unexpected upstream echo in successful final text/usage is redacted.
            self.study.collect_responses(self.artifact, "select", lambda prompt: self.gemini(prompt, answer=key), system="gemini")
            with tempfile.TemporaryDirectory() as temp:
                path = self.study.save_artifact(self.artifact, Path(temp)/"safe", create=True)
                view, result = self.study.display_tables(self.artifact)
                self.assertNotIn(key, path.read_text())
                self.assertNotIn(key, view)
                self.assertNotIn(key, json.dumps(result))
                self.assertIn("[REDACTED]", path.read_text())

    def test_bounded_stacked_panels_escape_text_and_three_way_disagreements(self):
        self.study.attach_encoder(self.artifact, predictions(self.rows), origin={"weights_sha256": "a"*64, "manifest_sha256": "b"*64})
        self.study.collect_responses(self.artifact, "select", self.qwen, system="qwen")
        self.study.collect_responses(self.artifact, "select", lambda prompt: self.gemini(prompt, answer="B"), system="gemini")
        self.study.collect_responses(self.artifact, "generate", lambda prompt: self.gemini(prompt, answer="<script>bad()</script>"), system="gemini")
        view, result = self.study.display_tables(self.artifact, max_items=1)
        self.assertIn("&lt;script&gt;", view)
        self.assertNotIn("<script>", view)
        self.assertEqual(view.count("<summary>"), 1)
        self.assertEqual(len(result["items"]), 2)
        self.assertIn("gemini", result["disagreements"][0]["valid_selections"])
        self.assertEqual(len(result["disagreements"][0]["valid_selections"]), 3)
        second, _ = self.study.display_tables(self.artifact, start=1, max_items=1)
        self.assertIn("Showing items 2–2", second)

    def test_interrupt_and_duplicate_arm_cannot_erase_other_provider(self):
        self.study.collect_responses(self.artifact, "select", self.qwen, system="qwen")
        qwen_before = deepcopy([r for r in self.artifact["records"] if r["system"] == "qwen"])
        with tempfile.TemporaryDirectory() as temp:
            path = self.study.save_artifact(self.artifact, Path(temp)/"run", create=True)
            with self.assertRaises(KeyboardInterrupt):
                self.study.collect_responses(self.artifact, "select", Mock(side_effect=KeyboardInterrupt), system="gemini",
                                             on_record=lambda artifact: self.study.save_artifact(artifact, path.parent))
            saved = self.study.load_artifact(path, expected_run_id="five")
        self.assertEqual([r for r in saved["records"] if r["system"] == "qwen"], qwen_before)
        self.assertEqual([r["status"] for r in saved["records"] if r["condition"] == "gemini_select"], ["interrupted", "not_run"])
        backend = Mock()
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.study.collect_responses(saved, "select", backend, system="qwen")
        backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
