"""Offline durability/identity checks, using only invented CSV inputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hebrew_acronyms import test_evaluation as runner


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "items.csv"
        self.output = self.root / "run"
        self.systems = [{"name": "mock", "provider": "fixture", "model": "mock-model", "settings": {}}]
        self.rows = [{"item_id": "one", "acronym": 'אב"ג', "sentence": 'דוגמה אב"ג בהקשר',
                      "gold_expansion": "ראשון", "candidates": "ראשון|שני"},
                     {"item_id": "two", "acronym": 'דב"ג', "sentence": 'דוגמה דב"ג בהקשר',
                      "gold_expansion": "ראשון", "candidates": "ראשון|שני"}]
        self.write_rows()
        self.response = {"status": "response_received", "response": "A", "usage_metadata": {"input_tokens": 10, "output_tokens": 2}}

    def write_rows(self):
        with self.source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def prepare(self, **kwargs):
        return runner.prepare_evaluation(self.source, self.output, cohort="fixture", systems=self.systems,
                                         code_revision="test-commit", **kwargs)

    def run_saved(self, responder=None, **kwargs):
        return runner.run_evaluation(self.output, code_revision="test-commit",
                                     responders={"mock": responder or Mock(return_value=self.response)}, sleep=lambda _: None, **kwargs)

    def test_completed_answers_never_retried_even_invalid(self):
        manifest = self.prepare()
        self.assertEqual(len(manifest["identity"]["requests"]), 4)
        responder = Mock(return_value={**self.response, "response": "wrong answer"})
        summary = self.run_saved(responder)
        again = self.run_saved(responder)
        self.assertEqual(responder.call_count, 4)
        self.assertEqual(summary["n_completed"], 4)
        self.assertEqual(again["n_calls"], 4)
        self.assertEqual(again["usage"]["mock"]["input_tokens"], 40)
        self.assertEqual(again["by_system_task"][1]["accuracy"], 0)
        self.assertEqual(self.prepare()["run_id"], manifest["run_id"])

    def test_disconnect_saves_earlier_response_and_does_not_resend_ambiguous(self):
        self.prepare()
        responder = Mock(side_effect=[self.response, KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_saved(responder)
        saved = runner.summarize_evaluation(self.output)
        self.assertEqual((saved["n_completed"], saved["n_ambiguous"], saved["n_pending"]), (1, 1, 2))
        resumed = Mock(return_value=self.response)
        summary = self.run_saved(resumed)
        self.assertEqual(resumed.call_count, 2)
        self.assertEqual((summary["n_calls"], summary["n_completed"], summary["n_ambiguous"]), (4, 3, 1))
        self.assertEqual(summary["usage"]["mock"]["attempts_without_usage"], 1)

    def test_durable_start_visible_inside_request(self):
        self.prepare()
        def request(prompt):
            saved = runner.summarize_evaluation(self.output)
            self.assertEqual(saved["n_ambiguous"], 1)
            self.assertGreater(saved["n_calls"], 0)
            return self.response
        self.run_saved(request)

    def test_configuration_input_and_code_mismatches_stop_before_calls(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare(seed=12)
        with self.assertRaisesRegex(ValueError, "code identity"):
            runner.run_evaluation(self.output, code_revision="wrong")
        with patch.object(runner, "_code_hashes", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "code identity"):
                self.run_saved()
        self.rows[0]["sentence"] += " changed"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "input identity"):
            self.run_saved()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare()

    def test_truncated_finish_preserves_bytes_and_recovers_in_new_segment(self):
        self.prepare()
        self.run_saved(max_new_calls=1)
        path = self.output / "attempts.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        broken = lines[0] + lines[1][:25]
        path.write_bytes(broken)
        summary = self.run_saved()
        self.assertEqual(path.read_bytes(), broken)
        self.assertEqual(summary["n_ambiguous"], 1)
        self.assertEqual(summary["n_calls"], 4)
        self.assertEqual(len(summary["truncated_journal_tails"]), 1)
        self.assertTrue((self.output / "attempts-000001.jsonl").exists())

    def test_corrupt_terminated_journal_line_blocks_calls(self):
        self.prepare()
        (self.output / "attempts.jsonl").write_text('{"broken":\n')
        responder = Mock(return_value=self.response)
        with self.assertRaisesRegex(ValueError, "Invalid journal"):
            self.run_saved(responder)
        responder.assert_not_called()

    def test_retries_bounded_across_resume_and_calls_include_errors(self):
        self.prepare(max_attempts=2, max_calls=8)
        responder = Mock(return_value={"status": "service_error", "response": "", "retryable": True})
        first = self.run_saved(responder, max_new_calls=1)
        self.assertEqual(first["n_calls"], 1)
        summary = self.run_saved(responder)
        self.assertEqual((summary["n_calls"], responder.call_count), (8, 8))
        self.assertEqual(summary["n_pending"], 0)
        self.run_saved(responder)
        self.assertEqual(responder.call_count, 8)
        self.assertEqual(summary["usage"]["mock"]["attempts_without_usage"], 8)

    def test_nonretryable_and_incomplete_do_not_retry(self):
        self.prepare(max_attempts=3)
        responder = Mock(side_effect=[{"status": "service_error", "response": "", "retryable": False},
                                      {"status": "incomplete_response", "response": "A", "retryable": True},
                                      self.response, self.response])
        result = self.run_saved(responder)
        self.assertEqual(responder.call_count, 4)
        self.assertEqual(result["n_incomplete"], 2)
        self.assertEqual(result["n_pending"], 0)

    def test_conservative_budget_reserve_counts_ambiguous_attempts(self):
        self.prepare(reserve_per_call_usd=.1, budget_usd=.2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_saved(Mock(side_effect=KeyboardInterrupt()))
        summary = self.run_saved()
        self.assertEqual(summary["n_calls"], 2)
        self.assertEqual(summary["stop_reason"], "budget_reserve_limit")
        self.assertAlmostEqual(summary["charged_or_reserved_usd"], .2)

    def test_more_than_26_candidates_and_shuffle_independent_of_subset(self):
        self.rows[1]["candidates"] = "|".join(f"candidate-{n}" for n in range(30))
        self.rows[1]["gold_expansion"] = "candidate-29"
        self.write_rows()
        manifest = self.prepare()
        selected = next(request for request in manifest["identity"]["requests"] if request["item_id"] == "two" and request["task"] == "select")
        self.assertEqual(len(selected["shown_order"]), 30)
        self.assertIn("AD.", selected["prompt"])
        self.rows = self.rows[1:]
        self.write_rows()
        self.output = self.root / "subset"
        other = self.prepare()
        self.assertEqual(other["identity"]["requests"][1]["shown_order"], selected["shown_order"])
        summary = self.run_saved(Mock(return_value={**self.response, "response": "AD"}))
        self.assertTrue(summary["records"][1]["valid"])

    def test_generation_prompt_has_no_candidate_list_or_gold_field(self):
        manifest = self.prepare()
        for request in manifest["identity"]["requests"]:
            if request["task"] == "generate":
                self.assertIsNone(request["shown_order"])
                self.assertNotIn("ראשון", request["prompt"])
                self.assertNotIn("שני", request["prompt"])

    def test_full_cohort_all_rows_and_pilot_first_ten(self):
        self.rows = [{**self.rows[0], "item_id": f"item-{n}"} for n in range(395)]
        self.write_rows()
        real_system = [{"name": "gemini", "provider": "gemini", "model": "gemini-3.8-flash", "settings": {
            "timeout": 120, "generation_config": {"maxOutputTokens": 128}}}]
        full = runner.prepare_evaluation(self.source, self.output, cohort="full_test", systems=real_system, code_revision="a" * 40)
        self.assertEqual(len(full["identity"]["rows"]), 395)
        self.assertEqual(len(full["identity"]["requests"]), 790)
        self.assertEqual(runner.summarize_evaluation(self.output)["n_pending"], 790)
        pilot = runner.prepare_evaluation(self.source, self.root / "pilot", cohort="dev_pilot", systems=real_system, code_revision="a" * 40)
        self.assertEqual([row["item_id"] for row in pilot["identity"]["rows"]], [f"item-{n}" for n in range(10)])
        with self.assertRaisesRegex(ValueError, "fixture"):
            runner.run_evaluation(self.output, code_revision="a" * 40, responders={})
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "395"):
            runner.prepare_evaluation(self.source, self.root / "wrong", cohort="full_test", systems=real_system, code_revision="a" * 40)

    def test_unknown_usage_cost_not_silently_zero(self):
        self.prepare()
        summary = self.run_saved()
        estimate = runner.estimate_cost(summary, {"mock": {"input": 1, "output": 2}})
        self.assertAlmostEqual(estimate["pilot_usd"], .000056)
        self.assertAlmostEqual(estimate["projected_full_usd"], .000056 * 395 / 2)
        summary["usage"]["mock"]["attempts_without_usage"] = 1
        with self.assertRaisesRegex(ValueError, "unknown billed usage"):
            runner.estimate_cost(summary, {"mock": {"input": 1, "output": 2}})

    def test_incomplete_pilot_cannot_underestimate_cost(self):
        self.prepare()
        summary = self.run_saved(max_new_calls=1)
        with self.assertRaisesRegex(ValueError, "Complete all pilot"):
            runner.estimate_cost(summary, {"mock": {"input": 1, "output": 2}})

    def test_actual_usage_releases_unused_reserve(self):
        self.prepare(reserve_per_call_usd={"mock": .1}, budget_usd=.11,
                     rates_usd_per_million={"mock": {"input": 1, "output": 2}})
        summary = self.run_saved()
        self.assertEqual(summary["n_completed"], 4)
        self.assertAlmostEqual(summary["charged_or_reserved_usd"], .000056)
        self.assertEqual(summary["reserved_usd"], 0)

    def test_unknown_truncated_start_blocks_resume(self):
        self.prepare()
        self.run_saved(max_new_calls=1)
        with (self.output / "attempts.jsonl").open("ab") as stream:
            stream.write(b'{"event":"started"')
        responder = Mock(return_value=self.response)
        with self.assertRaisesRegex(ValueError, "no identifiable"):
            self.run_saved(responder)
        responder.assert_not_called()

    def test_provider_usage_conventions_include_gemini_thinking(self):
        self.assertEqual(runner._usage({"usage_metadata": {"promptTokenCount": 12, "candidatesTokenCount": 3, "thoughtsTokenCount": 8}}, "gemini"),
                         {"input_tokens": 12, "output_tokens": 3, "thinking_tokens": 8, "cached_input_tokens": 0})
        self.assertEqual(runner._usage({"usage_metadata": {"prompt_tokens": 12, "completion_tokens": 3}}, "openai")["input_tokens"], 12)
        self.assertEqual(runner._usage({"usage_metadata": {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 4}}, "anthropic")["input_tokens"], 16)

    def test_existing_outputs_and_invalid_inputs_rejected(self):
        self.output.mkdir()
        old = self.output / "old.txt"
        old.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.prepare()
        self.assertEqual(old.read_text(), "preserve")
        self.output = self.root / "another"
        self.rows[0]["gold_expansion"] = "missing"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "gold outside"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
