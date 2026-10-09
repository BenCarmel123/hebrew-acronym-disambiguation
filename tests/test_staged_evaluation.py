"""Run staged workflows on invented cohorts with a blocked network boundary."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hebrew_acronyms import staged_evaluation as staged
from hebrew_acronyms import test_evaluation as runner


class StagedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "session"
        self.dev, self.test = self.root / "dev.csv", self.root / "test.csv"
        for path, count in ((self.dev, 12), (self.test, 395)):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "acronym", "sentence", "gold_expansion", "candidates"])
                writer.writeheader()
                writer.writerows({"item_id": f"item-{i}", "acronym": 'אב"ג', "sentence": 'הקשר אב"ג לדוגמה',
                                  "gold_expansion": "אחד", "candidates": "אחד|שניים|שלושה"} for i in range(count))
        self.openai = {"name": "openai", "provider": "openai", "model": "gpt-4.1-mini-2025-04-14",
                       "settings": {"timeout": 120, "max_output_tokens": 512, "temperature": 0}}
        self.anthropic = {"name": "anthropic", "provider": "anthropic", "model": "claude-haiku-5-5",
                          "settings": {"timeout": 120, "max_output_tokens": 1024, "effort": "low"}}
        self.response = {"status": "response_received", "response": "A", "identity_status": "verified",
                         "usage_metadata": {"prompt_tokens": 10, "completion_tokens": 2,
                                            "input_tokens": 10, "output_tokens": 2}}
        self.rates = {name: {"input": 1, "output": 1} for name in staged.SYSTEM_NAMES}
        self.reserves = dict.fromkeys(staged.SYSTEM_NAMES, .01)
        self.prior = 3.0
        self.prepare()
        socket_patch = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in staged tests"))
        socket_patch.start()
        self.addCleanup(socket_patch.stop)
        self.call = Mock(return_value=self.response)
        call_patch = patch.object(runner, "_call", self.call)
        call_patch.start()
        self.addCleanup(call_patch.stop)

    def prepare(self):
        return staged.prepare_session(self.output, self.dev, self.test, code_revision="a" * 40,
            prior_spend_ils=self.prior, rates=self.rates, reserves=self.reserves)

    def pilot(self, system=None, **kwargs):
        return staged.run_system_pilot(self.output, system or self.openai, **kwargs)

    def full(self, system=None, **kwargs):
        system = system or self.openai
        review = staged.review_system_pilot(self.output, system)
        return staged.run_system_full(self.output, system, inspected_pilot_identity=review["pilot_identity"], **kwargs)


    def test_xai_and_qwen14_have_distinct_pilots_and_shared_accounting(self):
        xai = {"name": "xai", "provider": "xai", "model": "grok-4.7",
               "settings": {"timeout": 120, "max_output_tokens": 1024, "effort": "low"}}
        qwen14 = {"name": "qwen14", "provider": "qwen", "model": "qwen2.5:14b",
                  "settings": {"timeout": 120, "options": {"temperature": 0, "seed": 42, "num_predict": 512},
                               "expected_digest": "fixture-digest"}}
        self.call.return_value = {"status": "response_received", "response": "A", "identity_status": "verified",
            "usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                               "output_tokens_details": {"reasoning_tokens": 3}},
            "response_metadata": {"prompt_eval_count": 10, "eval_count": 5}}
        first, second = self.pilot(xai), self.pilot(qwen14)
        self.assertEqual(first["summary"]["n_completed"], 20)
        self.assertEqual(second["summary"]["n_completed"], 20)
        self.assertAlmostEqual(staged.session_summary(self.output)["charged_or_reserved_usd"], 40 * .000015)
        for a, b in zip(first["manifest"]["identity"]["requests"], second["manifest"]["identity"]["requests"]):
            for field in ("item_id", "task", "prompt", "shown_order"):
                self.assertEqual(a[field], b[field])
        self.assertEqual(self.full(xai, max_new_calls=1)["n_calls"], 1)
        self.assertEqual(self.full(qwen14, max_new_calls=1)["n_calls"], 1)
        self.assertEqual(staged.session_summary(self.output)["systems"]["qwen"]["full_test"]["status"], "not_run")

    def test_system_switch_preserves_spend_prompts_order_and_reports_absent(self):
        first = self.pilot()
        second = self.pilot(self.anthropic)
        for a, b in zip(first["manifest"]["identity"]["requests"], second["manifest"]["identity"]["requests"]):
            for key in ("item_id", "task", "prompt", "shown_order"):
                self.assertEqual(a[key], b[key])
        summary = staged.session_summary(self.output)
        self.assertAlmostEqual(summary["charged_or_reserved_usd"], 40 * .000012)
        self.assertAlmostEqual(summary["total_accounted_ils"], 3 + 4 * 40 * .000012)
        self.assertEqual(summary["systems"]["gemini"]["dev_pilot"]["status"], "not_run")
        self.assertEqual(summary["systems"]["qwen"]["full_test"]["status"], "not_run")
        self.pilot()  # Selecting the first system again sends nothing twice.
        self.assertEqual(self.call.call_count, 40)

    def test_blocked_provider_stops_across_resume_healthy_can_finish_pilot(self):
        def respond(system, prompt):
            if system["name"] == "openai":
                return {"status": "service_error", "response": "", "provider_blocked": True,
                        "retryable": False, "error_category": "quota_exhausted"}
            return self.response
        self.call.side_effect = respond
        self.pilot()
        self.pilot()
        self.assertEqual(self.call.call_count, 1)
        self.pilot(self.anthropic)
        self.assertEqual(self.call.call_count, 21)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.full()
        self.assertEqual(self.full(self.anthropic, max_new_calls=1)["n_calls"], 1)
        summary = staged.session_summary(self.output)
        self.assertAlmostEqual(summary["charged_or_reserved_usd"], .01 + 21 * .000012)

    def test_disconnect_reserves_unknown_call_and_switch_cannot_erase_it(self):
        self.call.side_effect = [self.response, KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.pilot()
        interrupted = staged.session_summary(self.output)
        self.assertAlmostEqual(interrupted["charged_or_reserved_usd"], .010012)
        self.call.side_effect = None
        self.pilot(self.anthropic)
        resumed = self.pilot()["summary"]
        self.assertEqual(resumed["n_ambiguous"], 1)
        self.assertEqual(resumed["n_completed"], 19)
        self.assertAlmostEqual(staged.session_summary(self.output)["charged_or_reserved_usd"], .01 + 39 * .000012)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.full()

    def test_full_requires_own_pilot_and_exact_inspection_and_configuration(self):
        first = self.pilot()
        for system, identity in ((self.anthropic, first["manifest"]["identity_sha256"]), (self.openai, "stale")):
            with self.assertRaises(ValueError):
                staged.run_system_full(self.output, system, inspected_pilot_identity=identity)
        changed = deepcopy(self.openai)
        changed["settings"]["max_output_tokens"] = 1024
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            staged.run_system_full(self.output, changed, inspected_pilot_identity=first["manifest"]["identity_sha256"])
        self.assertEqual(self.call.call_count, 20)
        full = self.full(max_new_calls=1)
        self.assertEqual(full["n_items"], 395)
        self.assertEqual(full["n_records"], 790)

    def test_full_resume_then_add_system_recomputes_all_journals(self):
        self.pilot()
        self.full(max_new_calls=2)
        self.pilot(self.anthropic)
        resumed = self.full(max_new_calls=1)
        self.assertEqual(resumed["n_calls"], 3)
        self.assertEqual(self.call.call_count, 43)
        self.assertAlmostEqual(staged.session_summary(self.output)["charged_or_reserved_usd"], 43 * .000012)
        self.assertEqual(len(list((self.output / "openai/full-test").glob("attempts*.jsonl"))), 2)

    def test_complete_fixed_full_cohort_does_not_resend_on_resume(self):
        self.pilot()
        full = self.full()
        self.assertEqual((full["n_items"], full["n_records"], full["n_completed"]), (395, 790, 790))
        self.assertEqual({group["task"]: group["n_items"] for group in full["by_system_task"]},
                         {"generate": 395, "select": 395})
        self.full()
        self.assertEqual(self.call.call_count, 810)
        self.assertAlmostEqual(staged.session_summary(self.output)["charged_or_reserved_usd"], 810 * .000012)

    def test_budget_limit_accumulates_unknown_usage_across_switches(self):
        # A separate session with two conservative calls left in the global cap.
        self.output = self.root / "tight-session"
        self.prior = 99.92
        self.prepare()
        self.call.return_value = {"status": "response_received", "response": "A"}
        first = self.pilot(max_new_calls=1)["summary"]
        second = self.pilot(self.anthropic)["summary"]
        third = self.pilot()["summary"]
        self.assertEqual((first["n_calls"], second["n_calls"], third["n_calls"]), (1, 1, 1))
        self.assertEqual(self.call.call_count, 2)
        self.assertAlmostEqual(staged.session_summary(self.output)["total_accounted_ils"], 100)

    def test_prior_prices_inputs_code_and_model_metadata_are_immutable(self):
        pilot = self.pilot(model_identity={"digest": "stable"})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.pilot(model_identity={"digest": "changed"})
        self.prior = 0
        with self.assertRaisesRegex(ValueError, "Session identity mismatch"):
            self.prepare()
        self.prior = 3
        with patch.object(runner, "_code_hashes", return_value={"changed": "code"}):
            with self.assertRaisesRegex(ValueError, "source differs"):
                self.pilot()
        self.dev.write_text(self.dev.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "input identity"):
            self.pilot()
        self.assertEqual(self.call.call_count, 20)

    def test_unverified_identity_and_incomplete_response_fail_pilot_gate(self):
        for response in ({**self.response, "identity_status": "unverified"},
                         {**self.response, "status": "incomplete_response"}):
            self.output = self.root / str(len(list(self.root.iterdir())))
            self.prepare()
            self.call.return_value = response
            self.pilot()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.full()

    def test_projection_reloads_cost_and_blocks_before_test(self):
        self.call.return_value = {**self.response, "usage_metadata": {"prompt_tokens": 100000, "completion_tokens": 0}}
        self.pilot()
        with self.assertRaisesRegex(ValueError, "projection"):
            self.full()
        self.assertFalse((self.output / "openai/full-test/manifest.json").exists())
        self.assertEqual(self.call.call_count, 20)

    def test_unknown_tail_or_orphan_journal_blocks_other_system_before_calls(self):
        self.pilot(max_new_calls=1)
        journal = self.output / "openai/dev-pilot/attempts.jsonl"
        with journal.open("ab") as stream:
            stream.write(b'{"event":"started"')
        with self.assertRaisesRegex(ValueError, "unknown spending"):
            self.pilot(self.anthropic)
        self.assertEqual(self.call.call_count, 1)

    def test_existing_legacy_output_cannot_be_reinitialized_with_zero_spend(self):
        legacy = self.root / "legacy"
        (legacy / "dev-pilot").mkdir(parents=True)
        evidence = legacy / "dev-pilot/manifest.json"
        evidence.write_text('{"old":"keep"}')
        self.output = legacy
        with self.assertRaisesRegex(ValueError, "Existing evaluation outputs"):
            self.prepare()
        self.assertEqual(evidence.read_text(), '{"old":"keep"}')


if __name__ == "__main__":
    unittest.main()
