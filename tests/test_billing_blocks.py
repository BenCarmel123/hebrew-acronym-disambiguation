"""Documented spend-limit errors through real adapters and the offline runner.

Source shapes:
https://developers.openai.com/api/docs/guides/spend-limits
https://platform.claude.com/docs/en/api/rate-limits
Only the HTTP transport is mocked; no provider requests are sent.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hebrew_acronyms import test_evaluation as runner
from hebrew_acronyms.models.anthropic import eval as anthropic
from hebrew_acronyms.models.openai import eval as openai


OPENAI_CODES = (
    "project_spend_limit_exceeded", "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded", "credit_balance_exhausted",
)
ANTHROPIC_PREFIXES = (
    "You have reached your specified API usage limits",
    "You have reached your specified workspace API usage limits",
)
SYSTEMS = [
    {"name": "openai", "provider": "openai", "model": openai.MODEL,
     "settings": {"timeout": 120, "max_output_tokens": 512, "temperature": 0}},
    {"name": "anthropic", "provider": "anthropic", "model": anthropic.MODEL,
     "settings": {"timeout": 120, "max_output_tokens": 1024, "effort": "low"}},
]


def billing_cases():
    return [("openai", 429, {"code": code}) for code in OPENAI_CODES] + [
        ("anthropic", 429, {"type": "rate_limit_error",
                            "details": {"error_code": "enforced_spend_limit_reached"}}),
        *[("anthropic", 400, {"type": "invalid_request_error", "message": prefix + ". Access resumes later."})
          for prefix in ANTHROPIC_PREFIXES],
    ]


def http(payload, status=200):
    return Mock(status_code=status, headers={}, json=Mock(return_value=payload))


class BillingBlockTests(unittest.TestCase):
    def setUp(self):
        self.secret = "invented-billing-test-key"
        self.private = "private-workspace-and-account-value"
        env = patch.dict(os.environ, {"OPENAI_API_KEY": self.secret, "ANTHROPIC_API_KEY": self.secret})
        env.start()
        self.addCleanup(env.stop)

    def error_payload(self, error):
        return {"error": {**error,
                "message": error.get("message", "") + " " + self.secret + " " + self.private,
                "unrecognized": {"other_provider_key": "private-other-key"}},
                "request_id": self.private}

    def adapter(self, provider):
        return (openai.openai_response, openai.MODEL) if provider == "openai" else (anthropic.anthropic_response, anthropic.MODEL)

    def assert_filtered(self, value):
        text = json.dumps(value)
        for secret in (self.secret, self.private, "private-other-key"):
            self.assertNotIn(secret, text)
        for prefix in ANTHROPIC_PREFIXES:
            self.assertNotIn(prefix, text)

    def test_openai_spend_and_credit_codes_block_without_raw_errors(self):
        for provider, status, error in billing_cases()[:4]:
            with self.subTest(code=error["code"]), patch.object(
                    openai.requests, "post", return_value=http(self.error_payload(error), status)) as post:
                result = openai.openai_response("invented prompt", model=openai.MODEL)
            post.assert_called_once()
            self.assertEqual(result["error_category"], "billing_blocked")
            self.assertTrue(result["provider_blocked"])
            self.assertFalse(result["retryable"])
            self.assert_filtered(result)

    def test_anthropic_nested_code_and_both_message_prefixes_block(self):
        for provider, status, error in billing_cases()[4:]:
            with self.subTest(status=status, error=error), patch.object(
                    anthropic.requests, "post", return_value=http(self.error_payload(error), status)) as post:
                result = anthropic.anthropic_response("invented prompt", model=anthropic.MODEL)
            post.assert_called_once()
            self.assertEqual(result["error_category"], "billing_blocked")
            self.assertTrue(result["provider_blocked"])
            self.assertFalse(result["retryable"])
            self.assert_filtered(result)

    def test_ordinary_unknown_and_near_match_errors_do_not_block_provider(self):
        cases = [
            ("openai", 429, {"code": "rate_limit_exceeded"}, "rate_limit", True),
            ("openai", 429, {"code": "unknown_code"}, "rate_or_quota_unknown", True),
            ("openai", 400, {"code": "invalid_request_error"}, "http_error", False),
            ("anthropic", 429, {"type": "rate_limit_error"}, "rate_limit", True),
            ("anthropic", 429, {"type": "rate_limit_error", "details": {"error_code": "unknown_code"}}, "rate_limit", True),
            ("anthropic", 429, {"type": "unknown_error", "details": []}, "rate_or_quota_unknown", True),
            ("anthropic", 400, {"type": "invalid_request_error", "message": "Invalid parameter"}, "http_error", False),
            ("anthropic", 400, {"type": "invalid_request_error", "message": "Quoted: " + ANTHROPIC_PREFIXES[0]}, "http_error", False),
            ("anthropic", 400, {"type": "other_error", "message": ANTHROPIC_PREFIXES[0]}, "http_error", False),
            ("anthropic", 429, {"type": "invalid_request_error", "message": ANTHROPIC_PREFIXES[1]}, "rate_or_quota_unknown", True),
            ("anthropic", 400, {"type": "invalid_request_error", "details": {"error_code": "enforced_spend_limit_reached"}}, "http_error", False),
        ]
        for provider, status, error, category, retryable in cases:
            call, model = self.adapter(provider)
            with self.subTest(provider=provider, status=status, error=error), patch.object(
                    openai.requests, "post", return_value=http(self.error_payload(error), status)):
                result = call("invented prompt", model=model)
            self.assertEqual(result["error_category"], category)
            self.assertFalse(result["provider_blocked"])
            self.assertEqual(result["retryable"], retryable)
            self.assert_filtered(result)

    def test_adapter_runner_stops_after_one_call_resumes_without_calls_and_continues_other_provider(self):
        for blocked_provider, status, error in billing_cases():
            with self.subTest(provider=blocked_provider, status=status, error=error), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, output = root / "dev.csv", root / "pilot"
                with source.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["item_id", "acronym", "sentence", "gold_expansion", "candidates"])
                    writer.writeheader()
                    writer.writerows({"item_id": f"fixture-{i}", "acronym": 'אב"ג',
                                      "sentence": 'דוגמה אב"ג בהקשר', "gold_expansion": "ראשון",
                                      "candidates": "ראשון|שני"} for i in range(10))
                runner.prepare_evaluation(source, output, cohort="dev_pilot", systems=SYSTEMS,
                                          code_revision="a" * 40)
                calls = Counter()

                def transport(url, **kwargs):
                    model = kwargs["json"]["model"]
                    provider = "openai" if model == openai.MODEL else "anthropic"
                    calls[provider] += 1
                    if provider == blocked_provider:
                        return http(self.error_payload(error), status)
                    if provider == "openai":
                        return http({"model": model, "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
                    return http({"model": model, "content": [{"type": "text", "text": "A"}], "stop_reason": "end_turn",
                                 "usage": {"input_tokens": 10, "output_tokens": 1}})

                # Both real adapters use the requests module; mock only its HTTP boundary.
                with patch.object(openai.requests, "post", side_effect=transport) as post:
                    first = runner.run_evaluation(output, code_revision="a" * 40, sleep=Mock())
                    resumed = runner.run_evaluation(output, code_revision="a" * 40, sleep=Mock())
                healthy = "anthropic" if blocked_provider == "openai" else "openai"
                self.assertEqual(calls, {blocked_provider: 1, healthy: 20})
                self.assertEqual(post.call_count, 21)
                self.assertEqual(first["n_calls"], resumed["n_calls"])
                self.assertEqual(resumed["n_completed"], 20)
                self.assertTrue(resumed["provider_states"][blocked_provider]["blocked"])
                withheld = [r for r in resumed["records"] if r["system"] == blocked_provider and r["status"] == "not_run"]
                self.assertEqual(len(withheld), 19)
                self.assertTrue(all(r["not_run_reason"] == "provider_blocked" for r in withheld))
                self.assert_filtered(resumed)
                for journal in output.glob("attempts*.jsonl"):
                    self.assert_filtered(journal.read_text())


if __name__ == "__main__":
    unittest.main()
