"""Mocked provider records, exact configuration and safe preflight; no paid calls."""
import importlib
from datetime import datetime, timezone
import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

from hebrew_acronyms.models.anthropic import eval as anthropic
from hebrew_acronyms.models.openai import eval as openai
from hebrew_acronyms.models.gemini import eval as gemini
from hebrew_acronyms.models.common.errors import retry_after_seconds


def http(payload, status=200, headers=None):
    return Mock(status_code=status, headers=headers or {}, json=Mock(return_value=payload))


class ProviderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.secret = "invented-provider-secret-never-save"
        env = patch.dict(os.environ, {"OPENAI_API_KEY": self.secret, "ANTHROPIC_API_KEY": self.secret,
                                    "GEMINI_API_KEY": self.secret})
        env.start()
        self.addCleanup(env.stop)

    def cases(self):
        return [(openai, openai.openai_response, openai.inspect_openai_model),
                (anthropic, anthropic.anthropic_response, anthropic.inspect_anthropic_model)]

    def payload(self, backend, text=" A "):
        common = {"id": "response-fixture", "model": backend.MODEL}
        if backend is openai:
            return {**common, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}}
        return {**common, "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 2}}

    def test_one_request_explicit_cap_metadata_and_supported_parameters(self):
        for backend, call, _ in self.cases():
            with self.subTest(backend=backend.__name__), patch.object(backend.requests, "post", return_value=http(self.payload(backend))) as post:
                result = call("invented prompt", model=backend.MODEL, max_output_tokens=321, timeout=9)
            post.assert_called_once()
            request = post.call_args.kwargs
            body = request["json"]
            self.assertFalse(request["allow_redirects"])
            self.assertEqual(request["timeout"], 9)
            self.assertEqual(body["model"], backend.MODEL)
            self.assertEqual(body["messages"], [{"role": "user", "content": "invented prompt"}])
            self.assertNotIn(self.secret, json.dumps(body))
            self.assertNotIn("seed", body)
            self.assertEqual(result["response"], " A ")
            self.assertEqual(result["status"], "response_received")
            self.assertEqual(result["attempts"], 1)
            self.assertTrue(result["usage_metadata"])
            self.assertEqual(result["model_version"], backend.MODEL)
            self.assertGreaterEqual(result["elapsed_seconds"], 0)
            if backend is openai:
                self.assertEqual(body["max_completion_tokens"], 321)
                self.assertEqual(body["temperature"], 0)
                self.assertFalse(body["store"])
                self.assertNotIn("reasoning_effort", body)
            else:
                self.assertEqual(body["max_tokens"], 321)
                self.assertEqual(body["thinking"], {"type": "adaptive"})
                self.assertEqual(body["output_config"], {"effort": "low"})
                self.assertTrue({"temperature", "top_p", "top_k", "budget_tokens"}.isdisjoint(body))

    def test_thoughts_omitted_final_text_joined_and_usage_preserved(self):
        payload = self.payload(anthropic)
        payload["content"] = [{"type": "thinking", "thinking": "private chain", "signature": "private signature"},
                              {"type": "redacted_thinking", "data": "private hidden"},
                              {"type": "text", "text": "בית "}, {"type": "text", "text": "משפט"}]
        with patch.object(anthropic.requests, "post", return_value=http(payload)):
            result = anthropic.anthropic_response("prompt", model=anthropic.MODEL)
        self.assertEqual(result["response"], "בית משפט")
        self.assertEqual(result["thought_parts_omitted"], 2)
        self.assertNotIn("private", json.dumps(result))

    def test_incomplete_blocked_empty_and_identity_mismatch_preserve_text(self):
        for backend, call, _ in self.cases():
            for reason, expected in (("length" if backend is openai else "max_tokens", "incomplete_response"),
                                     ("content_filter" if backend is openai else "refusal", "blocked"),
                                     (None, "incomplete_response")):
                payload = self.payload(backend, "partial")
                if backend is openai:
                    payload["choices"][0]["finish_reason"] = reason
                else:
                    payload["stop_reason"] = reason
                with self.subTest(backend=backend.__name__, reason=reason), patch.object(backend.requests, "post", return_value=http(payload)):
                    result = call("prompt", model=backend.MODEL)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["response"], "partial")
                self.assertFalse(result["retryable"])
            for payload, expected in ((self.payload(backend, " "), "missing_response"),
                                      ({**self.payload(backend), "model": "different"}, "incomplete_response")):
                with patch.object(backend.requests, "post", return_value=http(payload)):
                    self.assertEqual(call("prompt", model=backend.MODEL)["status"], expected)

    def test_credentials_never_saved_and_http_errors_emit_only_safe_diagnostics(self):
        for backend, call, _ in self.cases():
            payload = self.payload(backend, "answer " + self.secret)
            payload["usage"][self.secret] = [{"nested": self.secret}]
            with patch.object(backend.requests, "post", return_value=http(payload, headers={"x-request-id": self.secret})):
                result = call("prompt", model=backend.MODEL)
            self.assertNotIn(self.secret, json.dumps(result))
            self.assertIn("[REDACTED]", result["response"])
            for status, retryable in ((302, False), (401, False), (404, False), (429, True), (500, True), (529, True)):
                response = http({"error": self.secret}, status, {"Retry-After": "2", "request-id": self.secret})
                with patch.object(backend.requests, "post", return_value=response) as post:
                    result = call("prompt", model=backend.MODEL)
                post.assert_called_once()
                response.json.assert_called_once()
                self.assertNotIn(self.secret, json.dumps(result))
                self.assertEqual(result["retryable"], retryable)
                self.assertEqual(result["retry_after"], 2)

    def test_quota_billing_and_rate_diagnostics_are_distinct_and_sanitized(self):
        cases = [
            (openai, openai.openai_response, openai.MODEL, 429,
             {"code": "insufficient_quota"}, "quota_exhausted", True),
            (openai, openai.openai_response, openai.MODEL, 429,
             {"code": "rate_limit_exceeded"}, "rate_limit", False),
            (anthropic, anthropic.anthropic_response, anthropic.MODEL, 400,
             {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API"},
             "billing_blocked", True),
            (anthropic, anthropic.anthropic_response, anthropic.MODEL, 429,
             {"type": "rate_limit_error"}, "rate_limit", False),
            (gemini, gemini.gemini_response, "gemini-3.8-flash", 402,
             {}, "billing_blocked", True),
        ]
        for backend, call, model, status, error, expected, blocked in cases:
            error = {**error, "message": error.get("message", "") + " " + self.secret,
                     "unrecognized_sensitive_field": "private account value",
                     "nested": [{"other_provider_key": "invented-other-provider-secret"}]}
            with self.subTest(provider=backend.__name__, category=expected), patch.object(
                    backend.requests, "post", return_value=http({"error": error}, status)) as post:
                result = call("prompt", model=model)
            post.assert_called_once()
            self.assertEqual(result["error_category"], expected)
            self.assertEqual(result["provider_blocked"], blocked)
            self.assertEqual(result["retryable"], not blocked)
            self.assertNotIn(self.secret, json.dumps(result))
            self.assertNotIn("private account value", json.dumps(result))
            self.assertNotIn("invented-other-provider-secret", json.dumps(result))

    def test_gemini_quota_details_rate_and_unknown_429(self):
        cases = [
            ({"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}, "quota_exhausted", True),
            ({"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaValue": "0"}, "quota_exhausted", True),
            ({"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaValue": "20"}, "rate_limit", False),
            ({"quotaId": "unrecognized", "quotaValue": False}, "rate_or_quota_unknown", False),
            ({}, "rate_or_quota_unknown", False),
        ]
        for violation, expected, blocked in cases:
            payload = {"error": {"status": "RESOURCE_EXHAUSTED", "message": self.secret, "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [violation]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "19.5s"},
            ]}}
            with self.subTest(violation=violation), patch.object(gemini.requests, "post", return_value=http(payload, 429)) as post:
                result = gemini.gemini_response("prompt", model="gemini-3.8-flash")
            post.assert_called_once()
            self.assertEqual(result["error_category"], expected)
            self.assertEqual(result["provider_blocked"], blocked)
            self.assertEqual(result["retryable"], not blocked)
            self.assertEqual(result["retry_after"], 19.5)
            self.assertNotIn(self.secret, json.dumps(result))
        payload = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "BILLING_DISABLED"}]}}
        with patch.object(gemini.requests, "post", return_value=http(payload, 403)):
            result = gemini.gemini_response("prompt", model="gemini-3.8-flash")
        self.assertEqual(result["error_category"], "billing_blocked")
        payload = {"error": {"status": "RESOURCE_EXHAUSTED", "message":
            "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: fixture"}}
        with patch.object(gemini.requests, "post", return_value=http(payload, 429)):
            result = gemini.gemini_response("prompt", model="gemini-3.8-flash")
        self.assertEqual(result["error_category"], "quota_exhausted")
        self.assertTrue(result["provider_blocked"])
        self.assertNotIn("generativelanguage.googleapis.com", json.dumps(result))

    def test_malformed_error_and_headers_preserve_uncertainty_without_raw_data(self):
        for backend, call, model in [(openai, openai.openai_response, openai.MODEL),
                                     (anthropic, anthropic.anthropic_response, anthropic.MODEL),
                                     (gemini, gemini.gemini_response, "gemini-3.8-flash")]:
            for payload in ([], {"error": "private account value"}, {"error": {"status": "RESOURCE_EXHAUSTED"}}):
                with patch.object(backend.requests, "post", return_value=http(payload, 429,
                                  {"Retry-After": self.secret, "request-id": "private account value"})):
                    result = call("prompt", model=model)
                self.assertEqual(result["error_category"], "rate_or_quota_unknown")
                self.assertFalse(result["provider_blocked"])
                self.assertTrue(result["retryable"])
                self.assertIsNone(result["retry_after"])
                self.assertNotIn(self.secret, json.dumps(result))
                self.assertNotIn("private account value", json.dumps(result))
            response = http(None, 429)
            response.json.side_effect = ValueError(self.secret)
            with patch.object(backend.requests, "post", return_value=response):
                result = call("prompt", model=model)
            self.assertEqual(result["error_category"], "rate_or_quota_unknown")
            self.assertNotIn(self.secret, json.dumps(result))

    def test_retry_after_numeric_date_and_invalid_headers(self):
        now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(retry_after_seconds("Fri, 09 Oct 2026 12:01:30 GMT", now=now), 90)
        self.assertEqual(retry_after_seconds("Fri, 09 Oct 2026 11:59:30 GMT", now=now), 0)
        self.assertEqual(retry_after_seconds("90000"), 90000)  # Runner must stop, not shorten this.
        self.assertEqual(retry_after_seconds("1.5"), 1.5)
        for value in (None, True, {}, "-1", "nan", "inf", "1e999", self.secret, "x" * 129):
            self.assertIsNone(retry_after_seconds(value))
        with patch.object(openai.requests, "post", return_value=http({}, 429,
                          {"Retry-After": "Fri, 09 Oct 2099 12:01:30 GMT"})):
            result = openai.openai_response("prompt", model=openai.MODEL)
        self.assertGreater(result["retry_after"], 120)

    def test_timeouts_decode_errors_and_missing_credentials_are_sanitized(self):
        for backend, call, _ in self.cases():
            for error, retryable in ((requests.Timeout(self.secret), True), (requests.ConnectionError(self.secret), True),
                                     (RuntimeError(self.secret), False)):
                with patch.object(backend.requests, "post", side_effect=error) as post:
                    result = call("prompt", model=backend.MODEL)
                post.assert_called_once()
                self.assertEqual(result["status"], "service_error")
                self.assertEqual(result["retryable"], retryable)
                self.assertNotIn(self.secret, json.dumps(result))
            with patch.dict(os.environ, {}, clear=True), patch.object(backend.requests, "post") as post:
                result = call("prompt", model=backend.MODEL)
            post.assert_not_called()
            self.assertEqual(result["attempts"], 0)
            for response in (http([]), Mock(status_code=200, headers={}, json=Mock(side_effect=ValueError(self.secret)))):
                with patch.object(backend.requests, "post", return_value=response):
                    result = call("prompt", model=backend.MODEL)
                self.assertEqual(result["status"], "service_error")
                self.assertNotIn(self.secret, json.dumps(result))

    def test_invalid_settings_no_network_or_model_substitution(self):
        for backend, call, _ in self.cases():
            for kwargs in ({"model": "different"}, {"model": "https://example.com/" + self.secret},
                           {"timeout": 0}, {"timeout": float("inf")}, {"timeout": True},
                           {"max_output_tokens": 0}, {"max_output_tokens": True}, {"max_output_tokens": 999999}):
                with patch.object(backend.requests, "post") as post, self.assertRaises(ValueError) as error:
                    call("prompt", **{"model": backend.MODEL, **kwargs})
                self.assertNotIn(self.secret, str(error.exception))
                post.assert_not_called()
        with self.assertRaises(TypeError):
            anthropic.anthropic_response("prompt", model=anthropic.MODEL, temperature=0)
        with self.assertRaises(ValueError):
            anthropic.anthropic_response("prompt", model=anthropic.MODEL, effort="invented")
        with self.assertRaises(ValueError):
            openai.openai_response("prompt", model=openai.MODEL, temperature=float("nan"))

    def test_model_lookup_has_no_generation_and_does_not_claim_credit(self):
        for backend, _, inspect in self.cases():
            payload = {"id": backend.MODEL, "capabilities": {"thinking": {"supported": True}}}
            with patch.object(backend.requests, "get", return_value=http(payload)) as get, patch.object(backend.requests, "post") as post:
                result = inspect(model=backend.MODEL, timeout=4)
            post.assert_not_called()
            get.assert_called_once()
            self.assertEqual(get.call_args.kwargs["timeout"], 4)
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["availability"], "model_lookup_succeeded")
            self.assertEqual(result["model_metadata"], payload)
            with patch.object(backend.requests, "get", return_value=http({"id": "substitute"})):
                self.assertEqual(inspect(model=backend.MODEL)["availability"], "unverified")
            with patch.object(backend.requests, "post") as post, patch.object(backend.requests, "get") as get:
                importlib.reload(backend)
            get.assert_not_called(); post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
