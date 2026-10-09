"""Mocked provider records, exact configuration and safe preflight; no paid calls."""
import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

from hebrew_acronyms.models.anthropic import eval as anthropic
from hebrew_acronyms.models.openai import eval as openai


def http(payload, status=200, headers=None):
    return Mock(status_code=status, headers=headers or {}, json=Mock(return_value=payload))


class ProviderAdapterTests(unittest.TestCase):
    def setUp(self):
        self.secret = "invented-provider-secret-never-save"
        env = patch.dict(os.environ, {"OPENAI_API_KEY": self.secret, "ANTHROPIC_API_KEY": self.secret})
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

    def test_credentials_never_saved_and_http_errors_not_decoded(self):
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
                response.json.assert_not_called()
                self.assertNotIn(self.secret, json.dumps(result))
                self.assertEqual(result["retryable"], retryable)
                self.assertEqual(result["retry_after"], "2")

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
