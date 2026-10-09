"""Offline Gemini response/authentication fixtures; no actual service requests."""
from copy import deepcopy
import importlib
import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import guard


def http(payload, status=200):
    return Mock(status_code=status, json=Mock(return_value=payload))


class GeminiStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.addaudithook(guard)
        from hebrew_acronyms.models.gemini import eval
        cls.backend = eval

    def setUp(self):
        self.key = "invented-Gemini-api-key-secret-123"
        self.model = "gemini-3.8-flash"
        self.config = {"thinkingConfig": {"thinkingLevel": "low"}}
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": self.key})
        self.env.start()
        self.addCleanup(self.env.stop)

    def payload(self, parts=None, finish="STOP"):
        return {"modelVersion": "gemini-3.8-flash-fixture-version",
                "usageMetadata": {"promptTokenCount": 25, "candidatesTokenCount": 3,
                                  "thoughtsTokenCount": 5, "totalTokenCount": 33},
                "candidates": [{"index": 0, "finishReason": finish,
                                "content": {"role": "model", "parts": parts or [{"text": "A"}]}}]}

    def call(self, payload, **kwargs):
        with patch.object(self.backend.requests, "post", return_value=http(payload)) as post:
            result = self.backend.gemini_response("invented marked sentence", model=self.model,
                                                  generation_config=self.config, **kwargs)
        self.assertEqual(post.call_count, 1)
        return result, post

    def test_environment_header_only_timeout_no_retry_and_low_config(self):
        result, post = self.call(self.payload(), timeout=11)
        self.assertEqual(post.call_args.args[0], self.backend.GEMINI_URL.format(model=self.model))
        self.assertNotIn(self.key, post.call_args.args[0])
        self.assertNotIn("params", post.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], self.key)
        self.assertEqual(post.call_args.kwargs["timeout"], 11)
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.kwargs["json"]["generationConfig"], self.config)
        self.assertNotIn(self.key, json.dumps(post.call_args.kwargs["json"]))
        self.assertEqual(result["request_settings"]["generationConfig"], self.config)
        self.assertEqual(result["request_settings"]["max_attempts"], 1)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["requested_model"], self.model)
        self.assertEqual(result["model_version"], "gemini-3.8-flash-fixture-version")
        self.assertEqual(result["usage_metadata"]["thoughtsTokenCount"], 5)
        self.assertNotIn("digest", result)

    def test_final_multipart_text_concatenated_thought_text_and_signature_excluded(self):
        parts = [{"text": "do not save this internal thought", "thought": True, "thoughtSignature": "opaque-secret"},
                 {"text": "בית ", "thought": False}, {"inlineData": {"mimeType": "other"}},
                 {"text": "מלאכה"}, {"text": "another thought", "thought": True}]
        result, _ = self.call(self.payload(parts))
        self.assertEqual(result["response"], "בית מלאכה")
        self.assertEqual(result["thought_parts_omitted"], 2)
        self.assertEqual(result["status"], "response_received")
        self.assertNotIn("do not save", json.dumps(result))
        self.assertNotIn("opaque-secret", json.dumps(result))
        self.assertNotIn("another thought", json.dumps(result))

    def test_prompt_and_candidate_blocks_keep_status_and_any_final_text(self):
        result, _ = self.call({"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["response"], "")
        self.assertEqual(result["prompt_block_reason"], "SAFETY")
        result, _ = self.call(self.payload([{"text": "partial final text"}], finish="SAFETY"))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["response"], "partial final text")
        self.assertEqual(result["finish_reason"], "SAFETY")

    def test_missing_and_partial_responses_are_not_successes(self):
        cases = [({}, "missing_response"),
                 (self.payload([{"text": "thought only", "thought": True}]), "missing_response"),
                 (self.payload([{"text": " "}]), "missing_response"),
                 (self.payload([{"text": "partial"}], finish="MAX_TOKENS"), "incomplete_response"),
                 (self.payload([{"text": "unconfirmed"}], finish=None), "incomplete_response")]
        for payload, status in cases:
            with self.subTest(status=status):
                result, _ = self.call(payload)
            self.assertEqual(result["status"], status)
        result, _ = self.call(cases[3][0])
        self.assertEqual(result["response"], "partial")

    def test_multiple_candidates_keep_evidence_without_selecting_best(self):
        payload = self.payload()
        payload["candidates"].append({"finishReason": "STOP", "content": {"parts": [{"text": "B"}]}})
        result, _ = self.call(payload)
        self.assertEqual(result["status"], "incomplete_response")
        self.assertEqual(result["candidate_texts"], ["A", "B"])

    def test_missing_model_version_preserves_answer_but_identity_is_unverified(self):
        for version in (None, "", " "):
            payload = self.payload()
            payload["modelVersion"] = version
            result, _ = self.call(payload)
            self.assertEqual(result["response"], "A")
            self.assertEqual(result["identity_status"], "unverified")
        result, _ = self.call(self.payload())
        self.assertEqual(result["identity_status"], "verified")
        self.assertEqual(result["identity_verification"], "provider_reported_model_version")

    def test_timeout_and_exception_strings_never_leak_keys(self):
        for error in (self.backend.requests.Timeout(self.key), self.backend.requests.HTTPError("url?key=" + self.key),
                      RuntimeError({"secret": self.key})):
            with self.subTest(error_type=type(error)), patch.object(self.backend.requests, "post", side_effect=error) as post:
                result = self.backend.gemini_response("prompt", model=self.model)
            self.assertEqual(result["status"], "service_error")
            self.assertNotIn(self.key, json.dumps(result))
            post.assert_called_once()
        response = http({"error": {"message": "leaked " + self.key}}, 429)
        with patch.object(self.backend.requests, "post", return_value=response) as post:
            result = self.backend.gemini_response("prompt", model=self.model)
        self.assertEqual(result["http_status"], 429)
        self.assertNotIn(self.key, json.dumps(result))
        response.json.assert_called_once()
        post.assert_called_once()

    def test_recursive_redaction_of_every_returned_field(self):
        payload = self.payload([{"text": "answer " + self.key}], finish="STOP")
        payload["modelVersion"] = "version-" + self.key
        payload["usageMetadata"][self.key] = [{"nested": self.key}]
        payload["promptFeedback"] = {"blockReason": "SAFETY-" + self.key}
        result, _ = self.call(payload)
        encoded = json.dumps(result)
        self.assertNotIn(self.key, encoded)
        self.assertIn("[REDACTED]", encoded)
        self.assertEqual(result["response"], "answer [REDACTED]")

    def test_missing_key_no_request_and_no_configuration_file_loading(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(self.backend.requests, "post") as post:
            result = self.backend.gemini_response("prompt", model=self.model)
        post.assert_not_called()
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["status"], "service_error")
        with patch("dotenv.load_dotenv") as dotenv, patch.object(self.backend.requests, "post") as post:
            importlib.reload(self.backend)
        dotenv.assert_not_called();post.assert_not_called()

    def test_required_safe_model_supported_config_and_finite_timeout(self):
        for model in (None, "", "https://example.com", "models/gemini-3.8-flash", "gemini-3.8-flash?key=" + self.key):
            with self.subTest(model=model), patch.object(self.backend.requests, "post") as post:
                with self.assertRaises(ValueError) as error:
                    self.backend.gemini_response("prompt", model=model)
                self.assertNotIn(self.key, str(error.exception))
                post.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Gemini 3"):
            self.backend.gemini_response("prompt", model="gemini-2.5-flash", generation_config=self.config)
        for timeout in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.backend.gemini_response("prompt", model=self.model, timeout=timeout)
        for config in ({"thinkingConfig": {"thinkingBudget": 1}}, {"thinkingConfig": {"thinkingLevel": "minimal"}},
                       {"temperature": 0}, {"candidateCount": 1}, {"api_key": self.key}, {"stopSequences": [self.key]}):
            with self.subTest(config_keys=list(config)), self.assertRaises(ValueError) as error:
                self.backend.gemini_response("prompt", model=self.model, generation_config=config)
            self.assertNotIn(self.key, str(error.exception))

    def test_malformed_payload_is_a_safe_record_and_input_config_is_unchanged(self):
        config = deepcopy(self.config)
        result, _ = self.call(["unexpected payload"])
        self.assertEqual(result["status"], "service_error")
        result, _ = self.call({"candidates": [{"finishReason": {}, "content": "invalid"}]})
        self.assertEqual(result["status"], "missing_response")
        self.assertEqual(self.config, config)
        with patch.object(self.backend.requests, "post", return_value=Mock(status_code=200, json=Mock(side_effect=ValueError(self.key)))):
            result = self.backend.gemini_response("prompt", model=self.model)
        self.assertNotIn(self.key, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
