"""Mocked Ollama identity, finite timeouts and response-evidence contracts."""
import sys
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import guard


def http(payload):
    return Mock(json=Mock(return_value=payload), raise_for_status=Mock())


class QwenStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.addaudithook(guard)
        from hebrew_acronyms.models.qwen import eval
        cls.backend = eval

    def setUp(self):
        self.model = "qwen2.5:7b"
        self.digest = "fixture-digest"
        self.identity = {"name": self.model, "model": self.model, "digest": self.digest}

    def test_inspection_reads_actual_server_version_digest_and_model_defaults(self):
        tags = {"models": [self.identity]}
        with (patch.object(self.backend.requests, "get", side_effect=[http(tags), http({"version": "fixture-version"}), http(tags)]) as get,
              patch.object(self.backend.requests, "post", return_value=http({"parameters": "temperature 0.4", "template": "fixture"})) as post):
            runtime = self.backend.inspect_ollama(self.model, timeout=30, options={"temperature": 0})
        self.assertEqual(runtime["digest"], self.digest)
        self.assertEqual(runtime["server_version"], "fixture-version")
        self.assertEqual(runtime["model_parameters"], "temperature 0.4")
        self.assertEqual(runtime["options"], {"temperature": 0})
        self.assertEqual(runtime["identity_verification"], "server_reported_before_generation")
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/show")
        self.assertTrue(all(call.kwargs["timeout"] == 30 for call in get.call_args_list))

    def test_generated_response_preserves_text_and_metadata_with_verified_digest(self):
        payload = {"response": " A ", "model": self.model, "done": True, "eval_count": 7, "done_reason": "stop"}
        with (patch.object(self.backend, "ollama_model_identity", return_value=self.identity),
              patch.object(self.backend.requests, "post", return_value=http(payload)) as post):
            result = self.backend.ollama_response("invented prompt", model=self.model, expected_digest=self.digest,
                                                 timeout=10, options={"temperature": 0})
        self.assertEqual(result["response"], " A ")
        self.assertEqual(result["response_metadata"]["eval_count"], 7)
        self.assertEqual(result["identity_status"], "verified")
        self.assertEqual(result["digest_before"], result["digest_after"])
        self.assertEqual(post.call_args.kwargs["timeout"], 10)
        self.assertEqual(post.call_args.kwargs["json"]["options"], {"temperature": 0})

    def test_changed_digest_before_generation_prevents_request(self):
        with (patch.object(self.backend, "ollama_model_identity", return_value=dict(self.identity, digest="changed")),
              patch.object(self.backend.requests, "post") as post):
            with self.assertRaisesRegex(ValueError, "before the request"):
                self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
        post.assert_not_called()

    def test_changed_digest_or_failed_postcheck_retains_raw_answer(self):
        payload = {"response": "original answer", "model": self.model, "done": True}
        for after in (dict(self.identity, digest="changed"), RuntimeError("inspection timed out")):
            with self.subTest(after=after), patch.object(self.backend, "ollama_model_identity", side_effect=[self.identity, after]), patch.object(self.backend.requests, "post", return_value=http(payload)):
                result = self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
            self.assertEqual(result["response"], "original answer")
            self.assertEqual(result["identity_status"], "unverified")
            self.assertTrue(result["error"])

    def test_timeout_http_and_missing_tag_fail_without_retry(self):
        with patch.object(self.backend.requests, "post", side_effect=self.backend.requests.Timeout("fixture")) as post:
            with self.assertRaises(self.backend.requests.Timeout):
                self.backend.ollama_generate("prompt", model=self.model, timeout=3)
            post.assert_called_once()
        with patch.object(self.backend.requests, "get", return_value=http({"models": []})):
            with self.assertRaisesRegex(ValueError, "not uniquely installed"):
                self.backend.ollama_model_identity(self.model)

    def test_nonfinite_and_nonpositive_timeout_rejected_before_network(self):
        for timeout in (0, -1, float("nan"), float("inf"), True, "10"):
            with self.subTest(timeout=timeout), patch.object(self.backend.requests, "post") as post:
                with self.assertRaises(ValueError):
                    self.backend.ollama_generate("prompt", model=self.model, timeout=timeout)
                post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
