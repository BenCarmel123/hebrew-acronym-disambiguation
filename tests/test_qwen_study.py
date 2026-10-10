"""Mocked Ollama identity, finite timeouts and response-evidence contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch

from tests.run_experimental_study import disable_guard, enable_guard


def http(payload):
    return Mock(json=Mock(return_value=payload), raise_for_status=Mock())


class QwenStudyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        disable_guard()

    @classmethod
    def setUpClass(cls):
        enable_guard()
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
        self.assertEqual(result["completion_status"], "complete")
        self.assertEqual(result["status"], "response_received")
        self.assertEqual(result["digest_before"], result["digest_after"])
        self.assertEqual(post.call_args.kwargs["timeout"], 10)
        self.assertEqual(post.call_args.kwargs["json"]["options"], {"temperature": 0})

    def test_incomplete_letter_retains_reason_and_fails_scoring_without_retry(self):
        from hebrew_acronyms import experimental_study as study
        cases = [(True, "length"), (False, "stop"), (False, None), (True, None), (True, "unknown"), (None, "stop")]
        for done, reason in cases:
            for identity_failure in (False, True):
                with self.subTest(done=done, reason=reason, identity_failure=identity_failure):
                    payload = {"response": "A", "model": self.model}
                    if done is not None:
                        payload["done"] = done
                    if reason is not None:
                        payload["done_reason"] = reason
                    after = dict(self.identity, digest="changed") if identity_failure else self.identity
                    with (patch.object(self.backend, "ollama_model_identity", side_effect=[self.identity, after]),
                          patch.object(self.backend.requests, "post", return_value=http(payload)) as post):
                        result = self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
                    post.assert_called_once()
                    self.assertEqual(result["response"], "A")
                    self.assertEqual(result["status"], "incomplete_response")
                    self.assertEqual(result["completion_status"], "incomplete")
                    self.assertEqual(result["response_metadata"].get("done_reason"), reason)
                    self.assertEqual(result["identity_status"], "unverified" if identity_failure else "verified")
                    self.assertTrue(result["completion_error"])
                    self.assertEqual(bool(result["identity_error"]), identity_failure)
                    artifact = self.collect(result)
                    record = next(r for r in artifact["records"] if r["condition"] == "qwen_select")
                    self.assertEqual(record["status"], "incomplete_response")
                    self.assertEqual(record["raw_response"], "A")
                    metric = study.inspect_results(artifact)["metrics"][1]
                    self.assertEqual(metric["n_valid_predictions"], 0)
                    self.assertEqual(metric["n_failures"], 1)
                    self.assertEqual(metric["micro_accuracy"], 0)
                    with tempfile.TemporaryDirectory() as temp:
                        path = study.save_artifact(artifact, Path(temp)/"run", create=True)
                        self.assertEqual(study.load_artifact(path, expected_run_id="completion"), artifact)

    def collect(self, result):
        from hebrew_acronyms import experimental_study as study
        # A singleton makes a completed A unambiguously correct.
        rows = study.fixture_rows()[1:]
        artifact = study.new_artifact(rows, "completion", {"qwen_model": self.model}, {})
        artifact["llm_runtimes"] = {"qwen": {"model": self.model, "digest": self.digest, "options": {}}}
        study.collect_responses(artifact, "select", lambda prompt: deepcopy(result))
        return artifact

    def test_completed_letter_is_decoded_and_historical_status_is_preserved(self):
        from hebrew_acronyms import experimental_study as study
        payload = {"response": "A", "model": self.model, "done": True, "done_reason": "stop"}
        with (patch.object(self.backend, "ollama_model_identity", return_value=self.identity),
              patch.object(self.backend.requests, "post", return_value=http(payload))):
            result = self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
        artifact = self.collect(result)
        self.assertEqual(study.inspect_results(artifact)["metrics"][1]["micro_accuracy"], 1)
        # Pre-fix artifacts lack completion_status; opening them cannot rewrite historical evidence.
        record = next(r for r in artifact["records"] if r["condition"] == "qwen_select")
        record["backend_metadata"].pop("completion_status")
        record["backend_metadata"]["response_metadata"]["done_reason"] = "length"
        with tempfile.TemporaryDirectory() as temp:
            path = study.save_artifact(artifact, Path(temp)/"old", create=True)
            before = path.read_bytes()
            loaded = study.load_artifact(path, expected_run_id="completion")
            self.assertEqual(loaded, artifact)
            self.assertEqual(record["status"], "response_received")
            self.assertEqual(path.read_bytes(), before)

    def test_changed_digest_before_generation_prevents_request(self):
        with (patch.object(self.backend, "ollama_model_identity", return_value=dict(self.identity, digest="changed")),
              patch.object(self.backend.requests, "post") as post):
            with self.assertRaisesRegex(ValueError, "before the request"):
                self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
        post.assert_not_called()

    def test_changed_digest_or_failed_postcheck_retains_raw_answer(self):
        payload = {"response": "original answer", "model": self.model, "done": True}
        for after in (dict(self.identity, digest="changed"), RuntimeError("private-key-url-and-account-details")):
            with self.subTest(after=after), patch.object(self.backend, "ollama_model_identity", side_effect=[self.identity, after]), patch.object(self.backend.requests, "post", return_value=http(payload)):
                result = self.backend.ollama_response("prompt", model=self.model, expected_digest=self.digest)
            self.assertEqual(result["response"], "original answer")
            self.assertEqual(result["identity_status"], "unverified")
            self.assertTrue(result["error"])
            self.assertNotIn("private-key-url-and-account-details", repr(result))

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
