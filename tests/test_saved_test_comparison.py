"""Offline contracts for saved test comparison; all examples are invented."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms import staged_evaluation as staged
from hebrew_acronyms import test_evaluation as evaluation
from hebrew_acronyms.saved_test_comparison import compare_saved_tests, result_table
from hebrew_acronyms.test_baselines import run_test_baselines


class SavedTestComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.test = self.root / "test_items.csv"
        self.dev = self.root / "dev.csv"
        for path, count in ((self.test, 395), (self.dev, 10)):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["item_id", "acronym", "sentence", "gold_expansion", "candidates"])
                writer.writeheader()
                writer.writerows({"item_id": f"item-{i}", "acronym": "אבג", "sentence": "הקשר אבג לדוגמה",
                                  "gold_expansion": "אחד", "candidates": "אחד|שניים"} for i in range(count))
        candidates = self.root / "candidate_table.csv"
        candidates.write_text("acronym,expansion,rank,mined_items\nאבג,אחד,1,1\nאבג,שניים,2,0\n")
        self.sources, self.expected, self.directories = [], {}, []
        rates = {name: {"input": 1, "output": 1} for name in staged.SYSTEM_NAMES}
        reserves = dict.fromkeys(staged.SYSTEM_NAMES, .01)
        network = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        prior = 4.0
        for index, (name, model, settings) in enumerate((
                ("openai", "gpt-4.1-mini-2025-04-14", {"timeout": 120, "max_output_tokens": 512, "temperature": 0}),
                ("anthropic", "claude-haiku-5-5", {"timeout": 120, "max_output_tokens": 1024, "effort": "low"}))):
            root = self.root / f"session-{index}"
            session = staged.prepare_session(root, self.dev, self.test, code_revision="a" * 40,
                                             prior_spend_ils=prior, rates=rates, reserves=reserves)
            system = {"name": name, "provider": name, "model": model, "settings": settings}
            manifest = staged._prepare(root, session, system, "full_test", None)
            directory = root / name / "full-test"
            response = {"status": "incomplete_response", "response": "אחד", "finish_reason": "length",
                        "identity_status": "verified", "usage_metadata": {"prompt_tokens": 10, "completion_tokens": 2}}
            if index:
                response = {"status": "response_received", "response": "אחד", "finish_reason": "end_turn"}
            with patch.object(evaluation, "_call", return_value=response):
                evaluation.run_evaluation(directory, code_revision="a" * 40, max_new_calls=1)
            self.sources.append((root, session["identity_sha256"]))
            self.expected[name] = manifest["run_id"]
            self.directories.append(directory)
            prior = staged.session_summary(root)["total_accounted_ils"]
        run_test_baselines(self.test, candidates, self.sources[0][0] / "baselines", code_revision="a" * 40)

    def compare(self):
        return compare_saved_tests(self.sources, self.expected)

    def mutate_manifest(self, mutation):
        directory = self.directories[1]
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        mutation(manifest["identity"])
        manifest["identity_sha256"] = evaluation._hash(manifest["identity"])
        path.write_text(json.dumps(manifest))
        journal = directory / "attempts.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        for event in events:
            event["identity_sha256"] = manifest["identity_sha256"]
        journal.write_text("".join(json.dumps(e) + "\n" for e in events))


    def test_explicit_legacy_and_extended_collectors_can_be_compared(self):
        root = self.sources[1][0]
        path = root / "session.json"
        session = json.loads(path.read_text())
        session["identity"]["code_sha256"]["test_evaluation.py"] = (
            "1434bedc4dac4a79e681d3178d8c2eb2f408ba12d3931255c09f90ca3f8abbff")
        session["identity"]["code_revision"] = "b" * 40
        session["identity_sha256"] = evaluation._hash(session["identity"])
        path.write_text(json.dumps(session))
        self.sources[1] = (root, session["identity_sha256"])
        def legacy(identity):
            identity["code_sha256"] = session["identity"]["code_sha256"]
            identity["code_revision"] = "b" * 40
            identity["metadata"]["session_identity"] = session["identity_sha256"]
        self.mutate_manifest(legacy)
        result = self.compare()
        self.assertEqual({m["collection_revision"] for m in result["models"]}, {"a" * 40, "b" * 40})
        self.assertAlmostEqual(result["cost"]["total_accounted_ils"], 4.040048)

    def test_unknown_collector_is_rejected(self):
        with patch("hebrew_acronyms.saved_test_comparison.COMPATIBLE_COLLECTOR_SHA256", set()):
            with self.assertRaisesRegex(ValueError, "Unreviewed collection"):
                self.compare()

    def test_complete_denominators_and_cost_carry_without_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.compare()
        self.assertEqual(result["unique_attempts"], 2)
        self.assertTrue(all(m["n_items"] == 395 for m in result["metrics"]))
        self.assertEqual(len(result["failures"]), 1579)
        self.assertAlmostEqual(result["cost"]["token_cost_ils"], .000048)
        self.assertAlmostEqual(result["cost"]["uncertain_attempt_allowances_ils"], .04)
        self.assertAlmostEqual(result["cost"]["total_accounted_ils"], 4.040048)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_duplicate_session_copy_is_rejected(self):
        copied = self.root / "copy"
        shutil.copytree(self.sources[0][0], copied)
        self.sources.append((copied, self.sources[0][1]))
        with self.assertRaisesRegex(ValueError, "Duplicate session"):
            self.compare()

    def test_explicit_continuation_counts_only_new_run_and_preserves_prior(self):
        original = self.sources[0][0]
        resumed = self.root / "resumed"
        shutil.copytree(original, resumed)
        session = json.loads((resumed / "session.json").read_text())
        system = {"name": "gemini", "provider": "gemini", "model": "gemini-3.8-flash",
                  "settings": {"timeout": 120, "generation_config": {"maxOutputTokens": 1024}}}
        manifest = staged._prepare(resumed, session, system, "full_test", None)
        directory = resumed / "gemini/full-test"
        response = {"status": "response_received", "response": "אחד", "finish_reason": "STOP",
                    "identity_status": "verified", "usage_metadata": {"promptTokenCount": 10,
                    "candidatesTokenCount": 2, "totalTokenCount": 12}}
        with patch.object(evaluation, "_call", return_value=response):
            evaluation.run_evaluation(directory, code_revision="a" * 40, max_new_calls=1)
        expected = self.expected | {"gemini": manifest["run_id"]}
        continuation = [(resumed, session["identity_sha256"])]
        result = compare_saved_tests(self.sources, expected, continuation_sources=continuation)
        increment = evaluation.summarize_evaluation(directory)["charged_or_reserved_usd"] * 4
        self.assertAlmostEqual(result["cost"]["total_accounted_ils"], self.compare()["cost"]["total_accounted_ils"] + increment)
        self.assertEqual(result["unique_attempts"], 3)
        self.assertTrue(result["cost_sessions"][-1]["continuation"])
        self.assertEqual(result["cost_sessions"][-1]["prior_ils"], 4)
        old_journal = resumed / "openai/full-test/attempts.jsonl"
        with old_journal.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "previously counted evidence"):
            compare_saved_tests(self.sources, expected, continuation_sources=continuation)

    def test_continuation_without_new_evidence_is_rejected(self):
        copied = self.root / "continuation-copy"
        shutil.copytree(self.sources[0][0], copied)
        with self.assertRaisesRegex(ValueError, "no new run"):
            compare_saved_tests(self.sources, self.expected,
                                continuation_sources=[(copied, self.sources[0][1])])

    def test_duplicate_attempt_across_runs_is_rejected(self):
        first = json.loads((self.directories[0] / "attempts.jsonl").read_text().splitlines()[0])["attempt_id"]
        path = self.directories[1] / "attempts.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        for event in events:
            event["attempt_id"] = first
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
        with self.assertRaisesRegex(ValueError, "Duplicate attempt"):
            self.compare()

    def test_prompt_candidate_order_and_input_mismatches_are_rejected(self):
        original = {p: p.read_bytes() for p in self.directories[1].iterdir() if p.is_file()}
        def prompt(identity):
            identity["requests"][0]["prompt"] += "changed"
            identity["requests"][0]["prompt_sha256"] = hashlib.sha256(identity["requests"][0]["prompt"].encode()).hexdigest()
        for mutation in (prompt,
                         lambda i: i["requests"][1]["shown_order"].reverse(),
                         lambda i: i["rows"][0].update(sentence="different input"),
                         lambda i: i.update(protocol="different protocol")):
            with self.subTest(mutation=mutation):
                for path, data in original.items():
                    path.write_bytes(data)
                self.mutate_manifest(mutation)
                with self.assertRaisesRegex(ValueError, "Test items, protocol, prompts or candidate orders differ"):
                    self.compare()

    def test_wrong_prior_and_wrong_run_are_rejected(self):
        path = self.sources[1][0] / "session.json"
        session = json.loads(path.read_text())
        session["identity"]["prior_spend_ils"] += 4
        session["identity_sha256"] = evaluation._hash(session["identity"])
        path.write_text(json.dumps(session))
        self.sources[1] = (self.sources[1][0], session["identity_sha256"])
        with self.assertRaisesRegex(ValueError, "Prior expenditure"):
            self.compare()

    def test_expected_run_id_and_baseline_gold_are_checked(self):
        self.expected["openai"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.compare()
        self.expected["openai"] = evaluation._load_manifest(self.directories[0])["run_id"]
        path = self.sources[0][0] / "baselines/baselines.json"
        baseline = json.loads(path.read_text())
        baseline["details"][0]["gold"] = "different"
        path.write_text(json.dumps(baseline))
        with self.assertRaisesRegex(ValueError, "Baseline"):
            self.compare()

    def test_identified_diagnostic_is_counted_once_without_session_total_duplication(self):
        before = self.compare()
        directory = self.root / 'diagnostic'
        directory.mkdir()
        start = {'event': 'start', 'max_requests': 1, 'reserved_ils': .04,
                 'source_session': self.sources[-1][0].name, 'source_revision': 'a' * 40,
                 'prior_accounted_ils': before['cost']['total_accounted_ils']}
        finish = {'event': 'finish_recovered_from_kernel',
                  'result': {'attempts': 1, 'http_status': 400, 'usage_metadata': None},
                  'accounted_ils_including_reserve': start['prior_accounted_ils'] + .04}
        paths = [directory / 'diagnostic.jsonl', directory / 'recovered-finish.json']
        for path, value in zip(paths, (start, finish)):
            path.write_text(json.dumps(value))
        source = (directory, *(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths))
        result = compare_saved_tests(self.sources, self.expected, diagnostic_sources=[source])
        self.assertAlmostEqual(result['cost']['total_accounted_ils'], 4.080048)
        self.assertAlmostEqual(result['cost']['uncertain_attempt_allowances_ils'], .08)
        with self.assertRaisesRegex(ValueError, 'Duplicate diagnostic'):
            compare_saved_tests(self.sources, self.expected, diagnostic_sources=[source, source])
        paths[0].write_text(json.dumps(start | {'reserved_ils': 0}))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            compare_saved_tests(self.sources, self.expected, diagnostic_sources=[source])

    def test_table_escapes_model_text(self):
        self.assertIn("&lt;script&gt;", result_table([{"answer": "<script>"}], {"answer": "Answer"}))


if __name__ == "__main__":
    unittest.main()
