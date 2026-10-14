"""Uniform post-hoc scoring, failure handling and preservation of original evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from hebrew_acronyms.leading_label_analysis import derive_saved_comparison, leading_label_result, RULE

ROOT = Path(__file__).resolve().parents[1]


class LeadingLabelTests(unittest.TestCase):
    def record(self, response, status="response_received", **metadata):
        return dict(task="select", response=response, status=status,
                    response_metadata=metadata, shown_order=[f"candidate-{i}" for i in range(30)],
                    correct=False, valid=False)

    def test_multiletter_range_and_token_boundary(self):
        for response, index in (("A", 0), ("  AA. explanation", 26), ("AD\nextra", 29),
                                ("A: contradictory extra text", 0)):
            with self.subTest(response=response):
                result = leading_label_result(self.record(response), f"candidate-{index}")
                self.assertTrue(result["leading_label_valid"])
                self.assertTrue(result["leading_label_correct"])
        for response in ("AE", "A1", "A_name", "Answer A", "a", "(A)", "", "הבחירה A"):
            with self.subTest(response=response):
                self.assertFalse(leading_label_result(self.record(response), "candidate-0")["leading_label_valid"])

    def test_failed_or_unverified_correct_prefix_never_earns_credit(self):
        for status in ("incomplete_response", "missing_response", "blocked", "service_error", "not_run", "ambiguous"):
            with self.subTest(status=status):
                result = leading_label_result(self.record("A", status), "candidate-0")
                self.assertFalse(result["leading_label_correct"])
                self.assertFalse(result["leading_label_valid"])
        self.assertFalse(leading_label_result(self.record("A", identity_status="unverified"),
                                             "candidate-0")["leading_label_correct"])

    def test_does_not_modify_response_or_original_decisions(self):
        record = self.record(" A. extra text ")
        before = deepcopy(record)
        result = leading_label_result(record, "candidate-0")
        self.assertTrue(result["leading_label_correct"])
        self.assertEqual(record, before)
        self.assertFalse(record["correct"])
        self.assertFalse(leading_label_result(record, "candidate-1")["leading_label_correct"])
        with self.assertRaises(ValueError):
            leading_label_result(record | {"task": "generate"}, "candidate-0")

    def test_saved_seven_models_uniform_rule_and_historical_preservation(self):
        inventory = json.loads((ROOT / "saved-results/files.json").read_text(encoding="utf-8"))["files"]
        before = {name: hashlib.sha256((ROOT / "saved-results" / name).read_bytes()).hexdigest()
                  for name in inventory}
        result = derive_saved_comparison(ROOT)
        expected = dict(qwen=223, qwen14=266, dictalm=243, gemini=357, openai=317, anthropic=339, xai=351)
        rows = [m for m in result["metrics"] if m["task"] == "select" and m["system"] in expected]
        self.assertEqual({m["system"]: m["n_correct"] for m in rows}, expected)
        self.assertEqual(len(result["selection_records"]), 2667)
        for row in rows:
            self.assertEqual(row["score"], RULE)
            self.assertEqual(row["n_items"], 381)
            self.assertEqual(row["strict_n_correct"], 0 if row["system"] == "dictalm" else row["n_correct"])
            self.assertEqual(row["n_changed_correct"], 243 if row["system"] == "dictalm" else 0)
        for r in result["selection_records"]:
            if r["status"] != "response_received":
                self.assertFalse(r["leading_label_correct"])
        after = {name: hashlib.sha256((ROOT / "saved-results" / name).read_bytes()).hexdigest()
                 for name in inventory}
        self.assertEqual(before, after)
        self.assertEqual(before, {name: info["sha256"] for name, info in inventory.items()})


if __name__ == "__main__":
    unittest.main()
