"""Known-answer development metrics and strict letter decoding, invented data only."""
from copy import deepcopy
import sys
import unittest

from tests.run_experimental_study import disable_guard, enable_guard


class SharedStudyEvaluationTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        disable_guard()

    @classmethod
    def setUpClass(cls):
        enable_guard()
        from hebrew_acronyms.models.common import eval
        cls.eval = eval

    def setUp(self):
        self.rows = [{"item_id": str(i), "type_id": "large" if i < 4 else "small",
                      "sentence": "invented sentence", "target_raw": "XYZ", "span_start": 0, "span_end": 3,
                      "candidates": "gold|other", "gold_expansion": " gold "} for i in range(5)]
        encoder = [{"item_id": str(i), "condition": "dictabert", "status": "ok",
                    "selected_candidate": "gold" if i in {0, 1, 4} else "other", "selected_index": 0}
                   for i in range(5)]
        choices = [("response_received", " B "), ("response_received", "A or B"),
                   ("service_error", None), ("not_run", None), ("response_received", "B")]
        select = [{"item_id": str(i), "condition": "select", "status": status,
                   "raw_response": raw, "shown_order": ["other", "gold"]} for i, (status, raw) in enumerate(choices)]
        generation = [{"item_id": str(i), "condition": "generate", "status": "response_received",
                       "raw_response": "plausible alternate expansion requiring human review"} for i in range(5)]
        self.records = encoder + select + generation

    def test_strict_uppercase_letter_after_trim_only(self):
        parse = self.eval.parse_letter_choice
        self.assertEqual(parse(" \nB\t", 2), 1)
        self.assertEqual(parse("A", 1), 0)
        self.assertEqual(parse("Z", 26), 25)
        for response in ("a", "b", "A or B", "Answer: A", "AA", "", " ", "C", "A.", "א", None):
            with self.subTest(response=response):
                self.assertIsNone(parse(response, 2))
        self.assertIsNone(parse("A", 0));self.assertEqual(parse("A", 27), 0)

    def test_imbalanced_types_known_micro_macro_with_failures_and_unrun(self):
        encoder = self.eval.summarize_selection(self.rows, self.records, "dictabert")
        selection = self.eval.summarize_selection(self.rows, self.records, "select")
        self.assertEqual(encoder["micro_accuracy"], 3 / 5)
        self.assertEqual(encoder["macro_accuracy"], .75)
        self.assertEqual(selection["micro_accuracy"], 2 / 5)
        self.assertEqual(selection["macro_accuracy"], .625)
        self.assertEqual(selection["n_items"], 5)
        self.assertEqual(selection["n_attempted"], 4)
        self.assertEqual(selection["n_valid_predictions"], 2)
        self.assertEqual(selection["status_counts"], {"ok": 2, "parse_error": 1, "service_error": 1, "not_run": 1})
        self.assertTrue(selection["partial"])
        self.assertFalse(encoder["partial"])
        self.records[8]["status"] = "service_error"
        complete_with_errors = self.eval.summarize_selection(self.rows, self.records, "select")
        self.assertFalse(complete_with_errors["partial"])
        self.assertEqual(complete_with_errors["micro_accuracy"], .4)

    def test_gold_exact_trim_no_alias_or_substring(self):
        self.records[0]["selected_candidate"] = " gold "
        summary = self.eval.summarize_selection(self.rows, self.records, "dictabert")
        self.assertTrue(summary["details"][0]["correct"])
        for value in ("gold with explanation", "Gold", "alternative"):
            self.records[0]["selected_candidate"] = value
            self.assertFalse(self.eval.summarize_selection(self.rows, self.records, "dictabert")["details"][0]["correct"])

    def test_missing_type_or_gold_is_reported_without_invention(self):
        self.rows[0].pop("type_id")
        summary = self.eval.summarize_selection(self.rows, self.records, "dictabert")
        self.assertIsNone(summary["macro_accuracy"])
        self.assertEqual(summary["micro_accuracy"], .6)
        self.assertIn("missing type_id", summary["warnings"][0])
        self.rows[1].pop("gold_expansion")
        summary = self.eval.summarize_selection(self.rows, self.records, "dictabert")
        self.assertIsNone(summary["micro_accuracy"])
        self.assertTrue(any("missing gold" in warning for warning in summary["warnings"]))

    def test_disagreements_raw_answers_mappings_and_generation_manual_only(self):
        self.records[9]["raw_response"] = "A"  # other, while encoder chose gold
        inspection = self.eval.inspect_predictions(self.rows, self.records)
        self.assertEqual([item["item_id"] for item in inspection["disagreements"]], ["4"])
        item = inspection["items"][4]
        self.assertEqual(item["letter_mapping"], {"A": "other", "B": "gold"})
        self.assertEqual(item["selection_decoded"], "other")
        self.assertEqual(item["selection_raw"], "A")
        self.assertEqual(item["generation_score_status"], "manual_review_unscored")
        self.assertNotIn("generation_correct", item)
        self.assertIn("select", inspection["items"][1]["failures"])
        with self.assertRaises(ValueError):
            self.eval.summarize_selection(self.rows, self.records, "generate")

    def test_missing_duplicate_extra_records_rejected_in_shared_scorer(self):
        encoder = self.records[:5]
        for bad in (encoder[:-1], encoder + [encoder[0]], encoder + [dict(encoder[0], item_id="extra")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.eval.summarize_selection(self.rows, bad, "dictabert")


if __name__ == "__main__":
    unittest.main()
