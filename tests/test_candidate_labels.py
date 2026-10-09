"""Every candidate survives display/strict decoding, including lists above 26."""
import unittest

from hebrew_acronyms.models.common.eval import candidate_labels, build_select_prompt, parse_letter_choice, inspect_predictions


class CandidateLabelTests(unittest.TestCase):
    def test_boundary_labels_and_strict_round_trip(self):
        labels = candidate_labels(703)
        self.assertEqual(labels[:3], ["A", "B", "C"])
        self.assertEqual(labels[25:30], ["Z", "AA", "AB", "AC", "AD"])
        self.assertEqual(labels[701:], ["ZZ", "AAA"])
        for index, label in enumerate(labels):
            self.assertEqual(parse_letter_choice(" \n" + label + "\t", len(labels)), index)
        for bad in ("AA.", "Answer: AA", "A or AA", "aa", "ＡＡ", "A A", "AE", ""):
            self.assertIsNone(parse_letter_choice(bad, 30))
        for count in (None, True, 0, -1, 1.5):
            self.assertIsNone(parse_letter_choice("A", count))

    def test_thirty_options_displayed_and_inspection_mapping_has_all(self):
        candidates = [f"invented option {index}" for index in range(30)]
        prompt = build_select_prompt("XYZ", "invented sentence", candidates)
        options = [line for line in prompt.splitlines() if ". invented option " in line]
        self.assertEqual(len(options), 30)
        self.assertEqual(options[-1], "AD. invented option 29")
        row = {"item_id": "fixture", "type_id": "type", "sentence": "XYZ", "target_raw": "XYZ",
               "span_start": 0, "span_end": 3, "candidates": "|".join(candidates), "gold_expansion": candidates[-1]}
        records = [{"item_id": "fixture", "condition": "select", "status": "response_received",
                    "shown_order": candidates, "raw_response": "AD"},
                   {"item_id": "fixture", "condition": "generate", "status": "not_run"}]
        inspected = inspect_predictions([row], records)
        self.assertEqual(inspected["metrics"][0]["micro_accuracy"], 1)
        self.assertEqual(inspected["items"][0]["letter_mapping"], dict(zip(candidate_labels(30), candidates)))


if __name__ == "__main__":
    unittest.main()
