"""Offline source binding and scoring checks; no model execution."""
import csv
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.saved_similarity_results import (
    SOURCE_PATH, SOURCE_SHA256, _validate_records, load_saved_similarity,
)
from hebrew_acronyms.test_cohort import SCORED_ITEMS


class SavedSimilarityResultsTests(unittest.TestCase):
    def setUp(self):
        self.cohort = [{"item_id": str(i), "acronym": "אב", "gold_expansion": "א ב",
                        "candidates": "א ב | ג ד"} for i in range(SCORED_ITEMS)]
        self.records = [{"item_id": row["item_id"], "acronym": row["acronym"],
                         "gold": row["gold_expansion"], "candidates": row["candidates"],
                         "shown_order": row["candidates"], "response": "א ב",
                         "correct": "True", "valid": "True"} for row in self.cohort]

    def test_saved_source_retains_245_of_381_and_portable_provenance(self):
        result = load_saved_similarity(Path(__file__).resolve().parents[1])
        metric = result["metric"]
        self.assertEqual((metric["n_correct"], metric["n_items"], metric["n_valid"]), (245, 381, 381))
        self.assertAlmostEqual(metric["accuracy"], 245 / 381)
        self.assertEqual(result["provenance"]["sha256"], SOURCE_SHA256)
        self.assertFalse(Path(result["provenance"]["path"]).is_absolute())
        self.assertIsNone(metric["run_id"])
        self.assertIsNone(metric["technical_failures"])

    def test_changed_source_bytes_fail_before_csv_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / SOURCE_PATH
            source.parent.mkdir(parents=True)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "identified saved CSV"):
                load_saved_similarity(directory)

    def test_duplicate_missing_or_different_cohort_ids_fail(self):
        for records in (self.records[:-1], self.records[:-1] + [self.records[0]],
                        self.records[:-1] + [self.records[-1] | {"item_id": "different"}]):
            with self.subTest(ids=[row["item_id"] for row in records[-2:]]):
                with self.assertRaises(ValueError):
                    _validate_records(records, self.cohort)

    def test_gold_acronym_and_candidates_must_match(self):
        for field in ("gold_expansion", "acronym", "candidates"):
            cohort = [self.cohort[0] | {field: "changed"}, *self.cohort[1:]]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "scored cohort"):
                _validate_records(self.records, cohort)

    def test_saved_score_and_candidate_membership_are_checked(self):
        for change in ({"correct": "False"}, {"response": "outside"}, {"valid": "False"},
                       {"shown_order": "ג ד | א ב"}):
            records = [self.records[0] | change, *self.records[1:]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                _validate_records(records, self.cohort)
        # A valid but incorrect prediction is distinct from an invalid prediction.
        self.records[0].update(response="ג ד", correct="False")
        _validate_records(self.records, self.cohort)

    def test_current_cohort_gold_change_rejected_by_loader(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "data/splits/test_items.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["gold_expansion"] = "changed"
        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory) / "cohort.csv"
            with cohort.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "scored cohort"):
                load_saved_similarity(root, cohort)
