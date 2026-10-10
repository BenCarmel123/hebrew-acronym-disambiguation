"""Scored test cohort rules with invented item IDs."""
import unittest

from hebrew_acronyms.test_cohort import (
    COLLECTED_ITEMS, DOCUMENT_OVERLAP_IDS, SCORED_ITEMS, check_scored_cohort, is_scored, scored_ids,
)

SCORED = [f"item-{i}" for i in range(SCORED_ITEMS)]


class TestCohortTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual((COLLECTED_ITEMS, SCORED_ITEMS, len(DOCUMENT_OVERLAP_IDS)), (395, 381, 14))
        self.assertFalse(is_scored("kn-0116"))
        self.assertTrue(is_scored("kn-0002"))

    def test_saved_and_new_cohorts_give_the_same_scored_items_in_order(self):
        saved = sorted(DOCUMENT_OVERLAP_IDS)[:7] + SCORED + sorted(DOCUMENT_OVERLAP_IDS)[7:]
        self.assertEqual(scored_ids(saved), SCORED)
        self.assertEqual(scored_ids(SCORED), SCORED)

    def test_other_cohorts_are_rejected(self):
        partial = SCORED[:-1] + sorted(DOCUMENT_OVERLAP_IDS)[:1]
        for ids in (SCORED[:-1], SCORED + ["extra"], partial, SCORED[:-1] + SCORED[:1]):
            with self.subTest(n=len(ids)), self.assertRaises(ValueError):
                scored_ids(ids)
        with self.assertRaisesRegex(ValueError, "without document overlap"):
            check_scored_cohort(SCORED[:-1] + sorted(DOCUMENT_OVERLAP_IDS)[:1])


if __name__ == "__main__":
    unittest.main()
