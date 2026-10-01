"""Human review application and sentence filters on invented inputs."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from hebrew_acronyms.data_processing import apply_dev_review
from hebrew_acronyms.data_processing.common.csv_io import load_rows
from hebrew_acronyms.data_processing.common import filters


class DataSafetyTests(unittest.TestCase):
    def test_review_applies_human_decisions_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items, review, out = [root / name for name in ("items.csv", "review.csv", "out.csv")]
            items.write_text("item_id,acronym,candidates,gold_expansion,provisional_expansion,n_candidates\n"
                             "one,א״ב,אור בהיר|אור בחוץ,אור בהיר,אור בהיר,2\n"
                             "two,א״ב,אור בהיר|אור בחוץ,אור בהיר,אור בהיר,2\n"
                             "three,ג״ד,גן דשא|גן דק,גן דשא,גן דשא,2\n", encoding="utf-8-sig")
            review.write_text("item_id,verdict,note\none,wrong_sense,אור בחוץ\ntwo,unsure,\nthree,broken,\n", encoding="utf-8-sig")
            originals = [path.read_bytes() for path in (items, review)]
            with patch.object(sys, "argv", ["review", str(review), str(items), str(out)]), redirect_stdout(io.StringIO()):
                apply_dev_review.main()
            self.assertEqual([path.read_bytes() for path in (items, review)], originals)
            rows = load_rows(out)
            self.assertEqual([row["item_id"] for row in rows], ["one", "two"])
            self.assertEqual(rows[0]["gold_expansion"], "אור בחוץ")
            self.assertEqual(rows[1]["gold_expansion"], "אור בהיר")

    def test_sentence_filter_rejects_short_expansion_leak_and_urls(self):
        sentence = "לאחר הישיבה הארוכה נמסר כי א״ב ימשיך לפעול במקום גם במהלך השבוע הקרוב."
        self.assertTrue(filters.is_clean_sentence(sentence, "א״ב"))
        self.assertFalse(filters.is_clean_sentence("א״ב", "א״ב"))
        self.assertFalse(filters.is_clean_sentence(sentence + " https://example.org", "א״ב"))
        self.assertTrue(filters.mentions_expansion("אור בהיר", "אור בהיר"))
