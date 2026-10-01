"""Candidate-table persistence, resume and metadata contracts on invented rows."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from hebrew_acronyms.data_processing.common import candidates

def fixture_rows(row_type):
    """Invented rows cover source metadata, Unicode, errors and threshold ties."""
    values = [
        ("א״ב", "א״ב (פירושונים)", "אלף בית", 50, "hebrew", True, False,
         'שורה עם "ציטוט",\nוהמשך', "wikipedia", ""),
        ("א״ב", "א״ב", "ארמון בדוי", 10, "hebrew", False, True,
         "שורה שנייה", "wiktionary", "דוגמה"),
        ("AI", "AI", "מונח מומצא", -1, "latin", None, False,
         "failed fixture", "wikipedia", ""),
        ("123", "123", "סימן מומצא", 0, "other", None, False,
         "zero fixture", "wiktionary", ""),
        ("א״ב", "עמוד נוסף", "אלף בית", 100, "hebrew", True, False,
         "duplicate fixture", "wiktionary", ""),
    ]
    return [row_type(*value) for value in values]


class CandidateTableTests(unittest.TestCase):
    def test_roundtrip_preserves_source_metadata_order_and_hit_cache(self):
        rows = fixture_rows(candidates.BulletRow)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.csv"
            candidates.write_csv(path, rows[2:], existing=rows[:2])
            loaded, done, cache = candidates.read_existing(path)
            self.assertEqual([asdict(r) for r in loaded], [asdict(r) for r in rows])
            self.assertEqual(done, {r.page_title for r in rows})
            self.assertEqual(cache, {"אלף בית": 100, "ארמון בדוי": 10, "סימן מומצא": 0})
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(candidates.read_existing(path.parent / "missing.csv"), ([], set(), {}))

    def test_interrupted_stream_can_resume_without_losing_rows(self):
        rows = fixture_rows(candidates.BulletRow)
        def interrupted():
            yield rows[1]
            raise RuntimeError("fixture interruption")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.csv"
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                candidates.write_csv(path, interrupted(), existing=rows[:1])
            existing, _, _ = candidates.read_existing(path)
            self.assertEqual(existing, rows[:2])
            candidates.write_csv(path, rows[2:], existing=existing)
            self.assertEqual(candidates.read_existing(path)[0], rows)

    def test_legacy_source_default_and_invalid_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text("acronym,page_title,expansion,hits,script,initials_match,looks_like_person,raw_line\n"
                            "אב,fixture,אלף בית,8,hebrew,unknown,True,raw\n", encoding="utf-8")
            rows, _, _ = candidates.read_existing(path)
            self.assertEqual((rows[0].source, rows[0].domain, rows[0].initials_match), ("wikipedia", "", None))
            path.write_text(path.read_text().replace(",8,", ",bad,"))
            with self.assertRaises(ValueError):
                candidates.read_existing(path)

    def test_script_detection_and_thresholds(self):
        self.assertEqual([candidates.acronym_script(a) for a in ("אב", "AI", "123", "אA")],
                         ["hebrew", "latin", "other", "hebrew"])
        result = candidates.summarise(fixture_rows(candidates.BulletRow), (10, 50, 100))
        self.assertEqual(result["n_bullets"], 5)
        self.assertEqual(result["n_pages"], 3)
        self.assertEqual(result["thresholds"], {
            "10": {"bullets_at_or_above": 3, "acronyms_with_2plus_senses": 1},
            "50": {"bullets_at_or_above": 2, "acronyms_with_2plus_senses": 1},
            "100": {"bullets_at_or_above": 1, "acronyms_with_2plus_senses": 0}})
