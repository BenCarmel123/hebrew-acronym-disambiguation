"""Offline table equivalence against the accepted S3a Git source."""

import ast
import csv
from dataclasses import asdict, dataclass, fields
import logging
from pathlib import Path
import socket
import subprocess
import tempfile
from typing import Iterable
import unittest
from unittest.mock import patch

from data_preprocess.common import candidates, hebrew_text

BASE = "2b9b84eb20b0be89d728b963cc753924e83b53ea"
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"BulletRow", "acronym_script", "read_existing", "write_csv", "summarise"}


def baseline_tables():
    """Execute only table definitions, excluding source access and all API calls."""
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:data_preprocess/wikipedia/source.py"],
        cwd=ROOT, text=True)
    nodes = [node for node in ast.parse(source).body
             if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in node.targets))]
    scope = {"__name__": __name__, "csv": csv, "dataclass": dataclass, "asdict": asdict,
             "Path": Path, "Iterable": Iterable, "hebrew_text": hebrew_text,
             "LOG": logging.getLogger("data_preprocess.wikipedia.source")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"git:{BASE}", "exec",
                 dont_inherit=True), scope)
    return scope, nodes


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


class CandidateTableEquivalenceTests(unittest.TestCase):
    def setUp(self):
        for name in ("connect", "connect_ex", "sendto"):
            guard = patch.object(socket.socket, name,
                                 side_effect=AssertionError("Network is forbidden in fixtures"))
            guard.start()
            self.addCleanup(guard.stop)
        self.old, self.nodes = baseline_tables()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_extracted_definitions_are_unchanged(self):
        current = ast.parse(Path(candidates.__file__).read_text())
        new_nodes = [node for node in current.body
                     if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES)
                     or (isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "FIELDS"
                                 for t in node.targets))]
        self.assertEqual([ast.dump(n) for n in new_nodes],
                         [ast.dump(n) for n in self.nodes])

    def test_dataclass_defaults_and_script_detection(self):
        args = ("אב", "fixture", "אלף בית", 1, "hebrew", True, False, "raw")
        self.assertEqual(asdict(candidates.BulletRow(*args)),
                         asdict(self.old["BulletRow"](*args)))
        self.assertEqual([(f.name, f.default) for f in fields(candidates.BulletRow)],
                         [(f.name, f.default) for f in fields(self.old["BulletRow"])])
        for value in ("", "123", "Ä", "אA", "AI", "ß", "אב", "a", "🙂"):
            with self.subTest(acronym=value):
                self.assertEqual(candidates.acronym_script(value),
                                 self.old["acronym_script"](value))
        self.assertEqual(candidates.FIELDS, self.old["FIELDS"])

    def test_csv_bytes_and_resume_sets_cache_and_order(self):
        results = []
        for label, row_type, writer, reader in (
            ("old", self.old["BulletRow"], self.old["write_csv"], self.old["read_existing"]),
            ("new", candidates.BulletRow, candidates.write_csv, candidates.read_existing),
        ):
            rows = fixture_rows(row_type)
            path = self.root / label / "candidates.csv"
            collected = writer(path, iter(rows[2:]), existing=rows[:2])
            loaded, done, cache = reader(path)
            results.append((path.read_bytes(), [asdict(r) for r in collected],
                            [asdict(r) for r in loaded], done, cache))
        self.assertEqual(*results)
        self.assertTrue(results[1][0].startswith(b"\xef\xbb\xbf"))
        self.assertEqual(results[1][4]["אלף בית"], 100)
        self.assertNotIn("מונח מומצא", results[1][4])

    def test_empty_missing_and_legacy_csv(self):
        missing = self.root / "missing.csv"
        empty = self.root / "empty.csv"
        empty.touch()
        legacy = self.root / "legacy.csv"
        legacy.write_text(
            "acronym,page_title,expansion,hits,script,initials_match,looks_like_person,raw_line\n"
            "אב,fixture,אלף בית,8,hebrew,unknown,True,raw\n", encoding="utf-8")
        header = self.root / "header.csv"
        header.write_text(",".join(candidates.FIELDS) + "\n", encoding="utf-8")
        for path in (missing, empty, legacy, header):
            with self.subTest(path=path.name):
                old_rows, old_done, old_cache = self.old["read_existing"](path)
                rows, done, cache = candidates.read_existing(path)
                self.assertEqual(([asdict(r) for r in rows], done, cache),
                                 ([asdict(r) for r in old_rows], old_done, old_cache))
        self.assertEqual(candidates.read_existing(legacy)[0][0].source, "wikipedia")

    def test_bad_csv_errors_are_preserved(self):
        malformed = self.root / "malformed.csv"
        malformed.write_text("acronym,hits\nאב,no-number\n", encoding="utf-8")
        invalid_hits = self.root / "invalid-hits.csv"
        candidates.write_csv(invalid_hits, fixture_rows(candidates.BulletRow)[:1])
        invalid_hits.write_bytes(invalid_hits.read_bytes().replace(b",50,", b",bad,"))
        for path in (malformed, invalid_hits, self.root):
            failures = []
            for reader in (self.old["read_existing"], candidates.read_existing):
                with self.assertRaises(Exception) as caught:
                    reader(path)
                failures.append((type(caught.exception), str(caught.exception)))
            self.assertEqual(*failures)

    def test_interrupted_stream_preserves_partial_bytes_and_resume(self):
        results = []
        for label, row_type, writer, reader in (
            ("old", self.old["BulletRow"], self.old["write_csv"], self.old["read_existing"]),
            ("new", candidates.BulletRow, candidates.write_csv, candidates.read_existing),
        ):
            rows = fixture_rows(row_type)
            path = self.root / label / "interrupted.csv"

            def interrupted():
                yield rows[1]
                raise RuntimeError("fixture interruption")

            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                writer(path, interrupted(), existing=rows[:1])
            partial = path.read_bytes()
            existing, done, cache = reader(path)
            writer(path, iter(rows[2:]), existing=existing)
            results.append((partial, path.read_bytes(), done, cache))
        self.assertEqual(*results)

    def test_summary_empty_and_custom_thresholds(self):
        for thresholds in ((10, 50, 100, 500), (50, 0, -1, 10, 50), ()):
            for empty in (False, True):
                old_rows = [] if empty else fixture_rows(self.old["BulletRow"])
                rows = [] if empty else fixture_rows(candidates.BulletRow)
                self.assertEqual(candidates.summarise(rows, thresholds),
                                 self.old["summarise"](old_rows, thresholds))

    def test_consumers_share_one_row_class(self):
        import data_preprocess
        from data_preprocess import dedupe_expansions, merge_sources
        from data_preprocess.wikipedia import source as wikipedia
        from data_preprocess.wiktionary import source as wiktionary
        for consumer in (data_preprocess, dedupe_expansions, merge_sources,
                         wikipedia, wiktionary):
            self.assertIs(consumer.BulletRow, candidates.BulletRow)
        self.assertIs(data_preprocess.summarise, candidates.summarise)
        self.assertIs(data_preprocess.write_csv, candidates.write_csv)


if __name__ == "__main__":
    unittest.main()
