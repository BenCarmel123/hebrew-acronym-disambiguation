"""Structural cleanup comparisons against the pre-cleanup Git revision."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from data_preprocess import apply_dev_review, build_splits, review_duplicates_cli
from data_preprocess.common import csv_io, filters
from model.common import pairs
from tests.reference import reference_git

BASE = "63b90acfae36e6b7fee8114760506868e29c681f"


def old_module(path, name):
    """Load a side-effect-free baseline module; its CLI is never invoked here."""
    source = reference_git(BASE, "show", f"{BASE}:{path}")
    module = ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    with patch.dict(sys.modules, {name: module}):
        exec(compile(source, f"git:{BASE}:{path}", "exec"), module.__dict__)
    return module


class CleanupEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("connect", "connect_ex", "sendto"):
            self.stack.enter_context(patch.object(
                socket.socket, name, side_effect=AssertionError("network forbidden")))
        for name in ("create_connection", "getaddrinfo"):
            self.stack.enter_context(patch.object(
                socket, name, side_effect=AssertionError("network forbidden")))
        self.old_pairs = old_module("model/common/pairs.py", "old_pairs")
        self.old_splits = old_module("data_preprocess/build_splits.py", "old_splits")
        self.old_reviewer = old_module(
            "data_preprocess/review_duplicates_cli.py", "data_preprocess.old_reviewer")

    def test_csv_readers_share_one_implementation_and_match_baseline(self):
        for loader in (pairs.load_rows, build_splits.load_rows,
                       review_duplicates_cli.load, apply_dev_review.load_rows):
            self.assertIs(loader, csv_io.load_rows)
        cases = ("", "a,b\r\n", 'a,b\r\nב,"טקסט, ושורה\r\nנוספת"\r\nא,ראשון\r\n',
                 "a,b\nעודף,שדה,שלישי\nחסר\n", "a,a\nראשון,שני\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            for encoding in ("utf-8", "utf-8-sig"):
                for content in cases:
                    with self.subTest(encoding=encoding, content=content):
                        path.write_bytes(content.encode(encoding))
                        expected = csv_io.load_rows(path)
                        for loader in (self.old_pairs.load_rows, self.old_splits.load_rows,
                                       self.old_reviewer.load):
                            self.assertEqual(loader(path), expected)
                            self.assertEqual([list(row) for row in loader(path)],
                                             [list(row) for row in expected])

    def test_csv_read_errors_match_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.csv"
            invalid.write_bytes(b"a\n\xff")
            for path in (invalid, Path(directory) / "missing.csv", Path(directory)):
                errors = []
                for loader in (csv_io.load_rows, self.old_pairs.load_rows,
                               self.old_splits.load_rows, self.old_reviewer.load):
                    with self.assertRaises((OSError, UnicodeError)) as caught:
                        loader(path)
                    errors.append((type(caught.exception), str(caught.exception)))
                self.assertEqual(errors, [errors[0]] * len(errors))

    def test_review_command_reads_and_outputs_match_baseline(self):
        old = old_module("data_preprocess/apply_dev_review.py", "old_apply_review")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items, review, out = (root / name for name in ("items.csv", "review.csv", "out.csv"))
            items.write_text(
                "item_id,acronym,candidates,gold_expansion,provisional_expansion,n_candidates\n"
                "one,א״ב,אור בהיר | אור בחוץ,אור בהיר,אור בהיר,2\n"
                "two,א״ב,אור בהיר | אור בחוץ,אור בהיר,אור בהיר,2\n"
                "three,ג״ד,גן דשא | גן דק,גן דשא,גן דשא,2\n",
                encoding="utf-8-sig")
            review.write_text(
                "item_id,verdict,note\none,wrong_sense,אור בחוץ\ntwo,unsure,\nthree,broken,\n",
                encoding="utf-8-sig")
            results = []
            for module in (old, apply_dev_review):
                output = io.StringIO()
                with patch.object(sys, "argv", ["apply-review", str(review), str(items), str(out)]):
                    with redirect_stdout(output):
                        module.main()
                results.append((out.read_bytes(), output.getvalue()))
            self.assertEqual(*results)

    def test_retained_sentence_filters_match_baseline(self):
        old = old_module("data_preprocess/common/filters.py", "data_preprocess.common.old_filters")
        template = "לאחר הישיבה הארוכה נמסר כי {} ימשיך לפעול במקום גם במהלך השבוע הקרוב."
        cases = [template.format(term) for term in
                 ("א״ב", 'א"ב', "בא״ב", "א״ב וגם א״ב", "א״ב וגם ג״ד", "א״ב (אור בהיר)",
                  "אור בהיר (א״ב)", "א״ב – אור בהיר", "ראשי תיבות א״ב", "ר׳ א״ב", "ב׳ א״ב")]
        cases += ["א״ב", "א״ב " * 90, "== כותרת א״ב ==", template.format("א״ב") + " https://example.org"]
        for sentence in cases:
            with self.subTest(sentence=sentence):
                for name in ("is_clean_sentence", "mentions_acronym", "mentions_expansion"):
                    self.assertEqual(getattr(filters, name)(sentence, "א״ב"),
                                     getattr(old, name)(sentence, "א״ב"))
                self.assertEqual(filters.is_numeral_reference(sentence),
                                 old.is_numeral_reference(sentence))
        text = "\n".join(cases)
        self.assertEqual(list(filters.page_sentences(text, "א״ב")),
                         list(old.page_sentences(text, "א״ב")))
        self.assertEqual(list(filters.page_sentences(text, "א״ב", is_clean=lambda *args: True)),
                         list(old.page_sentences(text, "א״ב", is_clean=lambda *args: True)))


if __name__ == "__main__":
    unittest.main()
