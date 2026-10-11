"""Simulated cp1252 default: preserve Hebrew evidence and derived export bytes.

This tests the saved-analysis path, not an actual Windows execution. Only text
opens under evidence/output roots receive the simulated default; dependencies
and unrelated interpreter configuration keep their host behavior.
"""
import builtins
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.saved_analysis import load_saved_analysis, export_dictalm
from hebrew_acronyms.saved_human_review import load_final_review, export_review_analysis
from hebrew_acronyms.dictalm_saved_audit import audit_saved_dictalm

ROOT = Path(__file__).resolve().parents[1]


class SavedAnalysisEncodingTests(unittest.TestCase):
    def test_hebrew_evidence_and_exports_with_simulated_cp1252_default(self):
        saved = load_saved_analysis(ROOT)
        review = load_final_review(ROOT / 'saved-results', ROOT / 'data/splits/test_items.csv')
        audit = audit_saved_dictalm(ROOT / 'saved-results', saved['original_manifest'])
        self.assertTrue(any(any('\u0590' <= c <= '\u05ff' for c in r['raw'])
                            for r in review['records']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline, simulated = root / 'baseline', root / 'simulated'
            baseline.mkdir(); simulated.mkdir()
            export_review_analysis(review, baseline)
            export_dictalm(audit, saved['comparison'], baseline)
            probe = root / 'probe.txt'
            probe.write_bytes('ש'.encode('utf-8'))
            original_io_open, original_open = io.open, builtins.open
            original_path_open = Path.open
            original_read_text, original_write_text = Path.read_text, Path.write_text

            def wrapped(original):
                def open_with_default(file, mode='r', buffering=-1, encoding=None,
                                      errors=None, newline=None, closefd=True, opener=None):
                    if not isinstance(file, int) and 'b' not in mode and encoding in (None, 'locale'):
                        path = Path(file).resolve()
                        if path.is_relative_to(ROOT) or path.is_relative_to(root):
                            encoding = 'cp1252'
                    return original(file, mode, buffering, encoding, errors, newline, closefd, opener)
                return open_with_default

            def path_open(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
                resolved = path.resolve()
                if ('b' not in mode and encoding is None
                        and (resolved.is_relative_to(ROOT) or resolved.is_relative_to(root))):
                    encoding = 'cp1252'
                return original_path_open(path, mode, buffering, encoding, errors, newline)

            def text_method(original):
                def call(path, *args, encoding=None, errors=None, **kwargs):
                    resolved = path.resolve()
                    if encoding is None and (resolved.is_relative_to(ROOT) or resolved.is_relative_to(root)):
                        encoding = 'cp1252'
                    return original(path, *args, encoding=encoding, errors=errors, **kwargs)
                return call

            with (patch.object(Path, 'open', path_open),
                  patch.object(Path, 'read_text', text_method(original_read_text)),
                  patch.object(Path, 'write_text', text_method(original_write_text)),
                  patch('io.open', wrapped(original_io_open)),
                  patch('builtins.open', wrapped(original_open))):
                # Prove the simulated default is active, and corrupts this Hebrew probe.
                with probe.open() as handle:
                    self.assertEqual(handle.encoding, 'cp1252')
                    self.assertNotEqual(handle.read(), 'ש')
                self.assertNotEqual(probe.read_text(), 'ש')
                replay = load_saved_analysis(ROOT)
                replay_review = load_final_review(ROOT / 'saved-results', ROOT / 'data/splits/test_items.csv')
                replay_audit = audit_saved_dictalm(ROOT / 'saved-results', replay['original_manifest'])
                self.assertEqual(replay, saved)
                self.assertEqual(replay_review, review)
                self.assertEqual(replay_audit, audit)
                export_review_analysis(replay_review, simulated)
                export_dictalm(replay_audit, replay['comparison'], simulated)
            self.assertEqual({p.name for p in baseline.iterdir()}, {p.name for p in simulated.iterdir()})
            for path in baseline.iterdir():
                self.assertEqual(path.read_bytes(), (simulated / path.name).read_bytes(), path.name)
                self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'), path.name)


if __name__ == '__main__':
    unittest.main()
