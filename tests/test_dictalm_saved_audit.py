"""Regression checks for the preserved DictaLM evidence and new review bindings."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.dictalm_saved_audit import audit_saved_dictalm

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'saved-results'
REFERENCE = SAVED / 'colab-runs/evaluation-20261009-a319f9d/extracted/qwen/full-test/manifest.json'


class DictalmAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_saved_dictalm(SAVED, REFERENCE)

    def test_denominators_and_separate_diagnostic(self):
        a = self.audit
        self.assertEqual([(m['n_items'], m['n_correct'], m['n_valid']) for m in a['metrics']],
                         [(381, 125, 152), (381, 0, 0)])
        self.assertEqual(len(a['records']), 762)
        self.assertEqual(len({(r['item_id'], r['task']) for r in a['records']}), 762)
        self.assertEqual(a['diagnostic_selection']['leading_label_matches_gold'], 243)
        self.assertEqual(a['finish_reasons'], {'stop': 790})
        self.assertEqual(len(a['excluded_items']), 14)
        self.assertEqual(a['local_increment_ils'], 0)

    def test_bindings_preserve_journal_whitespace(self):
        summary = json.loads((SAVED / 'local-runs/dictalm-20261010/dictalm/full-test/summary.json').read_text())
        original = {(r['item_id'], r['task']): r for r in summary['records']}
        for record in self.audit['records']:
            self.assertEqual(record['response'], original[record['item_id'], record['task']]['response'])
            self.assertEqual(record['response_sha256'], hashlib.sha256(record['response'].encode()).hexdigest())
            self.assertIsNone(record['human_judgment'])
        self.assertEqual(sum(r['response'].startswith('\n') for r in self.audit['records'] if r['task'] == 'select'), 381)

    def test_changed_candidate_order_is_rejected(self):
        reference = json.loads(REFERENCE.read_text())
        request = next(r for r in reference['identity']['requests'] if r['task'] == 'select')
        request['shown_order'] = list(reversed(request['shown_order']))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, 'inputs, prompts or orders differ'):
                audit_saved_dictalm(SAVED, path)


if __name__ == '__main__':
    unittest.main()
