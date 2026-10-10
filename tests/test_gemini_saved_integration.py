"""Offline regression checks against the identified, immutable Gemini evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.saved_test_comparison import compare_saved_tests
from hebrew_acronyms.test_review_export import export_review, review_coverage

ROOT = Path(__file__).resolve().parents[1] / 'saved-results'
SESSION_NAMES = [
    'evaluation-20261009-a319f9d', 'evaluation-gemini-retry-20261009-a319f9d',
    'evaluation-xai-qwen14-20261009-64f47b7', 'evaluation-xai-qwen14-20261009-e9c8ef7',
    'evaluation-gemini-381-20261010-79b767a',
]
SESSIONS = [(ROOT / 'colab-runs' / name / 'extracted', identity) for name, identity in zip(SESSION_NAMES, [
    '8aa951a4ac96c126335f40ddc869cc7b6b6010cc94d7be1cc30925f09be9aac0',
    '837973fb8065fa7c6467d9370fb170ad58df757d3dadd33ae105910ecd245deb',
    'f373731d81617852b35578fb78ab9d222598723f68e4907e9c5127392ebb5b64',
    '2b185ec5d6baf1975629d82ec4608da353c376aaadebfbe7a483e80f5e6ee2f3',
    '32f3aa9e0155ad993fa87df89bcce80d2f40276685968834c897db37c4e442b6',
])]
RUNS = {'qwen': 'cd1331b460364d4ebc341194c07dc1fe', 'openai': '358e1fe65fa048c191eda08c078fd4cd',
        'anthropic': '3248737a597b435181227711caea89d0', 'qwen14': 'f23e084aace743e8a61a1a71dfbb33d4',
        'xai': '885c4382ca904f0baca692ae9cc37ac9', 'gemini': '6658d323657f4df3aced950bfe6abb07'}
XAI_DIAGNOSTIC = (ROOT / 'colab-runs' / SESSION_NAMES[2] / 'diagnostic/extracted',
                  'e1436f143f73d846412d48f537cb8592afe1332de2879feba7f1572aae931d28',
                  '6b3e9430e2b9f9b5f0fb298b87447b48f4eefa8fb7c659a1b7981602297deb77')
LOCAL_DIAGNOSTIC = (ROOT / 'gemini-diagnostic-20261010-after-billing',
                    '208e223cf01dbbfab86d7bd94f85cf4079144c1d2747f521863822064f061d2a',
                    'e9de33e09adc215fa5443c19a0b28a7746ce2098f6002e60e1e281257616f18c')


def compare(sessions=SESSIONS, runs=RUNS, local=(LOCAL_DIAGNOSTIC,)):
    return compare_saved_tests(sessions, runs, diagnostic_sources=[XAI_DIAGNOSTIC],
                               local_gemini_diagnostics=local)


class GeminiSavedIntegrationTests(unittest.TestCase):
    def setUp(self):
        network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        network.start()
        self.addCleanup(network.stop)

    def test_scores_denominators_cost_and_previous_results_unchanged(self):
        prior = compare(SESSIONS[:-1], {k: v for k, v in RUNS.items() if k != 'gemini'}, ())
        current = compare()
        self.assertEqual(prior['metrics'], [m for m in current['metrics'] if m['system'] != 'gemini'])
        self.assertEqual(prior['models'], [m for m in current['models'] if m['system'] != 'gemini'])
        self.assertEqual(prior['baselines'], current['baselines'])
        self.assertEqual(prior['failures'], [f for f in current['failures'] if f['system'] != 'gemini'])
        gemini = {m['task']: m for m in current['metrics'] if m['system'] == 'gemini'}
        self.assertEqual((gemini['select']['n_correct'], gemini['select']['n_completed'], gemini['select']['n_items']),
                         (357, 381, 381))
        self.assertEqual((gemini['generate']['n_correct'], gemini['generate']['n_completed'], gemini['generate']['n_items']),
                         (207, 379, 381))
        failures = [f for f in current['failures'] if f['system'] == 'gemini']
        self.assertEqual({f['item_id'] for f in failures}, {'manual-0005', 'manual-0028'})
        self.assertEqual({f['finish_reason'] for f in failures}, {'MAX_TOKENS'})
        self.assertEqual(len(current['failures']), 17)
        self.assertEqual(current['unique_attempts'] - prior['unique_attempts'], 782)
        self.assertEqual(current['diagnostic_requests'], 3)
        self.assertAlmostEqual(current['cost']['total_accounted_ils'], 23.1276784, places=9)
        self.assertAlmostEqual(current['cost_sessions'][-1]['prior_ils'], 21.6620734, places=9)
        self.assertAlmostEqual(current['cost_sessions'][-1]['new_token_cost_ils'], 1.264833, places=9)
        self.assertAlmostEqual(current['cost_sessions'][-1]['new_uncertain_ils'], .200772, places=9)

    def test_missing_and_duplicate_diagnostic_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Prior expenditure'):
            compare(local=())
        with self.assertRaisesRegex(ValueError, 'Duplicate local diagnostic'):
            compare(local=(LOCAL_DIAGNOSTIC, LOCAL_DIAGNOSTIC))
        # A receipt with an internally consistent but wrong prior cannot bridge sessions.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [json.loads(s) for s in (LOCAL_DIAGNOSTIC[0] / 'diagnostic.jsonl').read_text().splitlines()]
            receipt = json.loads((LOCAL_DIAGNOSTIC[0] / 'receipt.json').read_text())
            events[0]['prior_accounted_ils'] += 4
            receipt['prior_accounted_ils'] += 4
            receipt['updated_accounted_ils'] += 4
            journal = root / 'diagnostic.jsonl'
            journal.write_text(''.join(json.dumps(e) + '\n' for e in events))
            journal_hash = hashlib.sha256(journal.read_bytes()).hexdigest()
            receipt['hashes']['diagnostic.jsonl'] = journal_hash
            path = root / 'receipt.json'; path.write_text(json.dumps(receipt))
            source = (root, journal_hash, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'Prior expenditure'):
                compare(local=(source,))

    def test_addon_preserves_exact_answers_and_has_no_judgments(self):
        saved = ROOT / 'human-review-gemini381-20261010/review-data.json'
        with tempfile.TemporaryDirectory() as temp:
            rebuilt = export_review([(SESSIONS[-1][0] / 'gemini/full-test', RUNS['gemini'])],
                                    Path(temp) / 'review.json', source_root=ROOT)
            self.assertEqual(rebuilt, json.loads(saved.read_text()))
            coverage = review_coverage(saved, Path(temp) / 'no-annotations.json')
        self.assertEqual(len(rebuilt['items']), 762)
        self.assertEqual(rebuilt['coverage'], {'automatic_positive': 564, 'automatic_nonpositive': 196,
                                             'technical_failure': 2})
        self.assertTrue(all(not Path(f['path']).is_absolute() for f in rebuilt['provenance']['files']))
        self.assertEqual(sum(r['total'] for r in coverage['rows']), 762)
        self.assertEqual(sum(r['reviewed'] for r in coverage['rows']), 0)
        self.assertEqual(sum(r['not_reviewed'] for r in coverage['rows']), 762)
        self.assertEqual({i['answers'][0]['binding']['run_id'] for i in rebuilt['items']}, {RUNS['gemini']})

    def test_bundled_evidence_matches_inventory(self):
        inventory = json.loads((ROOT / 'files.json').read_text())['files']
        for name, entry in inventory.items():
            content = (ROOT / name).read_bytes()
            self.assertEqual(len(content), entry['bytes'], name)
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry['sha256'], name)


if __name__ == '__main__':
    unittest.main()
