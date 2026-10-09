"""Review export checks with invented answers and no model calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.test_review_export import export_review


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'manifest.json').write_text('{}')
        self.rows = [dict(item_id=str(i), acronym='AB', sentence='AB in context',
                          candidates='alpha|beta', gold_expansion='alpha') for i in range(395)]
        self.manifest = {'run_id': 'fixture-run', 'identity': {
            'cohort': 'full_test', 'rows': self.rows,
            'systems': [{'name': 'fixture', 'model': 'invented'}]}}
        self.records = [dict(item_id=str(i), task=task, response='alpha',
                             status='response_received', correct=i % 2 == 0, valid=True,
                             selected_candidate='alpha', shown_order=['alpha', 'beta'] if task == 'select' else [],
                             prompt_sha256='fixture') for i in range(395) for task in ('generate', 'select')]

    def export(self, filename):
        with patch('hebrew_acronyms.test_review_export.evaluation._load_manifest', return_value=self.manifest), \
             patch('hebrew_acronyms.test_review_export.evaluation.summarize_evaluation', return_value={'records': self.records}):
            return export_review([(self.root, 'fixture-run')], self.root / filename)

    def test_complete_queues_and_technical_failure_separation(self):
        self.records[0].update(status='incomplete_response', correct=False)
        data = self.export('review.json')
        self.assertEqual(len(data['items']), 790)
        negative, positive = set(data['queues']['diagnosis']), set(data['queues']['evaluation'])
        self.assertFalse(negative & positive)
        self.assertEqual(len(negative | positive), 790)
        self.assertEqual(len(data['queues']['calibration']), 20)
        self.assertEqual(data['coverage']['technical_failure'], 1)
        self.assertNotIn('annotations', data)

    def test_answer_text_and_run_bind_review_identity(self):
        first = self.export('first.json')
        self.records[0]['response'] = 'different answer'
        second = self.export('second.json')
        first_ids = {item['id'] for item in first['items']}
        second_ids = {item['id'] for item in second['items']}
        self.assertEqual(len(first_ids & second_ids), 789)
        self.manifest['run_id'] = 'unexpected-run'
        with self.assertRaisesRegex(ValueError, 'identified full-test'):
            self.export('wrong-run.json')

    def test_existing_bundle_is_never_overwritten(self):
        path = self.root / 'existing.json'
        path.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.export('existing.json')
        self.assertEqual(path.read_text(), 'preserve')
