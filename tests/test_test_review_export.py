"""Review export checks with invented answers and no model calls."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.test_cohort import DOCUMENT_OVERLAP_IDS, SCORED_ITEMS
from hebrew_acronyms.test_review_export import export_review, export_encoder_review, review_coverage

# A saved 395-item run: the scored items followed by the 14 excluded ones.
ITEM_IDS = [str(i) for i in range(SCORED_ITEMS)] + sorted(DOCUMENT_OVERLAP_IDS)


class ReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'manifest.json').write_text('{}')
        self.rows = [dict(item_id=str(i), acronym='AB', sentence='AB in context',
                          candidates='alpha|beta', gold_expansion='alpha') for i in ITEM_IDS]
        self.manifest = {'run_id': 'fixture-run', 'identity': {
            'cohort': 'full_test', 'rows': self.rows,
            'systems': [{'name': 'fixture', 'model': 'invented'}]}}
        self.records = [dict(item_id=str(i), task=task, response='alpha',
                             status='response_received', correct=ITEM_IDS.index(i) % 2 == 0, valid=True,
                             selected_candidate='alpha', shown_order=['alpha', 'beta'] if task == 'select' else [],
                             prompt_sha256='fixture') for i in ITEM_IDS for task in ('generate', 'select')]

    def export(self, filename):
        with patch('hebrew_acronyms.test_review_export.evaluation._load_manifest', return_value=self.manifest), \
             patch('hebrew_acronyms.test_review_export.evaluation.summarize_evaluation', return_value={'records': self.records}):
            return export_review([(self.root, 'fixture-run')], self.root / filename)

    def test_complete_queues_and_technical_failure_separation(self):
        self.records[0].update(status='incomplete_response', correct=False)
        data = self.export('review.json')
        self.assertEqual(len(data['items']), 2 * SCORED_ITEMS)
        self.assertFalse(DOCUMENT_OVERLAP_IDS & {item['original_item_id'] for item in data['items']})
        negative, positive = set(data['queues']['diagnosis']), set(data['queues']['evaluation'])
        self.assertFalse(negative & positive)
        self.assertEqual(len(negative | positive), 2 * SCORED_ITEMS)
        self.assertEqual(len(data['queues']['calibration']), 20)
        self.assertEqual(data['coverage']['technical_failure'], 1)
        self.assertNotIn('annotations', data)

    def test_answer_text_and_run_bind_review_identity(self):
        first = self.export('first.json')
        self.records[0]['response'] = 'different answer'
        second = self.export('second.json')
        first_ids = {item['id'] for item in first['items']}
        second_ids = {item['id'] for item in second['items']}
        self.assertEqual(len(first_ids & second_ids), 2 * SCORED_ITEMS - 1)
        self.manifest['run_id'] = 'unexpected-run'
        with self.assertRaisesRegex(ValueError, 'identified full-test'):
            self.export('wrong-run.json')

    def test_existing_bundle_is_never_overwritten(self):
        path = self.root / 'existing.json'
        path.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.export('existing.json')
        self.assertEqual(path.read_text(), 'preserve')

    def test_coverage_without_judgments_keeps_full_denominators(self):
        self.export('review.json')
        result = review_coverage(self.root / 'review.json', self.root / 'absent.json')
        self.assertFalse(result['annotations_present'])
        self.assertIsNone(result['pace'])
        self.assertEqual(sum(r['total'] for r in result['rows']), 2 * SCORED_ITEMS)
        self.assertEqual(sum(r['unresolved'] for r in result['rows']), 2 * SCORED_ITEMS)
        self.assertEqual(sum(r['reviewed'] for r in result['rows']), 0)

    def test_only_explicit_completed_labels_are_counted_and_bound_to_source(self):
        data = self.export('review.json')
        item = next(i for i in data['items'] if not i['answers'][0]['auto_score'])
        answer = item['answers'][0]
        annotation = {'annotator': 'fixture reviewer', 'updated_at': '2026-10-09T10:01:00+00:00',
                      'answers': {answer['id']: {'quality': 'correct', 'system_id': answer['system_id']}}}
        record = {'completion': {'status': 'partial'}, 'draft': annotation, 'reviewed': annotation}
        bundle = {'schema_version': 'human-review-v2', 'dataset_id': data['dataset_id'],
                  'provenance': data['provenance'], 'records': {item['id']: record}}
        path = self.root / 'labels.json'
        path.write_text(json.dumps(bundle))
        self.assertEqual(sum(r['reviewed'] for r in review_coverage(self.root/'review.json', path)['rows']), 0)
        record['completion']['status'] = 'complete'
        path.write_text(json.dumps(bundle))
        rows = review_coverage(self.root/'review.json', path)['rows']
        self.assertEqual(sum(r['reviewed'] for r in rows), 1)
        self.assertEqual(sum(r['automatic_nonpositive_human_correct'] for r in rows), 1)
        bundle['dataset_id'] = 'different-source'
        path.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, 'source or schema mismatch'):
            review_coverage(self.root/'review.json', path)

    def test_changed_response_cannot_retain_an_old_judgment_identity(self):
        data = self.export('review.json')
        data['items'][0]['answers'][0]['raw'] = 'changed response'
        (self.root/'review.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            review_coverage(self.root/'review.json', self.root/'absent.json')

    def test_all_positive_group_retains_zero_negative_denominator(self):
        for record in self.records:
            record['correct'] = True
        self.export('positive-only.json')
        rows = review_coverage(self.root/'positive-only.json', self.root/'absent.json')['rows']
        self.assertTrue(all(row['automatic_nonpositive_total'] == 0 for row in rows))
        self.assertEqual(sum(row['automatic_positive_total'] for row in rows), 2 * SCORED_ITEMS)

    def test_coverage_of_an_earlier_full_bundle_ignores_excluded_items(self):
        # Bundles exported before the exclusion queued all 395 items.
        with patch('hebrew_acronyms.test_review_export.is_scored', return_value=True):
            data = self.export('earlier.json')
        self.assertEqual(len(data['items']), 2 * len(ITEM_IDS))
        rows = review_coverage(self.root/'earlier.json', self.root/'absent.json')['rows']
        self.assertEqual(sum(row['total'] for row in rows), 2 * SCORED_ITEMS)

    def test_encoder_queue_is_selection_only_and_binds_actual_candidate(self):
        predictions = [dict(selected_candidate='alpha', status='ok', correct=i % 2 == 0)
                       for i in range(len(ITEM_IDS))]
        (self.root/'predictions.jsonl').write_text('fixture')
        test_path = self.root/'test.csv'
        test_path.write_text('fixture')
        with patch('hebrew_acronyms.encoder_test_results.load_encoder_test',
                   return_value={'records': predictions, 'rows': self.rows}) as loader:
            bundle = export_encoder_review(self.root, test_path, self.root/'encoder.json',
                                           expected_run_id='encoder-fixture', expected_identity='identified',
                                           expected_predictions_sha256='predictions-hash')
        self.assertEqual(loader.call_args.kwargs['expected_identity'], 'identified')
        self.assertEqual(len(bundle['items']), SCORED_ITEMS)
        self.assertEqual(len(bundle['queues']['calibration']), 20)
        self.assertEqual(set(bundle['queues']['diagnosis']) | set(bundle['queues']['evaluation']),
                         {item['id'] for item in bundle['items']})
        for item in bundle['items']:
            answer = item['answers'][0]
            self.assertEqual(answer['task'], 'selection')
            self.assertEqual(answer['raw'], 'alpha')
            self.assertEqual(answer['auto_score_rule'], 'exact_candidate_match')
            self.assertEqual(answer['binding']['run_id'], 'encoder-fixture')
        result = review_coverage(self.root/'encoder.json', self.root/'absent.json')
        self.assertEqual(result['rows'][0]['not_reviewed'], SCORED_ITEMS)
