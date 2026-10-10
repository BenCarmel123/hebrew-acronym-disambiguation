"""Regression checks for the preserved DictaLM evidence and new review bindings."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.dictalm_saved_audit import (
    BEN_TAGS_COMMIT, BEN_TAGS_FILE, BEN_TAGS_SHA256, RUN_ID, SOURCE,
    _bind_ben_tags, audit_saved_dictalm, load_ben_dictalm_tags,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'saved-results'
REFERENCE = SAVED / 'colab-runs/evaluation-20261009-a319f9d/extracted/qwen/full-test/manifest.json'


class DictalmAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_saved_dictalm(SAVED, REFERENCE)
        cls.generation_records = [r for r in cls.audit['records'] if r['task'] == 'generate']
        with (SAVED / SOURCE / BEN_TAGS_FILE).open(encoding='utf-8-sig', newline='') as handle:
            cls.tags = list(csv.DictReader(handle))
        with (SAVED / SOURCE / 'dictalm_test_generate_details.csv').open(encoding='utf-8-sig', newline='') as handle:
            cls.details = list(csv.DictReader(handle))

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

    def test_ben_annotations_are_separate_and_repeatable(self):
        before = copy.deepcopy(self.generation_records)
        first = load_ben_dictalm_tags(SAVED, self.generation_records)
        second = load_ben_dictalm_tags(SAVED, self.generation_records)
        self.assertEqual(first, second)
        self.assertEqual(first, self.audit['ben_annotation'])
        self.assertEqual(before, self.generation_records)
        self.assertEqual(first['summary'], {'n_items': 381, 'n_yes': 193, 'n_no': 188,
                                           'accuracy': 193 / 381})
        self.assertEqual(first['cross_tab'], {'both_accept': 121, 'ben_only': 72,
                                             'automatic_only': 4, 'both_reject': 184})
        self.assertEqual(first['validation']['journal_exact_matches'], 234)
        self.assertEqual(first['validation']['journal_boundary_whitespace_only'], 147)
        self.assertEqual(first['validation']['source_csv_exact_matches'], 381)
        self.assertEqual(len({j['judgment_id'] for j in first['judgments']}), 381)
        for judgment in first['judgments']:
            self.assertEqual((judgment['run_id'], judgment['system'], judgment['task']),
                             (RUN_ID, 'dictalm', 'generate'))
            self.assertEqual(judgment['annotator'], 'Ben')
            self.assertEqual(judgment['source_commit'], BEN_TAGS_COMMIT)
            self.assertEqual(judgment['source_sha256'], BEN_TAGS_SHA256)
            self.assertNotIn('annotation_time', judgment)
            self.assertEqual(judgment['response_sha256'],
                             hashlib.sha256(judgment['response'].encode()).hexdigest())

    def test_four_reported_metrics_remain_distinct(self):
        metrics = self.audit['reported_metrics']
        self.assertEqual([(m['metric'], m['n_correct'], m['n_items']) for m in metrics], [
            ('automatic_generation', 125, 381), ('ben_generation', 193, 381),
            ('strict_selection', 0, 381), ('diagnostic_leading_label_selection', 243, 381)])
        self.assertEqual([round(m['accuracy'] * 100, 1) for m in metrics], [32.8, 50.7, 0, 63.8])
        self.assertEqual([m['judgment_layer'] for m in metrics],
                         ['automatic', 'human_ben', 'automatic', 'diagnostic'])

    def test_annotation_duplicate_and_out_of_cohort_items_are_rejected(self):
        for item_id in (self.tags[1]['item_id'], 'kn-0116', 'not-a-test-item'):
            with self.subTest(item_id=item_id):
                tags = copy.deepcopy(self.tags)
                tags[0]['item_id'] = item_id
                with self.assertRaisesRegex(ValueError, 'unique scored|cohort differs'):
                    _bind_ben_tags(tags, self.details, self.generation_records)

    def test_changed_annotation_gold_response_and_labels_are_rejected(self):
        for field, value in (('gold', 'changed gold'), ('response', 'changed response'),
                             ('same_as_gold', 'unknown')):
            with self.subTest(field=field):
                tags = copy.deepcopy(self.tags)
                tags[0][field] = value
                with self.assertRaisesRegex(ValueError, 'gold/response|yes or no'):
                    _bind_ben_tags(tags, self.details, self.generation_records)
        tags = copy.deepcopy(self.tags)
        tags[0]['same_as_gold'] = 'no'
        with self.assertRaisesRegex(ValueError, 'label counts differ'):
            _bind_ben_tags(tags, self.details, self.generation_records)

    def test_modified_original_csv_is_rejected(self):
        details = copy.deepcopy(self.details)
        details[0]['response'] = 'changed source response'
        with self.assertRaisesRegex(ValueError, 'gold/response'):
            _bind_ben_tags(self.tags, details, self.generation_records)

    def test_judgments_cannot_transfer_to_other_runs_models_or_tasks(self):
        for field, value in (('run_id', 'other-run'), ('system', 'other-model'), ('task', 'select')):
            with self.subTest(field=field):
                records = copy.deepcopy(self.generation_records)
                records[0][field] = value
                with self.assertRaisesRegex(ValueError, 'original DictaLM generation run'):
                    load_ben_dictalm_tags(SAVED, records)
        records = copy.deepcopy(self.generation_records)
        records[0]['response_sha256'] = 'altered-hash'
        with self.assertRaisesRegex(ValueError, 'response hash mismatch'):
            load_ben_dictalm_tags(SAVED, records)

    def test_altered_tag_file_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / SOURCE
            root.mkdir(parents=True)
            (root / BEN_TAGS_FILE).write_bytes((SAVED / SOURCE / BEN_TAGS_FILE).read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'annotation source hash mismatch'):
                load_ben_dictalm_tags(folder, self.generation_records)

    def test_bindings_preserve_journal_whitespace(self):
        summary = json.loads((SAVED / 'local-runs/dictalm-20261010/dictalm/full-test/summary.json').read_text())
        original = {(r['item_id'], r['task']): r for r in summary['records']}
        for record in self.audit['records']:
            original_record = original[record['item_id'], record['task']]
            self.assertEqual({key: record[key] for key in original_record}, original_record)
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
