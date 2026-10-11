"""Regression checks for the preserved DictaLM evidence and new review bindings."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.saved_human_review import load_review_table, REVIEW, PROVENANCE
from hebrew_acronyms.test_evaluation import summarize_evaluation

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
        table, provenance = load_review_table(SAVED)
        cls.tags = [r for r in table if r['layer'] == 'ben_dictalm']
        cls.provenance = provenance['ben']

    def test_denominators_and_separate_diagnostic(self):
        a = self.audit
        self.assertEqual([(m['n_items'], m['n_correct'], m['n_valid']) for m in a['metrics']],
                         [(381, 125, 152), (381, 0, 0)])
        self.assertEqual(len(a['records']), 762)
        self.assertEqual(len({(r['item_id'], r['task']) for r in a['records']}), 762)
        self.assertEqual(a['diagnostic_selection']['leading_label_matches_gold'], 243)
        self.assertEqual(a['diagnostic_selection']['csv_boundary_whitespace_trimmed'], 381)
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
                    _bind_ben_tags(tags, self.generation_records, self.provenance)

    def test_changed_annotation_gold_response_and_labels_are_rejected(self):
        for field, value in (('displayed_response', 'changed response'), ('label', 'unknown')):
            with self.subTest(field=field):
                tags = copy.deepcopy(self.tags)
                tags[0][field] = value
                with self.assertRaisesRegex(ValueError, 'response differs|yes or no'):
                    _bind_ben_tags(tags, self.generation_records, self.provenance)
        tags = copy.deepcopy(self.tags)
        tags[0]['label'] = 'no'
        with self.assertRaisesRegex(ValueError, 'label counts differ'):
            _bind_ben_tags(tags, self.generation_records, self.provenance)

    def test_historical_id_and_attribution_cannot_change(self):
        for field in ('judgment_id', 'annotator', 'run_id', 'system', 'task', 'label_updated_at'):
            tags = copy.deepcopy(self.tags); tags[0][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity or attribution'):
                _bind_ben_tags(tags, self.generation_records, self.provenance)
        provenance = copy.deepcopy(self.provenance); provenance['source_sha256'] = 'new-table-hash'
        with self.assertRaisesRegex(ValueError, 'historical provenance'):
            _bind_ben_tags(self.tags, self.generation_records, provenance)

    def test_judgments_cannot_transfer_to_other_runs_models_or_tasks(self):
        for field, value in (('run_id', 'other-run'), ('system', 'other-model'), ('task', 'select')):
            with self.subTest(field=field):
                records = copy.deepcopy(self.generation_records)
                records[0][field] = value
                with self.assertRaisesRegex(ValueError, 'original DictaLM generation run'):
                    load_ben_dictalm_tags(SAVED, records)
        records = copy.deepcopy(self.generation_records)
        records[0]['response_sha256'] = 'altered-hash'
        with self.assertRaisesRegex(ValueError, 'response differs|response hash mismatch|generation run journal'):
            load_ben_dictalm_tags(SAVED, records)

    def test_altered_consolidated_table_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'human-review').mkdir()
            import shutil
            (root / 'files.json').write_bytes((SAVED / 'files.json').read_bytes())
            shutil.copytree(SAVED / SOURCE / 'dictalm/full-test', root / SOURCE / 'dictalm/full-test',
                            ignore=shutil.ignore_patterns('summary.json'))
            (root / PROVENANCE).write_bytes((SAVED / PROVENANCE).read_bytes())
            (root / REVIEW).write_bytes((SAVED / REVIEW).read_bytes() + b'\n')
            inventory = json.loads((root / 'files.json').read_text())
            # The completed inventory rejects the modified table before its own
            # provenance link. Exercise both independent integrity boundaries.
            if REVIEW in inventory['files']:
                with self.assertRaisesRegex(ValueError, 'Saved review evidence differs'):
                    load_ben_dictalm_tags(root, self.generation_records)
            for relative in (REVIEW, PROVENANCE):
                inventory['files'].pop(relative, None)
            (root / 'files.json').write_text(json.dumps(inventory))
            with self.assertRaisesRegex(ValueError, 'table hash mismatch'):
                load_ben_dictalm_tags(root, self.generation_records)

    def test_bindings_preserve_journal_whitespace(self):
        summary = summarize_evaluation(SAVED / 'local-runs/dictalm-20261010/dictalm/full-test')
        original = {(r['item_id'], r['task']): r for r in summary['records']}
        for record in self.audit['records']:
            original_record = original[record['item_id'], record['task']]
            self.assertEqual({key: record[key] for key in original_record}, original_record)
            self.assertEqual(record['response_sha256'], hashlib.sha256(record['response'].encode()).hexdigest())
            self.assertIsNone(record['human_judgment'])
        self.assertEqual(sum(r['response'].startswith('\n') for r in self.audit['records'] if r['task'] == 'select'), 381)

    def test_supplied_gold_valid_and_per_record_scores_match_original_journal(self):
        for field, value in (('gold', 'corrupted-gold'), ('valid', not self.generation_records[0]['valid'])):
            records = copy.deepcopy(self.generation_records); records[0][field] = value
            with self.assertRaisesRegex(ValueError, 'generation run journal'):
                load_ben_dictalm_tags(SAVED, records)
        records = copy.deepcopy(self.generation_records)
        # Keep aggregate automatic/human cross-tab unchanged while altering two items.
        same_label = [j for j in self.audit['ben_annotation']['judgments'] if j['label'] == 'yes']
        positive = next(j['item_id'] for j in same_label if j['automatic_correct'])
        negative = next(j['item_id'] for j in same_label if not j['automatic_correct'])
        for record in records:
            if record['item_id'] in (positive, negative):
                record['correct'] = not record['correct']
        with self.assertRaisesRegex(ValueError, 'generation run journal'):
            load_ben_dictalm_tags(SAVED, records)

    def test_audit_never_reads_retired_summaries_or_review_csvs(self):
        retired = {'summary.json', 'session-summary.json', 'dictalm_generate_human_tags.csv',
                   'dictalm_test_generate_details.csv', 'dictalm_test_select_details.csv'}
        original = Path.open
        def guarded(path, *args, **kwargs):
            if SOURCE in str(path) and path.name in retired:
                raise AssertionError('Retired source read: ' + str(path))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            audit = audit_saved_dictalm(SAVED, REFERENCE)
        self.assertFalse(any(Path(s['path']).name in retired for s in audit['sources']))
        self.assertTrue({REVIEW, PROVENANCE} <= {s['path'] for s in audit['sources']})

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
