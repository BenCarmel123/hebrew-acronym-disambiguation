"""Independent review checks using invented records, never human annotations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.human_review_data import make_answer, mechanical_spans, SYSTEMS
from hebrew_acronyms.human_review_server import ReviewStore, SCHEMA


class IndependentSourceChecks(unittest.TestCase):
    def item(self):
        return {'item_id': 'invented-1', 'acronym': 'א״ב', 'gold_expansion': 'אור בהיר',
                'candidates': 'אור בהיר | אבק בית'}

    def row(self):
        return {'item_id': 'invented-1', 'acronym': 'א״ב', 'gold': 'אור בהיר',
                'candidates': 'אור בהיר | אבק בית', 'shown_order': 'אבק בית | אור בהיר',
                'response': 'B', 'correct': 'True', 'valid': 'True'}

    def test_saved_order_controls_letter(self):
        answer = make_answer(self.item(), SYSTEMS[3], self.row())
        self.assertEqual(answer['decoded'], 'אור בהיר')
        self.assertEqual(answer['option_mapping'][0]['candidate'], 'אבק בית')
        self.assertEqual(answer['system_id'], 'qwen_select')
        self.assertEqual(answer['id'], 'invented-1:qwen_select')

    def test_source_metadata_mismatch_rejected(self):
        row = self.row(); row['gold'] = 'אבק בית'
        with self.assertRaises(ValueError):
            make_answer(self.item(), SYSTEMS[3], row)

    def test_quote_fold_finds_all_occurrences(self):
        sentence = 'א״ב וגם א"ב'
        spans = mechanical_spans(sentence, 'א״ב')
        self.assertEqual([(v['start'], v['end']) for v in spans], [(0, 3), (8, 11)])

    def test_absent_prediction_remains_absent(self):
        answer = make_answer(self.item(), SYSTEMS[3], None)
        self.assertIsNone(answer['raw'])
        self.assertIsNone(answer['auto_score'])
        self.assertTrue(answer['missing'])

    def test_thirty_candidates_do_not_invent_letter_labels(self):
        candidates = ['invented ' + str(i) for i in range(30)]
        item = self.item(); item.update(candidates=' | '.join(candidates), gold_expansion=candidates[22])
        row = self.row(); row.update(candidates=item['candidates'], shown_order=item['candidates'],
                                     gold=item['gold_expansion'], response='W', correct='False', valid='False')
        answer = make_answer(item, SYSTEMS[3], row)
        self.assertEqual([v['letter'] for v in answer['option_mapping']], list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertFalse(answer['auto_score'])


class IndependentPersistenceChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'annotations.json'
        self.data = {'dataset_id': 'invented-data', 'provenance': {'files': []},
                     'items': [{'id': 'invented-1', 'answers': [{'id': 'a1', 'system_id': 'invented-system'}]}]}
        self.store = ReviewStore(self.data, self.path)
        self.annotation = {'schema_version': SCHEMA, 'annotator': 'Invented Fixture Reviewer',
                           'interpretation_kind': 'none', 'interpretation': '',
                           'item_problems': ['valid'], 'answers': {'a1': {'system_id': 'invented-system',
                            'quality': 'correct', 'format_ok': 'no', 'disagrees_auto': 'yes'}}}

    def update(self, action, **extra):
        return self.store.update({'revision': self.store.snapshot()['revision'], 'item_id': 'invented-1',
                                  'action': action, **extra})

    def test_wrong_system_never_saved(self):
        annotation = copy.deepcopy(self.annotation)
        annotation['answers']['a1']['system_id'] = 'other'
        with self.assertRaises(ValueError):
            self.update('review', annotation=annotation)
        self.assertFalse(self.path.exists())

    def test_review_json_csv_reload_preserve_identity_and_flags(self):
        self.update('review', annotation=self.annotation)
        saved = self.store.snapshot()
        csv_copy = self.store.csv_bundle(self.store.export_csv())
        self.assertEqual(csv_copy['records'], saved['records'])
        restarted = ReviewStore(self.data, self.path)
        self.assertEqual(restarted.snapshot(), saved)
        reviewed = saved['records']['invented-1']['reviewed']
        self.assertEqual(reviewed['annotator'], 'Invented Fixture Reviewer')
        self.assertEqual(reviewed['answers']['a1']['format_ok'], 'no')
        self.assertIn('reviewed_at', saved['records']['invented-1'])

    def test_draft_edit_preserves_reviewed_snapshot(self):
        self.update('review', annotation=self.annotation)
        changed = copy.deepcopy(self.annotation); changed['answers']['a1']['quality'] = 'wrong'
        self.update('draft', annotation=changed)
        record = self.store.snapshot()['records']['invented-1']
        self.assertEqual(record['reviewed']['answers']['a1']['quality'], 'correct')
        self.assertEqual(record['draft']['answers']['a1']['quality'], 'wrong')

    def test_stale_revision_does_not_clobber(self):
        self.update('draft', annotation=self.annotation)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.update({'revision': 0, 'item_id': 'invented-1', 'action': 'review', 'annotation': self.annotation})
        self.assertEqual(self.path.read_bytes(), before)

    def test_exposure_persists_across_drafts_restart(self):
        self.update('draft', annotation=self.annotation)
        for stage in ['candidates', 'gold', 'responses']:
            self.update('expose', stage=stage)
        self.update('draft', annotation=self.annotation)
        self.assertEqual(set(ReviewStore(self.data, self.path).snapshot()['records']['invented-1']['exposure']),
                         {'candidates', 'gold', 'responses'})

    def test_identity_requires_saved_current_review(self):
        self.update('draft', annotation=self.annotation)
        for stage in ['candidates', 'gold', 'responses']:
            self.update('expose', stage=stage)
        with self.assertRaises(ValueError):
            self.update('expose', stage='identities')
        self.update('review', annotation=self.annotation)
        changed = copy.deepcopy(self.annotation); changed['answers']['a1']['quality'] = 'wrong'
        self.update('draft', annotation=changed)
        with self.assertRaises(ValueError):
            self.update('expose', stage='identities')

    def test_import_wrong_dataset_is_transactional(self):
        self.update('review', annotation=self.annotation)
        bad = self.store.snapshot(); bad['dataset_id'] = 'another-dataset'
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.import_bundle(bad, self.store.snapshot()['revision'])
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
