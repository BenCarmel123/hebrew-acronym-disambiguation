"""Mechanical review-export tests with invented records; no semantic judgments."""
import unittest
from hebrew_acronyms.human_review_data import (
    SYSTEMS, csv_bool, decode_letter, index_rows, make_answer, mechanical_spans, select_sample, source_identity, sampling_identity,
)


class HumanReviewDataTests(unittest.TestCase):
    def setUp(self):
        self.item = {'item_id': 'invented-1', 'acronym': 'א״ב', 'gold_expansion': 'אלף בית',
                     'candidates': 'אלף בית | אב גד', 'sentence': 'א"ב ועוד א״ב'}
        self.row = {'item_id': 'invented-1', 'acronym': 'א״ב', 'gold': 'אלף בית',
                    'candidates': 'אלף בית | אב גד', 'shown_order': 'אב גד | אלף בית',
                    'response': 'B', 'correct': 'True', 'valid': 'True'}

    def test_saved_shuffled_order_controls_decode(self):
        answer = make_answer(self.item, SYSTEMS[3], self.row)
        self.assertEqual(answer['decoded'], 'אלף בית')
        self.assertEqual(answer['option_mapping'][0], {'letter': 'A', 'candidate': 'אב גד'})
        self.assertEqual(answer['mechanical_flags'], [])
        self.assertTrue(answer['auto_score'])

    def test_rejects_mismatched_metadata_and_candidate_mapping(self):
        for change in [{'acronym': 'א״ג'}, {'gold': 'אב גד'}, {'candidates': 'אב גד | אלף בית'},
                       {'shown_order': 'אב גד | אב גד'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_answer(self.item, SYSTEMS[3], dict(self.row, **change))

    def test_strict_parse_and_missing_record(self):
        for invalid in ['', 'a', 'A.', 'A B', 'C']:
            self.assertIsNone(decode_letter(invalid, ['x', 'y']))
        self.assertEqual(decode_letter(' B\n', ['x', 'y']), 'y')
        missing = make_answer(self.item, SYSTEMS[3], None)
        self.assertTrue(missing['missing'])
        self.assertIsNone(missing['raw'])
        self.assertIsNone(missing['auto_score'])

    def test_quotes_and_multiple_matches_keep_original_offsets(self):
        spans = mechanical_spans(self.item['sentence'], self.item['acronym'])
        self.assertEqual(spans, [{'start': 0, 'end': 3}, {'start': 9, 'end': 12}])
        self.assertEqual(mechanical_spans('אין כאן', 'א״ב'), [])

    def test_over_26_decode_preserves_original_parser_failure(self):
        candidates = [f'candidate-{i}' for i in range(30)]
        item = dict(self.item, gold_expansion=candidates[22], candidates=' | '.join(candidates))
        row = dict(self.row, gold=candidates[22], candidates=item['candidates'], shown_order=item['candidates'], response='W', correct='False', valid='False')
        answer = make_answer(item, SYSTEMS[3], row)
        self.assertEqual(answer['decoded'], candidates[22])
        self.assertFalse(answer['auto_score'])
        self.assertEqual(len(answer['option_mapping']), 26)
        self.assertEqual(answer['option_mapping'][-1]['letter'], 'Z')
        self.assertEqual(len(answer['omitted_candidates']), 4)
        self.assertIn('over_26_candidates', answer['mechanical_flags'])
        self.assertNotIn('stored_score_recomputation_mismatch', answer['mechanical_flags'])

    def test_original_score_preserved_and_discrepancy_flagged(self):
        answer = make_answer(self.item, SYSTEMS[3], dict(self.row, correct='False'))
        self.assertFalse(answer['auto_score'])
        self.assertTrue(answer['mechanical_recomputed_score'])
        self.assertIn('stored_score_recomputation_mismatch', answer['mechanical_flags'])

    def test_substring_not_exact_is_mechanical(self):
        answer = make_answer(self.item, SYSTEMS[2], dict(self.row, shown_order='', response='פירוש: אלף בית'))
        self.assertIn('substring_not_exact', answer['mechanical_flags'])
        self.assertNotIn('semantic_label', answer)

    def test_duplicate_and_empty_ids_rejected(self):
        for rows in [[self.item, self.item], [{'item_id': ''}]]:
            with self.assertRaises(ValueError):
                index_rows(rows, 'fixture')
        with self.assertRaises(ValueError):
            csv_bool('', 'fixture')

    def test_deterministic_outcome_independent_sample(self):
        items = [{'id': f'i{i}', 'acronym': f't{i}', 'sampling_stratum': str(i % 3),
                  'answers': [{'auto_score': False}]} for i in range(35)]
        sample = select_sample(items, 20, 'fixed')
        self.assertEqual(sample, select_sample(list(reversed(items)), 20, 'fixed'))
        for item in items:
            item['answers'][0]['auto_score'] = True
        self.assertEqual(sample, select_sample(items, 20, 'fixed'))
        self.assertEqual(len(set(sample)), 20)
        self.assertEqual(len(select_sample(items, 40, 'fixed')), 35)

    def test_source_identity_separate_from_queue_and_build_metadata(self):
        provenance = {'repository': 'fixture', 'files': [
            {'path': 'answers.csv', 'sha256': 'a' * 64, 'bytes': 10},
            {'path': 'items.csv', 'sha256': 'b' * 64, 'bytes': 20}],
            'source_root': '/one', 'commit': 'old'}
        same = dict(provenance, source_root='/two', commit='new', files=list(reversed(provenance['files'])))
        self.assertEqual(source_identity(provenance), source_identity(same))
        changed = dict(provenance, files=[dict(provenance['files'][0], sha256='c' * 64), provenance['files'][1]])
        self.assertNotEqual(source_identity(provenance), source_identity(changed))
        self.assertNotEqual(sampling_identity({'calibration': ['i1']}, 'seed', 1),
                            sampling_identity({'calibration': ['i1', 'i2']}, 'seed', 2))


if __name__ == '__main__':
    unittest.main()
