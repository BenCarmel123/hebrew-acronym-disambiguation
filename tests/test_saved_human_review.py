"""Offline checks for closed review history and exact answer attribution."""
from collections import Counter
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hebrew_acronyms.saved_human_review import (
    load_final_review, load_review_table, reconstruct_review_answers, expand_review_history,
    _merge_decisions, _validate_rounds, PROVENANCE, REVIEW,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT/'saved-results'
COHORT = ROOT/'data/splits/test_items.csv'


class FinalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.provenance = load_review_table(SAVED)
        cls.answers, cls.bundle_answers = reconstruct_review_answers(SAVED, COHORT, cls.provenance)
        cls.exports, cls.reuse = expand_review_history(cls.provenance)

    def validate(self, exports=None, answers=None, selection=None):
        _validate_rounds(exports or self.exports, answers or self.answers, self.bundle_answers,
                         selection or self.provenance['selection'])

    def setUp(self):
        network = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        network.start(); self.addCleanup(network.stop)

    def test_final_counts_deduplication_and_full_denominators(self):
        result = load_final_review(SAVED, COHORT)
        t = result['totals']
        self.assertEqual((t['total'], t['reviewed'], t['unique_groups_with_review']), (4953,741,638))
        self.assertEqual((t['new_human_reviewed'],t['historical_reused']), (707,34))
        self.assertEqual((t['new_decision_events'],t['new_unique_groups'],t['rechecked_responses']), (620,618,2))
        self.assertEqual((t['unreviewed'],t['valid_unreviewed'],t['technical_failures']), (4212,4195,17))
        self.assertEqual((t['fits'],t['not_fits'],t['unsure']), (396,345,0))
        self.assertEqual(len({r['answer_id'] for r in result['records']}),4953)
        self.assertTrue(all(r['total']==381 for r in result['rows']))
        self.assertTrue(all(not r['human_label'] for r in result['records'] if r['technical_failure']))
        self.assertEqual(sum(r['judgment_kind']=='unreviewed' for r in result['records']),4212)
        for r in result['rechecks']:
            self.assertEqual((r['previous']['label'],r['latest']['label']),('unsure','fits'))

    def test_scores_and_pending_cases_are_not_relabelled(self):
        result = load_final_review(SAVED,COHORT)
        scores=Counter(r['system'] for r in result['records'] if r['automatic_score'])
        self.assertEqual(dict(scores), dict(anthropic_generate=165,anthropic_select=339,
            dictabert_select=273,gemini_generate=207,gemini_select=357,openai_generate=129,
            openai_select=317,qwen14_generate=44,qwen14_select=266,qwen_generate=31,
            qwen_select=223,xai_generate=184,xai_select=351))
        pending=[r for r in result['cases'] if r['pending_clarification']]
        self.assertEqual(Counter(r['item_id'] for r in pending),
                         {'kn4-0525':6,'kn-0229':1,'manual-0023':1,'manual-0001':1})
        self.assertTrue(all(r['human_label']=='not_fits' and r['automatic_score'] for r in pending))
        contrast=[r for r in result['cases'] if not r['pending_clarification']]
        self.assertEqual([(r['item_id'],r['system'],r['human_label']) for r in contrast],
                         [('kn4-0525','gemini_generate','fits')])

    def test_every_review_response_matches_original_collection(self):
        from hebrew_acronyms.test_evaluation import summarize_evaluation
        result=load_final_review(SAVED,COHORT)
        by_run={}
        for r in result['records']:
            by_run.setdefault(r['run_id'],{})[(r['item_id'],r['task'])]=r
        checked=0
        for manifest_path in (SAVED/'colab-runs').glob('*/extracted/*/full-test/manifest.json'):
            manifest=json.loads(manifest_path.read_text());run_id=manifest['run_id']
            if run_id not in by_run: continue
            for r in summarize_evaluation(manifest_path.parent)['records']:
                task={'generate':'generation','select':'selection'}[r['task']]
                answer=by_run[run_id].get((r['item_id'],task))
                if answer is None: continue
                self.assertEqual((answer['raw'],answer['automatic_score'],answer['status']),
                                 (r['response'],r['correct'],r['status']))
                if task=='selection':self.assertEqual(answer['decoded'],r['selected_candidate'])
                checked+=1
        encoder=SAVED/'study-runs/dictabert-test-20261009-f49c3b5/predictions.jsonl'
        for line in encoder.read_text().splitlines():
            r=json.loads(line);a=by_run[r['run_id']].get((r['item_id'],'selection'))
            if a is None:continue
            self.assertEqual((a['raw'],a['automatic_score']),(r['selected_candidate'],r['correct']))
            checked+=1
        self.assertEqual(checked,4953)

    def test_focused_denominator_requires_exact_selection(self):
        self.validate()
        selection = copy.deepcopy(self.provenance['selection'])
        selection['reasons']['unknown'] = 'foreign group'
        with self.assertRaisesRegex(ValueError, 'Selected review group missing'):
            self.validate(selection=selection)
        selection = copy.deepcopy(self.provenance['selection'])
        selection['reasons'].pop(next(iter(selection['reasons'])))
        with self.assertRaisesRegex(ValueError, 'Foreign focused'):
            self.validate(selection=selection)

    def test_duplicate_decision_and_answer_are_rejected(self):
        wrong = copy.deepcopy(self.exports)
        wrong[0]['decisions'].append(wrong[0]['decisions'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.validate(exports=wrong)
        wrong = copy.deepcopy(self.exports)
        decision = wrong[0]['decisions'][0]
        decision['answer_ids'].append(decision['answer_ids'][0])
        decision['bindings'].append(decision['bindings'][0])
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            self.validate(exports=wrong)

    def test_foreign_run_binding_is_rejected(self):
        wrong = copy.deepcopy(self.exports)
        wrong[0]['decisions'][0]['bindings'][0]['run_id'] = 'foreign-run'
        with self.assertRaisesRegex(ValueError, 'response binding'):
            self.validate(exports=wrong)

    def test_technical_failure_cannot_receive_a_judgment(self):
        wrong = copy.deepcopy(self.answers)
        aid = self.exports[0]['decisions'][0]['answer_ids'][0]
        wrong[aid][1]['technical_failure'] = True
        with self.assertRaisesRegex(ValueError, 'Technical failure'):
            self.validate(answers=wrong)

    def test_undocumented_recheck_and_wrong_context_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Undocumented'):
            _merge_decisions(self.exports, self.answers, [])
        wrong = copy.deepcopy(self.exports)
        wrong[1]['decisions'][0]['decision_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'group differs'):
            _merge_decisions(wrong, self.answers, self.provenance['rechecks'])

    def test_recheck_must_follow_original_judgment(self):
        exports = copy.deepcopy(self.exports)
        rechecks = copy.deepcopy(self.provenance['rechecks'])
        recheck = rechecks[0]
        wrong_time = recheck['previous']['label_updated_at']
        recheck['latest']['label_updated_at'] = wrong_time
        event = next(d for d in exports[1]['decisions'] if recheck['answer_id'] in d['answer_ids'])
        event['judgment']['label_updated_at'] = wrong_time
        with self.assertRaisesRegex(ValueError, 'Recheck must be later'):
            _merge_decisions(exports, self.answers, rechecks)

    def test_reuse_needs_approval_and_matching_context(self):
        from hebrew_acronyms import saved_human_review as module
        for field in ('confirmed_by', 'new_bindings', 'context'):
            data = copy.deepcopy(self.provenance)
            reuse = next(iter(data['historical_reuse'].values()))
            if field == 'confirmed_by':
                reuse[field] = ''
            elif field == 'new_bindings':
                reuse['proposal'][field][0]['response_sha256'] = 'wrong'
            else:
                next(iter(data['reuse_contexts'].values()))['context']['sentence'] = 'changed'
            with patch.object(module, 'load_review_table', return_value=(self.table, data)), self.assertRaises(ValueError):
                load_final_review(SAVED, COHORT)

    def test_table_fields_must_match_full_event_attribution(self):
        from hebrew_acronyms import saved_human_review as module
        for field in ('label', 'run_id', 'response_sha256', 'annotator', 'label_updated_at', 'event_index'):
            table = copy.deepcopy(self.table)
            table[0][field] = 'changed'
            with patch.object(module, 'load_review_table', return_value=(table, self.provenance)), self.assertRaisesRegex(ValueError, 'attribution or response'):
                load_final_review(SAVED, COHORT)

    def test_table_hash_and_duplicate_judgment_are_rejected(self):
        import tempfile
        import hashlib
        import csv
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'human-review').mkdir()
            (root/'files.json').write_text(json.dumps({'files': {}}))
            (root/PROVENANCE).write_text(json.dumps(self.provenance))
            (root/REVIEW).write_bytes((SAVED/REVIEW).read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'table hash mismatch'):
                load_review_table(root)
            rows = copy.deepcopy(self.table); rows[1] = rows[0]
            with (root/REVIEW).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            provenance = copy.deepcopy(self.provenance)
            provenance['review_sha256'] = hashlib.sha256((root/REVIEW).read_bytes()).hexdigest()
            (root/PROVENANCE).write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'Duplicate judgment'):
                load_review_table(root)

    def test_history_preserves_attribution_and_explicit_confirmation(self):
        result = load_final_review(SAVED, COHORT)
        self.assertEqual(len(result['decisions']), 620)
        self.assertEqual(len({e['judgment']['label_updated_at'] for e in result['decisions']}), 620)
        self.assertEqual(len({e['judgment']['batch_id'] for e in result['decisions']}), 32)
        for event in result['decisions']:
            self.assertEqual(event['judgment']['annotator'], 'שקד')
            self.assertEqual(event['judgment']['interaction_mode'], 'explicit_table_confirmation')
            self.assertEqual(event['judgment']['exposure_evidence']['status'], 'unknown')
        self.assertEqual(result['totals']['automatic_negative_human_fits'], 338)


if __name__ == '__main__':
    unittest.main()
