"""Offline checks for closed review history and exact answer attribution."""
from collections import Counter
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hebrew_acronyms.saved_human_review import load_final_review, _merge_decisions, EXPORTS, FINAL
from hebrew_acronyms.test_review_export import identified_review_coverage

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT/'saved-results'
COHORT = ROOT/'data/splits/test_items.csv'


class FinalReviewTests(unittest.TestCase):
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

    def test_focused_denominator_requires_explicit_selection(self):
        with self.assertRaisesRegex(ValueError,'coverage differs'):
            identified_review_coverage(SAVED/EXPORTS[1],SAVED,COHORT)
        selection=json.loads((SAVED/FINAL/'selection-manifest.json').read_text())['reasons']
        result=identified_review_coverage(SAVED/EXPORTS[1],SAVED,COHORT,selected_group_ids=selection)
        self.assertEqual((result['decisions'],result['covered_answers']),(200,239))
        with self.assertRaisesRegex(ValueError,'Selected review group missing'):
            identified_review_coverage(SAVED/EXPORTS[1],SAVED,COHORT,selected_group_ids=[*selection,'unknown'])

    def test_undocumented_duplicate_or_wrong_binding_is_rejected(self):
        exports=[json.loads((SAVED/p).read_text()) for p in EXPORTS]
        audit=json.loads((SAVED/FINAL/'combined-review-coverage.json').read_text())
        sources={s['path'] for e in exports for s in e['source_files']}
        answers={a['id']:(i,a) for s in sources for i in json.loads((SAVED/s).read_text())['items'] for a in i['answers']}
        with self.assertRaisesRegex(ValueError,'Undocumented'):
            _merge_decisions(exports,answers,[])
        wrong=copy.deepcopy(exports);wrong[1]['decisions'][0]['decision_id']='wrong'
        with self.assertRaisesRegex(ValueError,'group differs'):
            _merge_decisions(wrong,answers,audit['rechecks'])

    def test_reuse_needs_approval_and_matching_context(self):
        from hebrew_acronyms import saved_human_review as module
        original=module._read
        for field in ('confirmed_by','new_bindings'):
            def changed(path):
                data=original(path)
                if str(path).endswith(EXPORTS[0]):
                    reuse=next(iter(data['historical_reuse'].values()))
                    if field=='confirmed_by': reuse[field]=''
                    else: reuse['proposal'][field][0]['response_sha256']='wrong'
                return data
            with patch.object(module,'_read',side_effect=changed), self.assertRaises(ValueError):
                load_final_review(SAVED,COHORT)


if __name__=='__main__':
    unittest.main()
