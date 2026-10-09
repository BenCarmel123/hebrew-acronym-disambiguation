"""Short qualitative route: mechanical fixtures, never human labels for research."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from hebrew_acronyms.human_review_short import ShortStore, build_plan, eligible, PROTOCOL


def fixture():
    items = []
    for n in range(40):
        group = n % 3
        scores = ([False, True, False, False], [False, False, False, False], [True, True, True, True])[group]
        answers = [{'id': f'i{n}:{system}', 'system_id': system, 'auto_score': score, 'raw': 'invented answer', 'decoded': 'invented', 'task': system.split('_')[1]}
                   for system, score in zip(('qwen_generate','qwen_select','gemini_generate','gemini_select'), scores)]
        items.append({'id': f'i{n}', 'acronym': str(n), 'source': str(n % 4), 'acronym_type_proxy': str(n % 2),
                      'sentence': 'invented sentence', 'gold': 'invented reference', 'answers': answers, 'candidates': ['invented'], 'source_metadata': {}})
    dataset = {'dataset_id': 'fixture', 'source_identity': 'fixture', 'provenance': {'repository': 'fixture', 'files': []},
               'queues': {'calibration': ['i0','i1']}, 'items': items}
    legacy = {'schema_version': 'human-review-v2','records': {'i0': {'exposure': {'gold': '2026-10-09T00:00:00+00:00'},
              'initial_interpretation': {'text': 'original'}, 'draft': {'annotator': 'QA_NOT_HUMAN', 'interpretation': 'original',
               'updated_interpretation': 'revised', 'updated_at': '2026-10-09T00:00:00+00:00',
               'answers': {'i0:qwen_generate': {'quality': 'wrong'}, 'i0:gemini_generate': {'quality': 'correct'}}}}}}
    dataset['short_plan'] = build_plan(dataset, legacy)
    return dataset, legacy


class ShortReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.d, self.legacy = fixture(); self.original = copy.deepcopy(self.legacy)
        self.path = Path(self.tmp.name)/'short.json'
        self.store = ShortStore(self.d, self.path, self.legacy, 'original history\n')

    def action(self, action, item='i0', **kwargs):
        return self.store.transact(action, {'revision':self.store.snapshot()['revision'], 'item_id':item, **kwargs})

    def test_selection_disjoint_exact_counts_same_model_and_old_queue_untouched(self):
        plan=self.d['short_plan']
        self.assertEqual(plan['actual'],dict(zip(('generation_fail_selection_pass','both_fail','both_generation_pass'),(12,4,4))))
        self.assertEqual(len(set(plan['queue'])),20)
        self.assertEqual(plan['queue'][0],'i0')
        self.assertEqual(self.d['queues']['calibration'],['i0','i1'])
        byid={i['id']:i for i in self.d['items']}
        for item_id,r in plan['selection_reasons'].items():
            self.assertIn(r['balance_witness'],eligible(byid[item_id])[r['group']])
        reordered=copy.deepcopy(self.d); reordered['items'].reverse()
        self.assertEqual(build_plan(reordered,self.legacy),plan)

    def test_retained_overlap_reserves_scarce_group(self):
        d=copy.deepcopy(self.d);d['items']=d['items'][:20]
        legacy={'records':{}}
        for n,i in enumerate(d['items']):
            scores=([False,True,False,False] if n<4 else [False,True,True,True] if n<16 else [True,True,True,True])
            for a,score in zip(i['answers'],scores): a['auto_score']=score
            if n<4:legacy['records'][i['id']]={}
        plan=build_plan(d,legacy)
        self.assertEqual(len(plan['queue']),20)
        self.assertEqual(plan['actual'],dict(zip(('generation_fail_selection_pass','both_fail','both_generation_pass'),(12,4,4))))

    def test_shortage_documented_without_duplicate_filler(self):
        d=copy.deepcopy(self.d);d['items']=d['items'][:3]
        plan=build_plan(d,self.legacy)
        self.assertEqual(len(plan['queue']),3);self.assertTrue(plan['adjustments'])

    def test_existing_work_inherited_exactly_without_requiring_repeat(self):
        self.assertEqual(self.store.snapshot()['counts']['complete_items'],1)
        self.assertEqual(self.store.snapshot()['counts']['judged_answers'],2)
        self.assertEqual(self.store.state['legacy_snapshot'],self.original)
        self.assertEqual(self.legacy,self.original)
        self.assertEqual(self.store.state['legacy_history'],'original history\n')
        self.assertNotIn('initial_interpretation', self.store.state['records']['i0'])
        self.assertFalse(self.store.state['records']['i0']['exposure'])

    def test_ambiguous_legacy_labels_never_silently_mapped(self):
        for quality in ('partial','legacy_partial','no_answer'):
            legacy=copy.deepcopy(self.legacy);legacy['records']['i0']['draft']['answers']['i0:qwen_generate']['quality']=quality
            s=ShortStore(self.d,Path(self.tmp.name)/(quality+'.json'),legacy)
            r=s.state['records']['i0'];self.assertEqual(len(r['judgments']),1)
            self.assertEqual(next(iter(r['unmapped_legacy'].values()))['quality'],quality)
            self.assertEqual(s.snapshot()['counts']['complete_items'],0)

    def test_judgment_payload_has_no_scores_models_sampling_or_legacy_notes(self):
        result=self.action('open')
        text=json.dumps(result)
        for leak in ('qwen','gemini','auto_score','selection_reason','disagrees_auto','original_quality'):
            self.assertNotIn(leak,text)
        self.assertTrue(result['item']['prior_reused'])
        self.assertFalse(self.store.state['records']['i0']['exposure']['reference_and_generation']['independent_attempt'])

    def test_partial_complete_edit_restart_and_export_preserve_original(self):
        new=next(i for i in self.d['short_plan']['queue'] if i!='i0')
        opened=self.action('open',new);ids=[a['id'] for a in opened['item']['answers']]
        self.action('save',new,judgments={ids[0]:'unsure'},note='QA only')
        self.assertEqual(self.store.snapshot()['records'][new]['completion']['status'],'partial')
        self.action('save',new,judgments={ids[0]:'unsure',ids[1]:'fits'},example=True,suspect=True,note='QA only')
        self.assertEqual(self.store.snapshot()['records'][new]['completion']['status'],'complete')
        self.action('save',new,judgments={ids[0]:'unsure'},note='QA only')
        restarted=ShortStore(self.d,self.path,self.legacy)
        self.assertEqual(restarted.state,self.store.state)
        export=restarted.export(); self.assertEqual(export['legacy_snapshot'],self.original)
        self.assertIn('save',export['short_history']);self.assertEqual(export['legacy_history'],'original history\n')
        self.assertEqual(export['summary']['counts']['partial_items'],1)
        self.assertEqual(export['summary']['unresolved'][0]['answers'][0]['comparison'],'undetermined')
        self.assertIn(new,restarted.markdown())
        self.assertNotIn('initial_interpretation',restarted.state['records'][new])

    def test_stale_unknown_answers_and_unopened_save_rejected(self):
        with self.assertRaises(ValueError): self.action('save',judgments={})
        self.action('open')
        with self.assertRaises(ValueError):self.store.transact('save',{'revision':0,'item_id':'i0'})
        with self.assertRaises(ValueError):self.action('save',judgments={'invented':'fits'})
        with self.assertRaises(ValueError):self.action('save',judgments={next(iter(self.store.answers['i0'])):'wrong'})

    def test_summary_and_details_explicitly_record_information_exposure(self):
        self.action('open');self.action('details')
        self.assertIn('case_details',self.store.state['records']['i0']['exposure'])
        result=self.action('summary');self.assertEqual(len(self.store.state['summary_exposures']),1)
        self.assertEqual(result['summary']['counts']['judged_answers'],2)
        self.assertEqual(len(result['summary']['disagreements']),1)
        self.assertIn('legacy_explicit_judgment',self.store.markdown())

    def test_changed_sources_or_plan_rejected_without_writes(self):
        before=self.path.read_bytes();d=copy.deepcopy(self.d);d['provenance']['repository']='changed'
        with self.assertRaises(ValueError): ShortStore(d,self.path,self.legacy)
        d=copy.deepcopy(self.d);d['short_plan']['plan_id']='different'
        with self.assertRaises(ValueError): ShortStore(d,self.path,self.legacy)
        self.assertEqual(before,self.path.read_bytes())

if __name__=='__main__':unittest.main()
