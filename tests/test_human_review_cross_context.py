"""Explicit common human decisions over exact-key contexts; entirely invented QA data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.human_review_masked import ContinuationStore, MaskedStore, FILTER_VERSION
from hebrew_acronyms.human_review_short import ShortStore


def fixture():
    # Two same-item occurrences in c1, one in c2/c3. Near matches must stay separate.
    specs = [('old','A','gold','shared','old-other'), ('c1','A','gold','shared','shared'),
             ('c2','A','gold','shared','different'), ('c3','A','gold','shared','gold'),
             ('other_acronym','B','gold','shared','other'), ('other_gold','A','different-gold','shared','other'),
             ('raw_space','A','gold','shared ','other')]
    items=[]
    for iid, acronym,gold,a,b in specs:
        items.append({'id':iid,'sentence':'distinct sentence '+iid,'acronym':acronym,'gold':gold,'source':'fixture',
                      'candidates':[gold], 'answers':[{'id':iid+':'+system,'system_id':system,'raw':raw,'auto_score':False}
                       for system,raw in zip(('qwen_generate','gemini_generate'),(a,b))]})
    plan={'plan_id':'old','queue':['old'],'selection_reasons':{'old':{'group':'both_fail'}},'actual':{'both_fail':1},'source_counts':{'fixture':1},'adjustments':[]}
    data={'dataset_id':'fixture','source_identity':'fixture','provenance':{'repository':'fixture','files':[]},'items':items,'short_plan':plan,
          'continuation_plan':{'plan_id':'next','queue':[i['id'] for i in items],'rules_version':FILTER_VERSION}}
    legacy={'schema_version':'human-review-v2','records':{'old':{'exposure':{},'draft':{'annotator':'QA','updated_at':'2026-10-09T08:00:00+00:00',
             'answers':{'old:qwen_generate':{'quality':'correct'},'old:gemini_generate':{'quality':'wrong'}}}}}}
    return data,legacy


class CrossContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        self.data,legacy=fixture();old=ShortStore(self.data,root/'old.json',legacy)
        masked=MaskedStore(self.data,root/'v2.json',old.state)
        self.previous=copy.deepcopy(masked.state);self.path=root/'v3.json'
        self.store=ContinuationStore(self.data,self.path,self.previous)
        self.gid=next(gid for gid,g in self.store.group_index().items() if set(g['members'])=={'c1','c2','c3'})

    def act(self,action,**kw):
        return self.store.transact(action,{'revision':self.store.state['revision'],'view':'all',**kw})

    def open(self):
        return self.act('group-open',group_id=self.gid)['group']

    def save(self,group,**kw):
        return self.act('group-save',group_id=group['id'],open_id=group['open_id'],**kw)

    def judgments(self,iid):
        return self.store.state['records'][iid]['judgments']

    def test_exact_key_only_and_prior_human_never_enrolled(self):
        before=copy.deepcopy(self.store.state['records']['old'])
        group=self.open()
        self.assertEqual({'c1','c2','c3'},{c['id'] for c in group['contexts']})
        self.assertEqual(4,sum(c['occurrence_count'] for c in group['contexts']))
        self.assertEqual('shared',group['text']);self.assertEqual('A',group['acronym']);self.assertEqual('gold',group['gold'])
        self.assertTrue(all(c['sentence']=='distinct sentence '+c['id'] for c in group['contexts']))
        self.save(group,common_label='fits')
        self.assertEqual(before,self.store.state['records']['old'])
        self.assertNotIn('other_acronym',self.store.state['records'])
        self.assertNotIn('other_gold',self.store.state['records'])
        self.assertNotIn('raw_space',self.store.state['records'])

    def test_common_exception_defer_and_single_action_accounting(self):
        group=self.open();before=self.store.counts()['human_decisions']
        self.save(group,common_label='fits',common_tags=['spelling'],exceptions={'c2':{'label':'not_fits'},'c3':{'label':'defer'}})
        self.assertEqual({'fits'},{j['label'] for j in self.judgments('c1').values()})
        self.assertEqual('not_fits',self.judgments('c2')['c2:qwen_generate']['label'])
        self.assertEqual('',self.judgments('c3')['c3:qwen_generate']['label'])
        self.assertEqual(before+2,self.store.counts()['human_decisions'])
        self.assertEqual(1,self.store.counts()['shared_common_actions'])
        self.assertEqual(2,self.store.counts()['group_context_judgments'])
        self.assertEqual(1,self.store.counts()['group_exception_decisions'])
        summary=self.store.summary(True)
        self.assertEqual('exact-cross-context-v1',summary['grouping_version'])
        self.assertIn('cross_context',summary['cases'][-1]['answers'][0])
        self.assertIn('הכרעות משותפות שנשמרו',self.store.markdown(True))
        same=list(self.judgments('c1').values())
        self.assertEqual(same[0]['decision_id'],same[1]['decision_id'])
        self.assertEqual(2,same[0]['occurrence_count'])
        links=[j['cross_context'] for iid in ('c1','c2','c3') for j in self.judgments(iid).values()]
        self.assertEqual(1,len({link['shared_action_id'] for link in links}))
        self.assertEqual({'common','exception','defer'},{link['kind'] for link in links})
        self.assertIn('c3',self.store.snapshot()['queues']['all'])

    def test_group_autosave_edit_after_completion_and_explicit_defer(self):
        group=self.open();self.save(group,common_label='fits')
        decisions=self.store.counts()['human_decisions']
        self.assertNotIn(self.gid,self.store.snapshot()['group_queue'])
        self.save(group,common_label='fits',common_tags=['equivalent'])
        self.assertEqual(decisions,self.store.counts()['human_decisions'])
        self.assertTrue(all(j['tags']==['equivalent'] for iid in ('c1','c2','c3') for j in self.judgments(iid).values()))
        self.save(group,common_label='fits',common_tags=['equivalent'],exceptions={'c3':{'label':'defer'}})
        self.assertEqual('',self.judgments('c3')['c3:qwen_generate']['label'])
        reopened=self.open()
        self.assertEqual(['c3'],[c['id'] for c in reopened['contexts']]);self.assertEqual('',reopened['form']['label'])
        with self.assertRaises(ValueError):self.save(group,common_label='wrong')
        self.save(reopened,common_tags=['spelling'])
        self.assertEqual('',self.judgments('c3')['c3:qwen_generate']['label'])

    def test_unknown_context_invalid_label_and_members_are_atomic(self):
        group=self.open()
        for extra in ({'exceptions':{'old':{'label':'fits'}}}, {'exceptions':{'missing':{'label':'fits'}}},
                      {'common_label':'correct'}, {'members':{'old':['fake']}},
                      {'context_updates':{'c2':{'suspect':'not-a-bool'}}}):
            before=self.path.read_bytes(); revision=self.store.state['revision']
            with self.assertRaises(ValueError):self.save(group,**extra)
            self.assertEqual(before,self.path.read_bytes());self.assertEqual(revision,self.store.state['revision'])

    def test_stale_revision_external_edit_and_context_change_rejected(self):
        group=self.open();revision=self.store.state['revision'];self.save(group,common_label='unsure')
        with self.assertRaises(ValueError):self.store.transact('group-save',{'revision':revision,'group_id':self.gid,'open_id':group['open_id'],'common_label':'fits'})
        token=next(c['answer_ids'][0] for c in group['contexts'] if c['id']=='c2')
        self.act('save',item_id='c2',judgments={token:'not_fits'})
        before=self.path.read_bytes()
        with self.assertRaises(ValueError):self.save(group,common_label='fits')
        self.assertEqual(before,self.path.read_bytes())
        self.assertEqual('individual_edit',self.judgments('c2')['c2:qwen_generate']['cross_context']['kind'])

    def test_outside_note_flag_edit_cannot_be_clobbered(self):
        group=self.open()
        self.act('save',item_id='c2',note='new independent note',suspect=True)
        before=self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.save(group,common_label='fits',context_updates={'c2':{'note':'','suspect':False}})
        self.assertEqual(before,self.path.read_bytes())
        self.assertEqual('new independent note',self.store.state['records']['c2']['note'])

    def test_source_key_change_blocks_atomic_group_save(self):
        group=self.open(); before=self.path.read_bytes()
        self.store.items['c3']['gold']='different'
        with self.assertRaises(ValueError):self.save(group,common_label='fits')
        self.assertEqual(before,self.path.read_bytes())

    def test_details_masking_history_and_restore_group_metadata(self):
        group=self.open();result=self.act('group-details',group_id=self.gid,open_id=group['open_id'])
        self.assertEqual(3,len(result['details']['contexts']))
        self.save(group,common_label='fits',exceptions={'c2':{'label':'unsure'}})
        saved=copy.deepcopy(self.judgments('c1'))
        backup=self.store.export()
        encoded=json.dumps(backup)
        self.assertNotIn('qwen',encoded);self.assertNotIn('gemini',encoded);self.assertNotIn('auto_score',encoded)
        self.save(group,common_label='not_fits')
        self.store.restore_masked(backup,self.store.state['revision'])
        for aid,j in saved.items():
            self.assertEqual(j['cross_context'],self.judgments('c1')[aid]['cross_context'])
            self.assertEqual(j['decision_id'],self.judgments('c1')[aid]['decision_id'])
        restarted=ContinuationStore(self.data,self.path,self.previous)
        self.assertEqual(self.store.state,restarted.state)
        events=[json.loads(line) for line in self.path.with_suffix('.history.jsonl').read_text().splitlines()]
        event=next(e for e in events if e['action']=='group-save')
        self.assertEqual({'c1','c2','c3'},set(event['records']))
        self.assertIn('group_session',event)

    def test_write_failure_and_no_implicit_tag_or_note_overwrite(self):
        self.act('open',item_id='c1')
        self.act('save',item_id='c1',note='preserve context note')
        group=self.open()
        # Existing note is an item-level human record, not a common-note default.
        before=copy.deepcopy(self.store.state)
        with patch('hebrew_acronyms.human_review_masked.atomic_json',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.save(group,common_label='fits')
        self.assertEqual(before,self.store.state)
        self.save(group,common_label='fits')
        self.assertEqual('preserve context note',self.store.state['records']['c1']['note'])

if __name__=='__main__':unittest.main()
