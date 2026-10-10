"""Invented response fixtures only; never human research judgments."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.human_review_identified import build_bundle, IdentifiedStore, coverage_rows


class IdentifiedReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        rows=[dict(item_id=str(i),sentence='context '+str(i),acronym='ABC',gold_expansion='gold') for i in range(381)]
        with (self.root/'cohort.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        items=[]
        for aid,idx,raw,failed in [('a',0,'variant',False),('b',0,'variant',False),('c',1,'variant',False),('d',2,'',True)]:
            answer=dict(id=aid,system_id=aid+'_generate',task='generation',raw=raw,decoded=raw,
                        technical_failure=failed,auto_score=False,option_mapping=[],
                        binding=dict(item_id=str(idx),run_id=aid,response_sha256=hashlib.sha256(raw.encode()).hexdigest()))
            items.append(dict(id=aid,original_item_id=str(idx),sentence=rows[idx]['sentence'],acronym='ABC',gold='gold',answers=[answer]))
        (self.root/'source.json').write_text(json.dumps({'items':items}))
        self.bundle=build_bundle([self.root/'source.json'],self.root/'cohort.csv','QA_NOT_HUMAN')
        self.store=IdentifiedStore(self.bundle,self.root/'annotations.json')

    def test_exact_group_context_and_resume(self):
        self.assertEqual(len(self.bundle['items']),3)
        item=next(i for i in self.bundle['items'] if len(i['occurrences'])==2)
        opened=self.store.transact('open',dict(revision=0,item_id=item['id']))
        token=opened['item']['answers'][0]['id']
        self.store.transact('save',dict(revision=1,item_id=item['id'],judgments={token:'unsure'}))
        resumed=IdentifiedStore(self.bundle,self.root/'annotations.json')
        self.assertEqual(resumed.counts()['human_answers'],2)
        self.assertEqual(resumed.counts()['human_decisions'],1)
        self.assertEqual(resumed.counts()['pending_answers'],1)
        self.assertEqual(resumed.counts()['missing_answers'],1)
        self.assertEqual(sum(r['reviewed'] for r in coverage_rows(self.bundle,resumed.state)),2)
        self.assertEqual(sum(r['unsure'] for r in coverage_rows(self.bundle,resumed.state)),2)
        self.assertNotIn('system_id',json.dumps(opened['item']))
        resumed.summary(True);resumed.export(True);resumed.markdown(True)
        with self.assertRaises(ValueError): resumed.export(False)

    def test_deadline_blocks_work_and_verifies_backup(self):
        from datetime import datetime, timezone, timedelta
        item=self.store.queues()['all'][0]
        opened=self.store.transact('open',dict(revision=0,item_id=item))
        self.store.transact('save',dict(revision=1,item_id=item,judgments={opened['item']['answers'][0]['id']:'unsure'}))
        self.assertIn('manual_session',self.store.state)
        self.store.state['manual_session']['deadline']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        from hebrew_acronyms.human_review_server import atomic_json
        atomic_json(self.store.path,self.store.state)
        self.assertTrue(self.store.stop_if_due())
        manifest=json.loads((self.root/'verified-backup-60min/sha256.json').read_text())
        for name, expected in manifest.items():
            self.assertEqual(hashlib.sha256((self.root/'verified-backup-60min'/name).read_bytes()).hexdigest(),expected)
        with self.assertRaises(ValueError): self.store.transact('open',dict(revision=2,item_id=item))

    def test_changed_answer_rejected_and_failure_separate(self):
        changed=copy.deepcopy(self.bundle);changed['items'][0]['answers'][0]['raw']='changed'
        with self.assertRaises(ValueError): IdentifiedStore(changed,self.root/'annotations.json')
        failure=self.store.queues()['filtered'][0]
        with self.assertRaises(ValueError): self.store.transact('save',dict(revision=0,item_id=failure,judgments={}))
        with self.assertRaises(ValueError): self.store.transact('group-save',dict(revision=0))

    def test_batch_confirmation_skip_resume_and_history(self):
        ids = self.store.queues()['all']
        opened = self.store.transact('batch-open', dict(revision=0, item_ids=ids))
        self.assertEqual(self.store.counts()['human_decisions'], 0)
        self.assertNotIn('manual_session', self.store.state)
        labels = {i: 'not_fits' if n == 0 else '' for n, i in enumerate(ids)}
        payload = dict(revision=1, batch_id=opened['batch_id'], labels=labels, confirmed=True)
        for invalid in [dict(payload, confirmed=False), dict(payload, revision=0),
                        dict(payload, labels={**labels, 'unseen': 'fits'})]:
            with self.assertRaises(ValueError): self.store.transact('batch-save', invalid)
        self.store.transact('batch-save', payload)
        resumed = IdentifiedStore(self.bundle, self.store.path)
        self.assertEqual(resumed.counts()['human_decisions'], 1)
        self.assertIn('manual_session', resumed.state)
        self.assertEqual(resumed.counts()['pending_decisions'], 1)
        event = json.loads(self.store.path.with_suffix('.history.jsonl').read_text().splitlines()[-1])
        self.assertEqual(set(event['records']), set(ids))
        self.assertEqual(event['batch']['judged_item_ids'], [ids[0]])
        with self.assertRaises(ValueError): resumed.transact('batch-save', dict(payload, revision=2))
        resumed.restore_masked(resumed.export(True), resumed.state['revision'])
        self.assertEqual(resumed.counts()['human_decisions'], 1)

    def test_batch_rejects_failure_and_deadline(self):
        with self.assertRaises(ValueError):
            self.store.transact('batch-open', dict(revision=0, view='filtered', item_ids=self.store.queues()['filtered']))
        ids = self.store.queues()['all']
        opened = self.store.transact('batch-open', dict(revision=0, item_ids=ids))
        self.store.state['manual_session'] = {'deadline': '2000-01-01T00:00:00+00:00'}
        with self.assertRaises(ValueError):
            self.store.transact('batch-save', dict(revision=1, batch_id=opened['batch_id'], labels={i:'fits' for i in ids}, confirmed=True))
        self.assertEqual(self.store.counts()['human_decisions'], 0)

    def test_table_http_save(self):
        import threading
        import urllib.request
        from hebrew_acronyms.human_review_server import make_server
        server = make_server(self.bundle, self.root/'http-annotations.json', port=0, qa=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def call(action, payload):
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/short/{action}',
                data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response: return json.load(response)
        try:
            ids = server.short_store.queues()['all']
            opened = call('batch-open', dict(revision=0, item_ids=ids))
            result = call('batch-save', dict(revision=1, batch_id=opened['batch_id'],
                          confirmed=True, labels={i:'unsure' for i in ids}))
            self.assertEqual(result['state']['counts']['human_answers'], 3)
            self.assertEqual(result['state']['counts']['human_decisions'], 2)
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_unconfirmed_draft_survives_deadline_and_restart(self):
        from hebrew_acronyms.human_review_server import atomic_json
        ids=self.store.queues()['all']
        opened=self.store.transact('batch-open',dict(revision=0,item_ids=ids))
        labels={ids[0]:'fits',ids[1]:'unsure'}
        self.store.transact('batch-draft',dict(batch_id=opened['batch_id'],labels=labels,edited_item_ids=ids))
        self.assertNotIn('manual_session',self.store.state)
        self.assertEqual(self.store.counts()['human_decisions'],0)
        records=copy.deepcopy(self.store.state['records'])
        self.store.state['manual_session']={'started_at':'1999-12-31T23:00:00+00:00','deadline':'2000-01-01T00:00:00+00:00'}
        atomic_json(self.store.path,self.store.state)
        self.assertTrue(self.store.stop_if_due())
        # A delayed persistence request after expiry still cannot approve labels.
        self.store.transact('batch-draft',dict(batch_id=opened['batch_id'],labels=labels,edited_item_ids=ids))
        resumed=IdentifiedStore(self.bundle,self.store.path)
        restored=resumed.transact('draft-restore',dict(batch_id=opened['batch_id']))
        self.assertTrue(restored['read_only'])
        self.assertEqual(restored['draft']['labels'],labels)
        self.assertTrue(restored['draft']['received_after_deadline'])
        self.assertEqual(resumed.state['records'],records)
        self.assertEqual(resumed.counts()['human_decisions'],0)
        self.assertEqual(sum(r['reviewed'] for r in coverage_rows(self.bundle,resumed.state)),0)
        with self.assertRaises(ValueError):
            resumed.transact('batch-save',dict(revision=resumed.state['revision'],batch_id=opened['batch_id'],confirmed=True,labels=labels))
        manifest=json.loads((self.root/'verified-backup-60min/sha256.json').read_text())
        for name,expected in manifest.items():
            self.assertEqual(hashlib.sha256((self.root/'verified-backup-60min'/name).read_bytes()).hexdigest(),expected)
        self.assertIn('annotations.drafts.json',manifest)
        from hebrew_acronyms.human_review_identified import export_judgments
        self.assertEqual(export_judgments(self.bundle,resumed.state,self.root/'export.json')['decisions'],[])

    def test_draft_confirmation_resumes_without_duplicates(self):
        ids=self.store.queues()['all'];opened=self.store.transact('batch-open',dict(revision=0,item_ids=ids))
        labels={i:'unsure' for i in ids}
        self.store.transact('batch-draft',dict(batch_id=opened['batch_id'],labels=labels,edited_item_ids=ids))
        restored=self.store.transact('draft-restore',dict(revision=1,batch_id=opened['batch_id']))
        self.assertFalse(restored['read_only'])
        self.store.transact('batch-save',dict(revision=2,batch_id=opened['batch_id'],labels=labels,confirmed=True))
        resumed=IdentifiedStore(self.bundle,self.store.path)
        self.assertEqual(resumed.queues()['all'],[])
        self.assertEqual(resumed.snapshot()['pending_drafts'],[])
        self.assertEqual(resumed.counts()['human_decisions'],2)
        with self.assertRaises(ValueError): resumed.transact('draft-restore',dict(batch_id=opened['batch_id']))

    def test_historical_reuse_requires_confirmation_and_preserves_sources(self):
        from hebrew_acronyms.human_review_reuse import verified_reuse_proposals
        from hebrew_acronyms.human_review_identified import export_judgments
        session=self.root/'artifacts'/'review';session.mkdir(parents=True)
        item=next(i for i in self.bundle['items'] if len(i['occurrences'])==2)
        old_item=dict(id=item['original_item_id'],sentence=item['sentence'],acronym=item['acronym'],gold=item['gold'],
                      answers=[dict(id='old-answer',task='generation',raw=item['answers'][0]['raw'])])
        judgment=dict(label='fits',origin='old-protocol',label_updated_at='2020-01-01T00:00:00+00:00',
                      annotator='QA_NOT_HUMAN',exposure_evidence={'status':'after_reveal'})
        data={'source_identity':'old-source','items':[old_item]}
        annotations={'records':{old_item['id']:{'judgments':{'old-answer':judgment},'exposure':{'old':'preserved'}}}}
        sources={'review-data-continuation-v1.json':data,'annotations.continuation-v1.json':annotations,'review-data-381.json':self.bundle}
        hashes={}
        for name,content in sources.items():
            path=self.root/name;path.write_text(json.dumps(content));hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        match=dict(group_id=item['id'],original_item_id=old_item['id'],new_bindings=[a['binding'] for a in item['occurrences']],
                   previous_judgments=[dict(answer_id='old-answer',**{k:judgment[k] for k in ['label','origin','label_updated_at','annotator']})])
        report=dict(source_sha256=hashes,historical_exact_context_response_matches=[match],match_count=1,new_answer_occurrences=2)
        path=session/'coordinator-review.json';path.write_text(json.dumps(report))
        store=IdentifiedStore(self.bundle,session/'annotations.json')
        before=copy.deepcopy(store.state['records'])
        payload=dict(revision=0,item_ids=[item['id']],confirmed=True)
        with self.assertRaises(ValueError): store.transact('reuse-confirm',payload)
        store.transact('reuse-open',dict(revision=0))
        self.assertEqual(store.counts()['reused_answers'],0)
        store.transact('reuse-confirm',dict(payload,revision=1))
        resumed=IdentifiedStore(self.bundle,store.path)
        self.assertEqual(resumed.counts()['human_decisions'],0)
        self.assertEqual(resumed.counts()['human_answers'],0)
        self.assertEqual(resumed.counts()['reused_answers'],2)
        self.assertEqual(resumed.state['records'],before)
        self.assertNotIn('manual_session',resumed.state)
        self.assertNotIn(item['id'],resumed.queues()['all'])
        reuse=resumed.state['historical_reuse'][item['id']]
        self.assertEqual(reuse['proposal']['historical_sources'][0]['judgment'],judgment)
        exported=export_judgments(self.bundle,resumed.state,self.root/'export.json')
        self.assertEqual(exported['decisions'],[])
        self.assertEqual(exported['historical_reuse'][item['id']],reuse)
        for name,h in hashes.items(): self.assertEqual(hashlib.sha256((self.root/name).read_bytes()).hexdigest(),h)
        # Include every matching historical answer: contradictory evidence blocks reuse.
        data['items'][0]['answers'].append(dict(old_item['answers'][0],id='contradiction'))
        annotations['records'][old_item['id']]['judgments']['contradiction']=dict(judgment,label='not_fits')
        for name,content in list(sources.items())[:2]:
            (self.root/name).write_text(json.dumps(content));report['source_sha256'][name]=hashlib.sha256((self.root/name).read_bytes()).hexdigest()
        report['historical_exact_context_response_matches'][0]['previous_judgments'].append(dict(match['previous_judgments'][0],answer_id='contradiction',label='not_fits'))
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'Conflicting'): verified_reuse_proposals(path,self.root,self.bundle)

    def test_signed_backup_restore_and_reveal(self):
        i=self.store.queues()['all'][0]
        opened=self.store.transact('open',dict(revision=0,item_id=i));token=opened['item']['answers'][0]['id']
        self.store.transact('save',dict(revision=1,item_id=i,judgments={token:'fits'}))
        backup=self.store.export(True)
        self.store.restore_masked(backup,self.store.state['revision'])
        self.store.transact('reveal',dict(revision=self.store.state['revision']))
        self.store.export(False);self.store.markdown(False)
        self.assertTrue(self.store.state['pre_reveal_snapshot'])


if __name__=='__main__': unittest.main()
