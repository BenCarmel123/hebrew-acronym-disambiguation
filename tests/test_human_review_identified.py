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
