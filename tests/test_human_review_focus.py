"""Bounded diagnostic sampling and timing, using invented responses only."""
import copy
import hashlib
import json
import unittest

from tests.test_human_review_identified import IdentifiedReviewTests
from hebrew_acronyms.human_review_focus import build_focus, variant_key
from hebrew_acronyms.human_review_identified import IdentifiedStore
from hebrew_acronyms.human_review_server import atomic_json


class FocusedReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = IdentifiedReviewTests(); fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        self.root, self.original = fixture.root, fixture
        self.focus = build_focus(fixture.bundle, fixture.bundle, fixture.store.state, limit=2)
        self.store = IdentifiedStore(self.focus, self.root/'focused.json')

    def test_sampling_does_not_transfer_judgments_or_drop_source_evidence(self):
        self.assertEqual(len(self.original.bundle['items']), 3)
        self.assertEqual(len(self.focus['items']), 1)  # One representative of two contexts.
        self.assertEqual(self.focus['provenance']['focused_review']['full_source_answers'], 4)
        self.assertFalse(self.store.state['records'])
        original = {i['id']: i for i in self.original.bundle['items']}
        for item in self.focus['items']:
            self.assertEqual(item, original[item['id']])
        variant = copy.deepcopy(self.focus['items'][0])
        variant['answers'][0]['raw'] = '**' + variant['answers'][0]['raw'] + '.**'
        self.assertEqual(variant_key(variant), variant_key(self.focus['items'][0]))
        self.assertFalse(self.original.store.state['records'])
        for limit, minutes in [(201,20),(200,21),(0,20)]:
            with self.assertRaises(ValueError):
                build_focus(self.original.bundle,self.original.bundle,self.original.store.state,limit=limit,minutes=minutes)

    def test_explicit_start_and_completion_cap_preserve_export(self):
        ids = self.store.queues()['all']
        self.assertNotIn('manual_session', self.store.state)
        with self.assertRaises(ValueError): self.store.transact('batch-open',dict(revision=0,item_ids=ids))
        started = self.store.transact('focus-start',dict(revision=0))['manual_session']
        from datetime import datetime
        self.assertEqual((datetime.fromisoformat(started['deadline'])-datetime.fromisoformat(started['started_at'])).total_seconds(),1200)
        self.store.transact('focus-start',dict(revision=1))
        self.assertEqual(self.store.state['manual_session'],started)
        opened=self.store.transact('batch-open',dict(revision=1,item_ids=ids))
        self.store.transact('batch-save',dict(revision=2,batch_id=opened['batch_id'],labels={i:'unsure' for i in ids},confirmed=True))
        self.assertTrue(self.store.stop_if_due())
        exported=json.loads((self.root/'review-export.json').read_text())
        self.assertEqual(len(exported['decisions']),1)
        self.assertEqual(exported['decisions'][0]['judgment']['label'],'unsure')
        with self.assertRaises(ValueError): self.store.authorize_continuation('Invented extension request')
        self.verify_backup()

    def test_expiry_keeps_drafts_separate_and_blocks_further_confirmation(self):
        self.store.transact('focus-start',dict(revision=0))
        ids=self.store.queues()['all'];opened=self.store.transact('batch-open',dict(revision=1,item_ids=ids))
        labels={i:'fits' for i in ids}
        self.store.transact('batch-draft',dict(batch_id=opened['batch_id'],labels=labels,edited_item_ids=ids))
        self.store.state['manual_session']['deadline']='2000-01-01T00:00:00+00:00'
        atomic_json(self.store.path,self.store.state)
        self.assertTrue(self.store.stop_if_due())
        resumed=IdentifiedStore(self.focus,self.store.path)
        self.assertEqual(resumed.counts()['human_decisions'],0)
        restored=resumed.transact('draft-restore',dict(batch_id=opened['batch_id']))
        self.assertTrue(restored['read_only']);self.assertEqual(restored['draft']['labels'],labels)
        with self.assertRaises(ValueError):
            resumed.transact('batch-save',dict(revision=resumed.state['revision'],batch_id=opened['batch_id'],labels=labels,confirmed=True))
        self.assertEqual(json.loads((self.root/'review-export.json').read_text())['decisions'],[])
        self.verify_backup()

    def verify_backup(self):
        backup=self.root/'verified-backup-focus'
        for name,expected in json.loads((backup/'sha256.json').read_text()).items():
            self.assertEqual(hashlib.sha256((backup/name).read_bytes()).hexdigest(),expected)
