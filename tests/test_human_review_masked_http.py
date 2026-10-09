"""Masking is enforced across HTTP entry points, not only display templates."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from tests.test_human_review_short import fixture
from hebrew_acronyms.human_review_short import ShortStore
from hebrew_acronyms.human_review_server import make_server


class MaskedHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.d,legacy=fixture();self.d['short_protocol']='qualitative-generation-v2'
        root=Path(self.tmp.name);self.old=ShortStore(self.d,root/'annotations.short-v1.json',legacy)
        self.before=self.old.path.read_bytes()
        self.server=make_server(self.d,root/'annotations.json',port=0,qa=True)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close_server)
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def close_server(self):
        self.server.shutdown();self.thread.join();self.server.server_close()

    def call(self,path,payload=None):
        request=urllib.request.Request(self.base+path,data=json.dumps(payload).encode() if payload is not None else None,
                                       headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request) as r:return r.status,r.read().decode()
        except urllib.error.HTTPError as e:return e.code,e.read().decode()

    def test_all_historical_and_full_endpoints_blocked_before_reveal(self):
        for path in ('/api/data','/api/state','/api/export.json','/api/export.csv','/legacy','/app.js','/review_logic.js',
                     '/api/short/full-export.json','/api/short/full-summary.md'):
            self.assertEqual(self.call(path)[0],403,path)
        self.assertEqual(self.call('/api/update',{})[0],403)
        self.assertEqual(self.call('/api/import',{})[0],403)
        self.assertEqual(self.old.path.read_bytes(),self.before)

    def test_masked_payloads_exports_and_summary_have_no_identity_or_scores(self):
        state=json.loads(self.call('/api/short/state')[1])
        for action in ('open','details','summary'):
            code,text=self.call('/api/short/'+action,{'revision':state['revision'],'item_id':'i0'})
            self.assertEqual(code,200,text)
            state=json.loads(text)['state']
            for leak in ('qwen','gemini','auto_score','selection_reason','original_answer_id','source_metadata'):
                self.assertNotIn(leak,text)
        for path in ('/api/short/export.json','/api/short/summary/export.md'):
            code,text=self.call(path);self.assertEqual(code,200)
            for leak in ('qwen','gemini','auto_score','original_answer_id','legacy_snapshot'):
                self.assertNotIn(leak,text)
        self.assertFalse(self.server.short_store.state['revealed'])

    def test_explicit_reveal_gates_full_summary_and_export_and_freezes(self):
        state=json.loads(self.call('/api/short/state')[1])
        code,text=self.call('/api/short/reveal',{'revision':state['revision']})
        self.assertEqual(code,200,text)
        payload=json.loads(text);self.assertFalse(payload['summary']['masked'])
        self.assertIn('scoring_rule',payload['summary']);self.assertIn('recommendations',payload['summary'])
        self.assertTrue(self.server.short_store.state['pre_reveal_snapshot'])
        code,text=self.call('/api/short/full-export.json');self.assertEqual(code,200)
        self.assertIn('qwen',text)
        code,text=self.call('/api/short/export.json');self.assertEqual(code,200)
        self.assertNotIn('qwen',text)
        self.assertEqual(self.old.path.read_bytes(),self.before)

if __name__=='__main__':unittest.main()
