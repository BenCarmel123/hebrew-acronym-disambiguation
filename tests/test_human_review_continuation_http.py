"""Continuation HTTP integration uses isolated fixtures and immutable prior work."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from hebrew_acronyms.human_review_masked import MaskedStore
from hebrew_acronyms.human_review_server import make_server
from hebrew_acronyms.human_review_short import ShortStore
from tests.test_human_review_masked import fixture


class ContinuationHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.data, legacy = fixture()
        old = ShortStore(self.data, self.folder / 'annotations.short-v1.json', legacy)
        v2 = MaskedStore(self.data, self.folder / 'annotations.short-v2.json', old.state)
        self.previous = v2.path.read_bytes()
        self.data['short_protocol'] = 'qualitative-generation-v3'
        self.data['continuation_plan'] = {
            'plan_id': 'fixture-continuation', 'queue': [i['id'] for i in self.data['items']],
            'rules_version': 'generation-filter-v1', 'source_identity': self.data['source_identity']}
        self.server = make_server(self.data, self.folder / 'annotations.json', port=0, qa=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def call(self, path, payload=None):
        request = urllib.request.Request(self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_session_identifies_continuation_without_changing_historical_plan(self):
        _, text = self.call('/api/session')
        session = json.loads(text)
        self.assertEqual('qualitative-generation-v3', session['short_protocol'])
        self.assertEqual('fixture-continuation', session['short_plan_id'])
        self.assertTrue(session['qa'])
        self.assertEqual(20, len(self.data['short_plan']['queue']))
        self.assertEqual(self.previous, (self.folder / 'annotations.short-v2.json').read_bytes())

    def test_legacy_routes_blocked_and_masked_summary_remains_safe(self):
        for route in ('/api/data', '/api/state', '/api/export.json', '/legacy', '/api/short/full-export.json'):
            self.assertEqual(403, self.call(route)[0], route)
        state = json.loads(self.call('/api/short/state')[1])
        code, body = self.call('/api/short/summary', {'revision': state['revision']})
        self.assertEqual(200, code, body)
        self.assertTrue(json.loads(body)['summary']['masked'])
        for route in ('/api/short/export.json', '/api/short/summary/export.md'):
            code, text = self.call(route)
            self.assertEqual(200, code)
            for secret in ('qwen', 'gemini', 'auto_score', 'source-secret', 'original_answer_id'):
                self.assertNotIn(secret, text)

    def test_reveal_does_not_turn_stop_summary_into_unmasked_summary(self):
        state = json.loads(self.call('/api/short/state')[1])
        code, text = self.call('/api/short/reveal', {'revision': state['revision']})
        self.assertEqual(200, code, text)
        result = json.loads(text)
        self.assertFalse(result['summary']['masked'])
        code, text = self.call('/api/short/summary', {'revision': result['state']['revision']})
        self.assertEqual(200, code, text)
        self.assertTrue(json.loads(text)['summary']['masked'])
        self.assertEqual(200, self.call('/api/short/full-export.json')[0])
        self.assertEqual(self.previous, (self.folder / 'annotations.short-v2.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
