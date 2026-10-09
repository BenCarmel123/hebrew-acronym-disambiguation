"""Continuation tests use invented data and isolated annotation files only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.human_review_masked import (ContinuationStore, MaskedStore, mechanical_status,
                                               technical_normalize, contains_foreign_letters, FILTER_VERSION)
from hebrew_acronyms.human_review_short import ShortStore


def fixture():
    pairs = [('old', 'ייחוס', 'רע', 'ייחוס'), ('mixed', 'נכון', ' נכון ', 'עוד נכון'),
             ('technical', 'אב״ג', 'אב"ג', 'אב-ג'), ('duplicate', 'טוב', 'זהה שגוי', 'זהה שגוי'),
             ('foreign', 'עברית', 'English עברית', 'עברית שונה'), ('context', 'אחר', 'זהה שגוי', 'הקשר נוסף'),
             ('missing', 'אמת', '', None)]
    items = []
    for item_id, gold, a, b in pairs:
        items.append({'id': item_id, 'sentence': 'משפט ' + item_id, 'acronym': 'אבג', 'gold': gold,
                      'source': 'fixture-source', 'candidates': [gold],
                      'answers': [{'id': item_id + ':' + system, 'system_id': system, 'raw': raw, 'auto_score': False, 'status': 'recorded'}
                                  for system, raw in zip(('qwen_generate', 'gemini_generate'), (a, b))]})
    old_plan = {'plan_id': 'old-fixture-plan', 'queue': ['old'], 'selection_reasons': {'old': {'group': 'both_fail'}},
                'actual': {'both_fail': 1}, 'source_counts': {'fixture-source': 1}, 'adjustments': []}
    d = {'dataset_id': 'fixture-source', 'source_identity': 'fixture-source', 'provenance': {'repository': 'fixture', 'files': []},
         'items': items, 'short_plan': old_plan,
         'continuation_plan': {'plan_id': 'continuation-fixture', 'queue': [i['id'] for i in items], 'rules_version': FILTER_VERSION}}
    legacy = {'schema_version': 'human-review-v2', 'records': {'old': {'exposure': {}, 'draft': {
        'annotator': 'QA only', 'updated_at': '2026-10-09T08:00:00+00:00', 'note': 'preserve original note',
        'answers': {'old:qwen_generate': {'quality': 'wrong'}, 'old:gemini_generate': {'quality': 'correct'}}}}}}
    return d, legacy


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.data, legacy = fixture()
        root = Path(self.tmp.name)
        oldshort = ShortStore(self.data, root/'oldshort.json', legacy, 'old full history')
        oldmasked = MaskedStore(self.data, root/'oldmasked.json', oldshort.state, 'old short history')
        self.previous = copy.deepcopy(oldmasked.state)
        self.previous_bytes = (root/'oldmasked.json').read_bytes()
        self.path = root/'continuation.json'
        self.store = ContinuationStore(self.data, self.path, self.previous, 'old masked history')

    def act(self, action, item='mixed', view='all', **kwargs):
        return self.store.transact(action, {'revision': self.store.state['revision'], 'item_id': item, 'view': view, **kwargs})

    def test_partition_and_exact_historical_preservation(self):
        c = self.store.counts()
        self.assertEqual(14, c['source_answers'])
        self.assertEqual(14, sum(c[k] for k in ('human_answers','exacttrim_answers','technical_answers','missing_answers','pending_answers')))
        self.assertEqual((2,1,1,2,8,7,1), tuple(c[k] for k in ('human_answers','exacttrim_answers','technical_answers','missing_answers','pending_answers','pending_decisions','duplicate_savings')))
        self.assertEqual(self.previous['records'], self.store.state['records'])
        self.assertEqual(self.previous, self.store.state['previous_v2_snapshot'])
        self.assertEqual(self.previous_bytes, (self.path.parent/'oldmasked.json').read_bytes())
        self.assertEqual(self.data['short_plan'], self.store.state['historical_plan'])
        self.assertEqual(self.previous['presentation']['old'], self.store.state['presentation']['old'])
        self.assertNotIn('old', self.store.snapshot()['queues']['all'])

    def test_normalization_is_limited_to_whole_string_nfc_space_and_quotes(self):
        for raw, gold, expected in [(' x ', 'x', 'exacttrim'), ('a  b','a b','technical'), ('אב״ג','אב"ג','technical'),
                                    ('אב־ג','אבג','pending'), ('אב-ג','אבג','pending'), ('x extra','x','pending'),
                                    ('dogs','dog','pending'), ('אימות','אמת','pending'), ('אמת!','אמת','pending'),
                                    ('e\u0301','é','technical')]:
            self.assertEqual(expected, mechanical_status({'raw': raw}, gold)[0], (raw,gold))
        self.assertEqual('אב-ג', technical_normalize(' אב-ג '))
        self.assertEqual('missing', mechanical_status({'raw':'normal', 'status':'technical_failure'}, 'ref')[0])

    def test_foreign_feature_excludes_marks_digits_punctuation_and_is_not_label(self):
        self.assertFalse(contains_foreign_letters('שָׁלוֹם 123 - ״׳!?'))
        for text in ('English', '中文', 'Привет', 'עברית A', 'עברית é'):
            self.assertTrue(contains_foreign_letters(text))
        self.assertEqual(['foreign'], self.store.snapshot()['queues']['foreign'])
        opened = self.act('open','foreign','foreign')['item']
        self.assertEqual(1, len(opened['answers']))
        self.assertNotIn('label', opened['answers'][0])
        self.assertFalse(self.store.state['records']['foreign']['judgments'])

    def test_answer_level_open_filtered_restore_and_reload(self):
        item = self.act('open')['item']
        self.assertEqual(1, len(item['answers']))
        self.assertEqual('עוד נכון', item['answers'][0]['text'])
        filtered = self.act('open','mixed','filtered')['item']
        self.assertEqual(1, len(filtered['answers']))
        token = filtered['answers'][0]['id']
        with self.assertRaises(ValueError): self.act('save',judgments={token:'fits'})
        self.act('restore','mixed','filtered',answer_id=token)
        item = self.act('open')['item']
        self.assertEqual(2, len(item['answers']))
        inventory = self.store.state['inventory']['mixed']
        restored = next(v for v in inventory.values() if v['restored'])
        self.assertEqual('pending', restored['status']); self.assertEqual('exacttrim',restored['mechanical_status'])
        restarted = ContinuationStore(self.data,self.path,self.previous)
        self.assertEqual(self.store.state,restarted.state)
        self.assertEqual(self.store.snapshot()['queues'],restarted.snapshot()['queues'])

    def test_duplicate_collapses_only_same_item_then_propagates_tags_after_autosave(self):
        opened = self.act('open','duplicate')['item']
        self.assertEqual(1,len(opened['answers'])); self.assertEqual(2,opened['answers'][0]['occurrence_count'])
        token = opened['answers'][0]['id']
        self.act('save','duplicate',judgments={token:'not_fits'},tags={token:['gibberish']})
        judgments = self.store.state['records']['duplicate']['judgments']
        self.assertEqual(2,len(judgments)); self.assertEqual(1,len({j['decision_id'] for j in judgments.values()}))
        self.assertTrue(all(j['occurrence_count']==2 for j in judgments.values()))
        self.assertNotIn('duplicate',self.store.snapshot()['queues']['all'])
        self.act('save','duplicate',tags={token:['extra_text']})
        self.assertTrue(all(j['tags']==['extra_text'] for j in self.store.state['records']['duplicate']['judgments'].values()))
        self.assertIn('context',self.store.snapshot()['queues']['all'])
        self.assertFalse(self.store.state['records'].get('context',{}).get('judgments'))
        self.act('save','duplicate',judgments={token:'unsure'})
        self.assertIn('duplicate',self.store.snapshot()['queues']['unsure'])
        self.assertEqual(1,len(self.act('open','duplicate','unsure')['item']['answers']))

    def test_unsure_and_suspicion_do_not_return_to_pending_or_disappear(self):
        token = self.act('open')['item']['answers'][0]['id']
        self.act('save',judgments={token:'unsure'},suspect=True)
        self.assertNotIn('mixed',self.store.snapshot()['queues']['all'])
        self.assertIn('mixed',self.store.snapshot()['queues']['unsure'])
        self.assertIn('mixed',self.store.snapshot()['queues']['suspicions'])
        shown = self.act('open','mixed','suspicions')['item']['answers']
        self.assertEqual({'human','exacttrim'},{a['review_status'] for a in shown})
        self.assertEqual(1,len(self.act('open','mixed','unsure')['item']['answers']))

    def test_historical_reveal_not_reset_and_interim_summary_always_masked(self):
        previous = copy.deepcopy(self.previous); previous['revealed']=True; previous['reveal_events']=[{'at':'2026-10-09T09:00:00+00:00'}]
        store = ContinuationStore(self.data,self.path.parent/'exposed.json',previous)
        self.assertTrue(store.snapshot()['historical_revealed']); self.assertTrue(store.snapshot()['prior_exposure'])
        self.assertEqual('after_reveal',store.phase('foreign'))
        self.assertEqual(previous,store.state['previous_v2_snapshot'])
        self.assertTrue(store.transact('summary',{'revision':0})['summary']['masked'])
        self.act('reveal')
        self.assertTrue(self.act('summary')['summary']['masked'])
        self.assertTrue(self.store.summary()['masked'])
        self.assertNotIn('qwen',json.dumps(self.store.export()))
        self.assertIn('inventory',self.store.export(False))

    def test_source_mismatch_rejected_without_annotation_mutation(self):
        before=self.path.read_bytes(); changed=copy.deepcopy(self.data);changed['provenance']['repository']='other'
        with self.assertRaises(ValueError): ContinuationStore(changed,self.path,self.previous)
        self.assertEqual(before,self.path.read_bytes())


if __name__=='__main__':unittest.main()
