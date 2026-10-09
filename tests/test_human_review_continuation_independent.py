"""Independent continuation checks with invented text and isolated annotation files."""
import copy
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.human_review_masked import (
    ContinuationStore, MaskedStore, contains_foreign_letters,
    mechanical_status, technical_normalize,
)
from hebrew_acronyms.human_review_short import ShortStore
from tests.test_human_review_masked import fixture


class IndependentFilterRulesTests(unittest.TestCase):
    def status(self, raw, gold, **metadata):
        answer = {'raw': raw, 'status': 'recorded', 'missing': False,
                  'auto_score': False, 'auto_valid': False, **metadata}
        before = copy.deepcopy(answer)
        result = mechanical_status(answer, gold)
        self.assertEqual(answer, before, 'Filtering must not mutate original results')
        return result[0]

    def test_whole_trim_exact_has_distinct_precedence(self):
        self.assertEqual('exacttrim', self.status(' \nאב גד\t ', 'אב גד'))
        self.assertEqual('technical', self.status('אב\t \nגד', 'אב גד'))
        self.assertEqual('technical', self.status('אב\u00a0גד', 'אב גד'))

    def test_nfc_and_explicit_quote_equivalents_only(self):
        self.assertEqual('technical', self.status('e\u0301', 'é'))
        for double in ('״', '“', '”'):
            with self.subTest(double=double):
                self.assertEqual('technical', self.status('א' + double + 'ב', 'א"ב'))
        for single in ('׳', '‘', '’'):
            with self.subTest(single=single):
                self.assertEqual('technical', self.status('ג' + single, "ג'"))
        self.assertNotEqual(technical_normalize('Ａ'), technical_normalize('A'), 'NFKC is not authorized')

    def test_forbidden_semantic_spelling_punctuation_and_substring_shortcuts(self):
        pairs = [('שקל חדש', 'שקלים חדשים'), ('כתיב', 'כתב'), ('מכון', 'מיכוון'),
                 ('תל-אביב', 'תל אביב'), ('תל־אביב', 'תל אביב'), ('תל—אביב', 'תל אביב'),
                 ('אב, גד', 'אב גד'), ('אב גד.', 'אב גד'), ('(אב גד)', 'אב גד'),
                 ('פירוש: אב גד', 'אב גד'), ('אב גד והסבר נוסף', 'אב גד'),
                 ('אבא', 'אב'), ('אָב', 'אב'), ('A', 'a')]
        for response, gold in pairs:
            with self.subTest(response=response, gold=gold):
                self.assertEqual('pending', self.status(response, gold))

    def test_invalid_automatic_score_is_neither_missing_nor_technical_failure(self):
        self.assertEqual('pending', self.status('תשובה ממשית', 'ייחוס אחר'))
        self.assertEqual('pending', self.status('Error 500', 'ייחוס אחר'))
        self.assertEqual('missing', self.status('server text retained', 'ייחוס', status='technical_failure'))
        for raw in (None, '', ' \t\n'):
            self.assertEqual('missing', self.status(raw, 'ייחוס'))

    def test_foreign_detector_counts_unicode_letters_not_marks_digits_or_punctuation(self):
        for text in ('אבְגּ 123 — !?', 'אַב', '״׳“”‘’', '\u0342\u0301', '🙂 Ⅻ ١٢'):
            with self.subTest(text=text):
                self.assertFalse(contains_foreign_letters(text))
        for text in ('אב A', 'שלוםé', 'Ж', '中', 'العربية', 'カ', 'Ω', 'ɑ'):
            with self.subTest(text=text):
                self.assertTrue(contains_foreign_letters(text))


class IndependentContinuationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, legacy = fixture()
        for item in self.data['items']:
            for answer in item['answers']:
                if answer['system_id'].endswith('_generate'):
                    answer['raw'] = 'invented ' + answer['id']
                    answer['status'] = 'recorded'
                    answer['missing'] = False
        by_id = {i['id']: i for i in self.data['items']}
        def set_pair(item_id, a, b):
            answers = [x for x in by_id[item_id]['answers'] if x['system_id'].endswith('_generate')]
            answers[0]['raw'], answers[1]['raw'] = a, b
        set_pair('item0', by_id['item0']['gold'], by_id['item0']['gold'])
        set_pair('item1', 'same text', ' same text ')
        set_pair('item2', 'identical raw', 'identical raw')
        set_pair('item3', 'identical raw', 'another raw')
        set_pair('item4', by_id['item4']['gold'], 'needs review')
        set_pair('item5', by_id['item5']['gold'], by_id['item5']['gold'])
        old = ShortStore(self.data, self.root / 'short-v1.json', legacy)
        previous = MaskedStore(self.data, self.root / 'short-v2.json', old.state, 'prior-v1-history')
        self.previous = copy.deepcopy(previous.state)
        self.prior_bytes = previous.path.read_bytes()
        self.data['continuation_plan'] = {'plan_id': 'independent-continuation-fixture',
            'queue': [i['id'] for i in self.data['items']], 'rules_version': 'generation-filter-v1'}
        self.path = self.root / 'continuation.json'
        self.store = ContinuationStore(self.data, self.path, self.previous, 'prior-v2-history')

    def action(self, action, item_id, **kwargs):
        return self.store.transact(action, {'revision': self.store.state['revision'], 'item_id': item_id, **kwargs})

    def test_history_and_human_decisions_retained_and_partition_balances(self):
        self.assertEqual(self.previous['records'], self.store.state['records'])
        self.assertEqual(self.previous, self.store.state['previous_v2_snapshot'])
        self.assertEqual(self.prior_bytes, (self.root / 'short-v2.json').read_bytes())
        c = self.store.counts()
        self.assertEqual(48, c['source_answers'])
        self.assertEqual(c['source_answers'], sum(c[k] for k in
            ('human_answers', 'exacttrim_answers', 'technical_answers', 'missing_answers', 'pending_answers')))
        self.assertEqual(2, c['human_answers'])
        self.assertTrue(all(e['status'] == 'human' and e['mechanical_status'] == 'exacttrim'
                            for e in self.store.state['inventory']['item0'].values()))
        self.assertNotIn('item0', self.store.queues()['all'])

    def test_dedup_is_raw_identical_within_same_item_only(self):
        self.assertEqual(2, len(self.store.groups('item1')), 'Trim equality cannot merge response occurrences')
        self.assertEqual(1, len(self.store.groups('item2')))
        self.assertEqual(2, len(self.store.groups('item2')[0]))
        self.assertEqual(2, len(self.store.groups('item3')), 'Matching text in another context stays separate')
        self.assertEqual(1, self.store.counts()['duplicate_savings'])

    def test_filtered_answer_hidden_and_restore_preserves_source_and_work(self):
        opened = self.action('open', 'item4')['item']
        self.assertEqual(['needs review'], [a['text'] for a in opened['answers']])
        original = copy.deepcopy(self.store.state['inventory']['item4'])
        token = opened['answers'][0]['id']
        self.action('save', 'item4', judgments={token: 'unsure'})
        self.assertNotIn('item4', self.store.queues()['all'])
        self.assertIn('item4', self.store.queues()['unsure'])
        filtered = self.action('open', 'item4', view='filtered')['item']['answers']
        self.assertEqual(1, len(filtered))
        self.action('restore', 'item4', view='filtered', answer_id=filtered[0]['id'])
        self.assertIn('item4', self.store.queues()['all'])
        self.assertIn('item4', self.store.queues()['unsure'])
        for aid, record in self.store.state['inventory']['item4'].items():
            self.assertEqual(original[aid]['original_answer'], record['original_answer'])
        reloaded = ContinuationStore(self.data, self.path, self.previous, 'prior-v2-history')
        self.assertEqual(self.store.state, reloaded.state)
        self.assertEqual(self.store.queues(), reloaded.queues())

    def test_suspected_item_survives_even_when_both_answers_mechanically_filtered(self):
        self.action('open', 'item5', view='filtered')
        self.action('save', 'item5', view='filtered', suspect=True)
        self.assertNotIn('item5', self.store.queues()['all'])
        self.assertIn('item5', self.store.queues()['suspicions'])
        self.assertEqual(2, sum(len(group) for group in self.store.groups('item5', 'suspicions')))

    def test_masked_export_carries_every_answer_filter_rule_without_model_mapping(self):
        exported = self.store.export()
        self.assertEqual(self.data['source_identity'], exported['source_identity'])
        ledger = exported['answer_ledger']
        self.assertEqual(48, sum(len(answers) for answers in ledger.values()))
        self.assertEqual(set(self.store.answers), set(ledger))
        for item_id, answers in ledger.items():
            slots = self.store.state['presentation'][item_id]
            self.assertEqual({slot['token'] for slot in slots}, set(answers))
            for slot in slots:
                original = self.store.state['inventory'][item_id][slot['answer_id']]
                public = answers[slot['token']]
                self.assertEqual(original['status'], public['review_status'])
                self.assertEqual(original['filter_rule'], public['filter_rule'])
                self.assertEqual(original['rules_version'], public['rules_version'])
                self.assertEqual(original['original_answer']['raw'], public['original_text'])
        filtered = ledger['item5']
        self.assertEqual({'exacttrim'}, {v['review_status'] for v in filtered.values()})
        self.assertTrue(all(v['filter_rule'] for v in filtered.values()))

    def test_roundtrip_does_not_erase_historical_exposure_evidence(self):
        before = copy.deepcopy(self.store.state['records']['item0']['judgments'])
        exported = self.store.export()
        self.store.restore_masked(exported, self.store.state['revision'])
        after = self.store.state['records']['item0']['judgments']
        for aid, judgment in before.items():
            for key in ('exposure_evidence', 'tags_exposure_evidence', 'label_updated_at', 'tags_updated_at', 'label_phase', 'tags_phase'):
                self.assertEqual(judgment.get(key), after[aid].get(key), key)

    def test_masked_backup_restores_labels_and_duplicate_decision_identity(self):
        item = self.action('open', 'item2')['item']
        token = item['answers'][0]['id']
        self.action('save', 'item2', judgments={token: 'fits'}, tags={token: ['equivalent']})
        before = copy.deepcopy(self.store.state['records']['item2']['judgments'])
        exported = self.store.export()
        self.action('open', 'item2', view='suspicions')
        self.action('save', 'item2', view='suspicions', judgments={token: 'unsure'}, tags={token: ['spelling']})
        self.store.restore_masked(exported, self.store.state['revision'])
        restored = self.store.state['records']['item2']['judgments']
        for aid in before:
            for key in ('label', 'tags', 'decision_id', 'occurrence_count', 'applied_to_original_answers'):
                self.assertEqual(before[aid][key], restored[aid][key], key)
        self.assertEqual(self.previous, self.store.state['previous_v2_snapshot'])
        reloaded = ContinuationStore(self.data, self.path, self.previous, 'prior-v2-history')
        self.assertEqual(self.store.state, reloaded.state)

    def test_single_duplicate_decision_applied_to_two_occurrences_with_common_identity(self):
        opened = self.action('open', 'item2')['item']
        self.assertEqual(1, len(opened['answers']))
        self.assertEqual(2, opened['answers'][0]['occurrence_count'])
        self.action('save', 'item2', judgments={opened['answers'][0]['id']: 'fits'},
                    tags={opened['answers'][0]['id']: ['equivalent']})
        decisions = list(self.store.state['records']['item2']['judgments'].values())
        self.assertEqual(2, len(decisions))
        self.assertEqual({'fits'}, {d['label'] for d in decisions})
        self.assertTrue(decisions[0].get('decision_id'))
        self.assertEqual(decisions[0]['decision_id'], decisions[1]['decision_id'])
        self.assertEqual({2}, {d.get('occurrence_count') for d in decisions})
        self.assertNotIn('item2', self.store.queues()['all'])
        self.assertIn('item3', self.store.queues()['all'])


if __name__ == '__main__':
    unittest.main()
