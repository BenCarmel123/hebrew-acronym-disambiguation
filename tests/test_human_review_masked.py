"""Masked-route QA fixtures; no research source or human annotation writes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.human_review_masked import MaskedStore, PROTOCOL, TAGS
from hebrew_acronyms.human_review_short import ShortStore, build_plan, digest, PROTOCOL as OLD


def fixture():
    items = []
    for n in range(24):
        scores = (False, True, False, False) if n < 16 else ((False, False, False, False) if n < 20 else (True, True, True, True))
        answers = [{'id': f'item{n}:{system}', 'system_id': system, 'raw': 'unchanged raw response ' + str(n),
                    'auto_score': score, 'auto_score_rule': 'original-rule', 'task': system.split('_')[1]}
                   for system, score in zip(('qwen_generate', 'qwen_select', 'gemini_generate', 'gemini_select'), scores)]
        items.append({'id': f'item{n}', 'sentence': 'fixture sentence', 'gold': 'fixture reference',
                      'acronym': str(n), 'answers': answers, 'candidates': ['fixture reference'],
                      'source': 'source-secret', 'source_metadata': {'private_filename': 'original-secret.csv'}})
    d = {'dataset_id': 'fixture', 'source_identity': 'fixture', 'provenance': {'repository': 'fixture-repo', 'files': []}, 'items': items}
    legacy = {'schema_version': 'human-review-v2', 'records': {'item0': {'exposure': {'auto_scores': '2026-10-09T09:00:00+00:00'},
               'draft': {'annotator': 'QA_ONLY', 'updated_at': '2026-10-09T08:00:00+00:00', 'note': 'old qwen auto_score note',
                         'answers': {'item0:qwen_generate': {'quality': 'wrong'}, 'item0:gemini_generate': {'quality': 'correct'}}}}}}
    d['short_plan'] = build_plan(d, legacy)
    return d, legacy


class MaskedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dataset, legacy = fixture()
        self.oldpath = Path(self.tmp.name) / 'short-v1.json'
        old = ShortStore(self.dataset, self.oldpath, legacy, 'full-v2 immutable history')
        self.previous = copy.deepcopy(old.state)
        self.oldbytes = self.oldpath.read_bytes()
        self.path = Path(self.tmp.name) / 'short-v2.json'
        self.store = MaskedStore(self.dataset, self.path, self.previous, 'short-v1 immutable history')
        self.newid = next(i for i in self.dataset['short_plan']['queue'] if i != 'item0')

    def action(self, action, item_id=None, **kwargs):
        return self.store.transact(action, {'revision': self.store.state['revision'], 'item_id': item_id or self.newid, **kwargs})

    def open(self, item_id=None):
        return self.action('open', item_id)['item']

    def test_original_bundle_and_histories_preserved_byte_for_byte(self):
        self.open(); self.action('summary')
        self.assertEqual(self.oldbytes, self.oldpath.read_bytes())
        self.assertEqual(self.previous, self.store.state['previous_protocol'])
        self.assertEqual('short-v1 immutable history', self.store.state['previous_history'])
        self.assertEqual('full-v2 immutable history', self.store.state['previous_protocol']['legacy_history'])
        labels = self.store.state['records']['item0']['judgments']
        self.assertEqual('not_fits', labels['item0:qwen_generate']['label'])
        self.assertEqual('fits', labels['item0:gemini_generate']['label'])
        self.assertEqual('prior_protocol', labels['item0:qwen_generate']['label_phase'])
        self.assertEqual('2026-10-09T08:00:00+00:00', labels['item0:qwen_generate']['label_updated_at'])
        self.assertEqual([], labels['item0:qwen_generate']['tags'])
        self.assertIsNone(labels['item0:qwen_generate']['tags_updated_at'])

    def test_per_item_random_order_is_stable_and_migration_tracks_original_answers(self):
        class AlternatingRandom:
            calls = 0
            def shuffle(self, values):
                type(self).calls += 1
                if type(self).calls % 2: values.reverse()
        with patch('hebrew_acronyms.human_review_masked.secrets.SystemRandom', AlternatingRandom):
            path = Path(self.tmp.name) / 'ordered.json'
            store = MaskedStore(self.dataset, path, self.previous)
        self.assertEqual(len(self.dataset['short_plan']['queue']), AlternatingRandom.calls)
        order = store.state['presentation']
        self.assertEqual(2, len({store.answers[i][slots[0]['answer_id']]['system_id'] for i, slots in order.items()}))
        restarted = MaskedStore(self.dataset, path, self.previous)
        self.assertEqual(order, restarted.state['presentation'])
        self.assertEqual(store.public_item('item0'), restarted.public_item('item0'))
        tokens = [s['token'] for slots in order.values() for s in slots]
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_labels_remain_attached_to_original_answer_under_random_order(self):
        item = self.open()
        first = item['answers'][0]['id']
        original = self.store.state['presentation'][self.newid][0]['answer_id']
        self.action('save', judgments={first: 'fits'})
        self.assertEqual('fits', self.store.state['records'][self.newid]['judgments'][original]['label'])
        self.assertEqual('fits', self.store.snapshot()['records'][self.newid]['judgments'][first]['label'])
        self.assertEqual('before_reveal', self.store.state['records'][self.newid]['judgments'][original]['label_phase'])

    def test_tags_optional_independent_editable_and_exported_without_semantic_inference(self):
        item = self.open(); a, b = [x['id'] for x in item['answers']]
        self.action('save', judgments={a: '', b: 'fits'}, tags={a: ['gibberish', 'extra_text'], b: ['spelling']})
        r = self.store.snapshot()['records'][self.newid]
        self.assertEqual(1, r['completion']['judged_answers'])
        self.assertEqual('', r['judgments'][a]['label'])
        self.assertEqual('fits', r['judgments'][b]['label'])
        self.assertEqual(['extra_text', 'gibberish'], r['judgments'][a]['tags'])
        before = r['judgments'][b]['label_updated_at']
        self.action('save', tags={a: [], b: ['inflection', 'equivalent']})
        exported = self.store.export()
        j = exported['records'][self.newid]['judgments']
        self.assertEqual('not_marked', j[a]['tag_status'])
        self.assertEqual(before, j[b]['label_updated_at'])
        self.assertTrue(j[b]['tags_updated_at'])
        restarted = MaskedStore(self.dataset, self.path, self.previous)
        self.assertEqual(self.store.state, restarted.state)

    def test_public_routes_mask_system_scores_sources_reasons_and_old_notes(self):
        item = self.open()
        self.action('save', judgments={item['answers'][0]['id']: 'fits'}, tags={item['answers'][0]['id']: ['spelling']})
        outputs = [self.store.snapshot(), self.action('details'), self.action('summary'), self.store.export(), self.store.markdown(), item]
        for output in outputs:
            text = json.dumps(output, ensure_ascii=False)
            for secret in ('qwen', 'gemini', 'auto_score', 'original-rule', 'source-secret', 'original-secret.csv', 'selection_reason', 'previous_protocol', 'original_answer_id', 'comparison'):
                self.assertNotIn(secret, text)
        self.assertEqual('unchanged raw response ' + self.newid.removeprefix('item'), item['answers'][0]['text'])
        self.assertFalse(self.store.state['revealed'])

    def test_full_outputs_gated_and_reveal_freezes_before_edits(self):
        for fn in (lambda: self.store.summary(False), lambda: self.store.export(False), lambda: self.store.markdown(False)):
            with self.assertRaises(ValueError): fn()
        item = self.open(); aid = item['answers'][0]['id']
        self.action('save', judgments={aid: 'fits'}, tags={aid: ['spelling']})
        original = self.store.state['presentation'][self.newid][0]['answer_id']
        result = self.action('reveal')
        self.assertTrue(result['state']['revealed']); self.assertFalse(result['summary']['masked'])
        frozen = copy.deepcopy(self.store.state['pre_reveal_snapshot'])
        self.action('save', judgments={aid: 'not_fits'}, tags={aid: ['extra_text']})
        self.assertEqual(frozen, self.store.state['pre_reveal_snapshot'])
        j = self.store.state['records'][self.newid]['judgments'][original]
        self.assertEqual('after_reveal', j['label_phase']); self.assertEqual('after_reveal', j['tags_phase'])
        self.assertEqual('fits', frozen['records'][self.newid]['judgments'][original]['label'])
        self.action('reveal')
        self.assertEqual(1, len(self.store.state['reveal_events']))
        self.assertEqual(frozen, self.store.state['pre_reveal_snapshot'])
        masked = self.store.export()
        self.assertEqual('fits', masked['pre_reveal_snapshot']['records'][self.newid]['judgments'][aid]['label'])
        full = self.store.export(False)
        self.assertEqual(frozen, full['pre_reveal_snapshot'])
        self.assertIn('answer_id', full['summary']['cases'][-1]['answers'][0])
        self.assertTrue(self.store.summary()['masked'] is False)

    def test_new_tags_on_old_judgment_have_separate_phase_and_timestamp(self):
        item = self.open('item0'); token = item['answers'][0]['id']
        old = copy.deepcopy(self.store.snapshot()['records']['item0']['judgments'][token])
        self.action('save', 'item0', tags={token: ['punctuation']})
        j = self.store.snapshot()['records']['item0']['judgments'][token]
        self.assertEqual('prior_protocol', j['label_phase'])
        self.assertEqual(old['label_updated_at'], j['label_updated_at'])
        self.assertEqual('after_reveal', j['tags_phase'])
        self.assertTrue(j['tags_updated_at'])

    def test_previous_global_summary_and_details_affect_new_judgment_phase(self):
        for kind in ('summary', 'details', 'unknown'):
            previous = copy.deepcopy(self.previous)
            if kind == 'summary': previous['summary_exposures'] = [{'at': '2026-10-09T10:00:00+00:00', 'items': ['item0']}]
            elif kind == 'details': previous['records']['item0']['exposure']['case_details'] = {'at': '2026-10-09T10:00:00+00:00'}
            else: previous.pop('summary_exposures')
            store = MaskedStore(self.dataset, Path(self.tmp.name) / (kind + '.json'), previous)
            opened = store.transact('open', {'revision': 0, 'item_id': self.newid})
            token = opened['item']['answers'][0]['id']
            store.transact('save', {'revision': 1, 'item_id': self.newid, 'judgments': {token: 'fits'}})
            j = next(iter(store.state['records'][self.newid]['judgments'].values()))
            self.assertEqual('unknown' if kind == 'unknown' else 'after_reveal', j['label_phase'])
            self.assertFalse(store.state['revealed'])

    def test_inherited_timing_excludes_later_global_reveal(self):
        for when, expected in [('2026-10-09T07:00:00+00:00', 'after_reveal'), ('2026-10-09T10:00:00+00:00', 'unknown'), (None, 'unknown')]:
            previous = copy.deepcopy(self.previous)
            previous['summary_exposures'] = [{'at': when}]
            store = MaskedStore(self.dataset, Path(self.tmp.name) / ('time-' + str(when).replace(':', '') + '.json'), previous)
            old = store.state['records']['item0']['judgments']['item0:qwen_generate']
            self.assertEqual('prior_protocol', old['label_phase'])
            self.assertEqual(expected, old['exposure_evidence']['status'])
        previous = copy.deepcopy(self.previous)
        previous['records'][self.newid] = {'judgments': {}, 'suspect': False, 'example': False, 'note': '',
                                         'exposure': {'reference_and_generation': {'at': '2026-10-09T07:00:00+00:00'}}}
        store = MaskedStore(self.dataset, Path(self.tmp.name) / 'previously-open.json', previous)
        self.assertEqual('unknown', store.phase(self.newid))

    def test_concurrent_same_revision_has_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        item = self.open(); token = item['answers'][0]['id']; revision = self.store.state['revision']
        def save(label):
            try:
                self.store.transact('save', {'revision': revision, 'item_id': self.newid, 'judgments': {token: label}})
                return 'saved'
            except ValueError:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(save, ['fits', 'not_fits']))
        self.assertCountEqual(['saved', 'conflict'], outcomes)
        self.assertEqual(revision + 1, self.store.state['revision'])

    def test_masked_backup_restores_only_same_random_handles_and_keeps_exposure(self):
        item = self.open(); token = item['answers'][0]['id']
        self.action('save', judgments={token: 'fits'}, tags={token: ['equivalent']})
        backup = self.store.export()
        self.action('save', judgments={token: 'unsure'}, tags={token: []})
        self.store.restore_masked(backup, self.store.state['revision'])
        self.assertEqual('fits', self.store.snapshot()['records'][self.newid]['judgments'][token]['label'])
        self.assertIn('reference_and_generation', self.store.state['records'][self.newid]['exposure'])
        changed = copy.deepcopy(backup); changed['records'][self.newid]['judgments'][token]['label_phase'] = 'prior_protocol'
        with self.assertRaises(ValueError): self.store.restore_masked(changed, self.store.state['revision'])
        other = MaskedStore(self.dataset, Path(self.tmp.name) / 'other.json', self.previous)
        with self.assertRaises(ValueError): other.restore_masked(backup, 0)

    def test_stale_unknown_labels_tags_and_source_changes_rejected(self):
        item = self.open(); token = item['answers'][0]['id']; before = self.path.read_bytes()
        for payload in ({'revision': 0, 'judgments': {token: 'fits'}}, {'revision': 1, 'judgments': {'qwen_generate': 'fits'}}, {'revision': 1, 'judgments': {token: 'correct'}}, {'revision': 1, 'tags': {token: ['invented']}}):
            with self.assertRaises(ValueError): self.store.transact('save', {'item_id': self.newid, **payload})
        self.assertEqual(before, self.path.read_bytes())
        changed = copy.deepcopy(self.dataset); changed['provenance']['repository'] = 'different'
        with self.assertRaises(ValueError): MaskedStore(changed, self.path, self.previous)
        self.assertEqual(before, self.path.read_bytes())

    def test_write_failure_preserves_state_and_reveal_snapshot(self):
        self.open(); before = copy.deepcopy(self.store.state)
        with patch('hebrew_acronyms.human_review_masked.atomic_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.action('reveal')
        self.assertEqual(before, self.store.state)
        self.assertFalse(self.store.state['revealed'])

    def test_sample_and_reviewed_composition_only_after_reveal(self):
        self.assertNotIn('composition', self.store.summary())
        self.assertNotIn('generation_fail_selection_pass', json.dumps(self.store.export()))
        item = self.open()
        self.action('save', judgments={item['answers'][0]['id']: 'unsure'})
        summary = self.action('reveal')['summary']
        composition = summary['composition']
        self.assertEqual(20, sum(composition['selected'].values()))
        self.assertEqual(2, sum(composition['reviewed'].values()))
        self.assertEqual(1, sum(composition['completed'].values()))
        self.assertEqual(2, sum(composition['sources_reviewed'].values()))
        self.assertEqual(self.dataset['short_plan']['actual'], composition['selected'])
        self.assertIn('הרכב המדגם שנבחר והחלק שנבדק', self.store.markdown(False))
        self.assertNotIn('composition', self.store.summary(True))
        self.assertNotIn('הרכב המדגם שנבחר והחלק שנבדק', self.store.markdown(True))
        self.assertNotIn('generation_fail_selection_pass', json.dumps(self.store.export(True)))

    def test_categories_include_correctly_scored_tagged_cases(self):
        item = self.open(); tokens = [a['id'] for a in item['answers']]
        self.action('save', judgments={t: 'not_fits' for t in tokens}, tags={tokens[0]: ['gibberish']})
        summary = self.action('reveal')['summary']
        case = next(c for c in summary['cases'] if c['id'] == self.newid)
        self.assertEqual('gibberish', case['answers'][0]['tags'][0])
        self.assertIn('accepted_human_rejected_auto', summary)
        self.assertIn('rejected_human_accepted_auto', summary)
        self.assertIn('unresolved', summary)
        self.assertIn('suspicions', summary)


if __name__ == '__main__': unittest.main()
