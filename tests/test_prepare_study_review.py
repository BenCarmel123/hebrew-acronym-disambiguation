"""P1 diagnostic tests on invented files only; no models, research or network."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.data_processing import prepare_study_review as review

FIELDS = ['item_id', 'acronym', 'sentence', 'gold_expansion', 'candidates',
          'n_candidates', 'source', 'page_title', 'category', 'provenance',
          'label_origin', 'label_status', 'review_verdict', 'review_note']


def item(identifier, acronym='א״ב', label='אור בהיר', doc='doc', category='wiki_substituted', **overrides):
    result = dict(item_id=identifier, acronym=acronym, sentence=f'הנושא הוא {acronym} היום {identifier}.',
                  gold_expansion=label, candidates=f'{label} | אור אחר', n_candidates='2',
                  source='knesset' if category=='knesset' else 'wikipedia', page_title=doc,
                  category=category, provenance='invented fixture', label_origin='fixture',
                  label_status='provisional', review_verdict='', review_note='')
    result.update(overrides)
    return result


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(review.csv_text(rows, fields), encoding='utf-8-sig')


def fixture(root):
    train = [item('t1', doc='control-one'), item('t2', doc='control-two')]
    dev = [item('d1', 'ג״ד', 'גן דשא', 'dev-doc', review_verdict='unsure'),
           item('d2', 'ג״ד', 'גן חדש', 'dev-changed')]
    test = [item('x1', 'ה״ו', 'הר ורוד', 'test-doc', category='knesset')]
    extra = [item('nd1', 'ג״ד', 'גן דשא', 'new-dev-doc', category='knesset'),
             item('e1', 'ה״ו', 'הר ורוד', 'extra-one'),
             item('e2', 'ה״ו', 'הר ורוד', 'extra-two')]
    all_rows = train + dev + test + extra
    for role, rows in [('train', train), ('dev', dev), ('test', test), ('all', all_rows)]:
        write(root/f'data/splits/{role}_items.csv', FIELDS, rows)
    candidate_rows = [{'acronym': a, 'expansion': e} for a,e in [('א״ב','אור בהיר'),('ג״ד','גן דשא'),('ה״ו','הר ורוד')]]
    for name in ['candidate_table.csv', 'merged_counts.csv', 'wiktionary/wiktionary_counts.csv', 'wikipedia/bullet_counts.csv']:
        write(root/'data/mined'/name, ['acronym','expansion'], candidate_rows)
    write(root/'data/mined/dev_review.csv', ['item_id','acronym','expansion','verdict','note'], [
        dict(item_id='d1', acronym='ג״ד', expansion='גן דשא', verdict='unsure', note='fixture unsure'),
        dict(item_id='d2', acronym='ג״ד', expansion='תווית ישנה', verdict='wrong_sense', note='fixture old label'),
        dict(item_id='absent', acronym='ג״ד', expansion='תווית חסרה', verdict='broken', note='preserve unmatched')])
    write(root/'data/mined/knesset/knesset_reviewed.csv', FIELDS, [])
    for name in ['duplicate_review.csv','merge_review.csv','wikipedia/unmined_triage.csv']:
        write(root/'data/mined'/name, ['acronym','verdict'], [])
    (root/'data/README.md').write_text('invented fixture', encoding='utf-8')
    pair_source = root/'src/hebrew_acronyms/models/common/pairs.py'
    pair_source.parent.mkdir(parents=True)
    pair_source.write_text('# invented source version reference', encoding='utf-8')
    return all_rows


class ReviewPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.rows = fixture(self.root)
        self.output = self.root/'data/study_v1/review'

    def run_prepare(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')):
            return review.prepare(self.root)

    def test_sources_preserved_and_records_traceable(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = self.run_prepare()
        self.assertEqual(report['natural_dev_proposed']['rows'], 1)
        self.assertEqual(report['metadata_difference_items'], 0)
        self.assertEqual(report['historical_label_changed_review_links'], 1)
        self.assertEqual(report['unmatched_review_records'], 1)
        for p, value in before.items():
            self.assertEqual(p.read_bytes(), value)
        audits = review.read_csv(self.output/'item_audit.csv')
        self.assertEqual(len(audits), len(self.rows))
        self.assertEqual(len({r['stable_item_id'] for r in audits}), len(audits))
        for audit in audits:
            for ref in json.loads(audit['source_refs_json']):
                source = review.read_csv(self.root/ref['path'])[ref['record']-1]
                self.assertEqual(review.row_id(source), audit['stable_item_id'])
        changed = next(r for r in audits if r['raw_item_id']=='d2')
        self.assertIn('historical_adverse_review', changed['flags'])
        self.assertEqual(json.loads(changed['historical_label_changed_review_evidence_json'])[0]['source_record']['expansion'], 'תווית ישנה')
        unsure = next(r for r in audits if r['raw_item_id']=='d1')
        self.assertIn('uncertain_review', unsure['flags'])

    def test_decisions_blank_then_byte_preserved(self):
        self.run_prepare()
        path = self.output/'review_decisions.csv'
        rows = review.read_csv(path)
        self.assertTrue(any(r['review_id'].startswith('inv-') for r in rows))
        self.assertTrue(all(not v for r in rows for k,v in r.items() if k!='review_id'))
        rows[0].update(reviewer='Fixture human', decision='uncertain', elapsed_seconds='41', problem_types='target')
        path.write_text(review.csv_text(rows, review.DECISION_FIELDS), encoding='utf-8')
        before = path.read_bytes()
        self.run_prepare()
        self.assertEqual(path.read_bytes(), before)
        (self.root/'data/README.md').write_text('changed fixture')
        with self.assertRaisesRegex(ValueError, 'Sources changed'):
            self.run_prepare()
        self.assertEqual(path.read_bytes(), before)

    def test_symlink_preflight_no_partial_refresh(self):
        self.run_prepare()
        manifest = self.output/'source_manifest.json'
        before = manifest.read_bytes()
        victim = self.root/'external.txt'
        victim.write_text('protected')
        target = self.output/'pilot_rubric.md'
        target.unlink()
        target.symlink_to(victim)
        with self.assertRaisesRegex(ValueError, 'output symlink'):
            self.run_prepare()
        self.assertEqual(victim.read_text(), 'protected')
        self.assertEqual(manifest.read_bytes(), before)

    def test_output_directory_symlink_refused(self):
        outside = self.root/'outside'
        outside.mkdir()
        self.output.parent.mkdir(parents=True)
        self.output.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_prepare()
        self.assertEqual(list(outside.iterdir()), [])

    def test_identity_and_spans_are_not_legacy_id_or_first_occurrence(self):
        one = item('collision')
        two = item('collision', sentence='א״ב נאמר אחרת.')
        self.assertNotEqual(review.row_id(one), review.row_id(two))
        self.assertEqual(review.row_id(one), review.row_id(dict(one, item_id='renamed', review_note='changed')))
        self.assertEqual(review.spans('ובא"ב, א״ב', 'א״ב'), [[2, 5], [7, 10]])
        self.assertEqual(review.spans('אין מטרה', 'א״ב'), [])
        self.assertTrue(review.scope_reason('ארבע (גימטרייה)'))
        self.assertTrue(review.scope_reason(review.IDENTITY))
        self.assertFalse(review.scope_reason('תעודת זהות'))

    def test_extension_matches_actual_pair_budget_and_gold_does_not_select(self):
        train = [item('t1', doc='c1'), item('t2', doc='c2'), item('t3', doc='c3')]
        extension = [item('e1','ה״ו','הר ורוד','a'), item('e2','ה״ו','הר ורוד','b')]
        test = [item('x','ה״ו','הר ורוד','test',category='knesset')]
        _, members, summary = review.extension_scenarios(train, extension, test)
        capped = summary['capped_5_min_2_documents']
        self.assertEqual(capped['stored_candidate_pairs'], 4)
        self.assertEqual(capped['historical_pair_builder_pairs'], 4)
        self.assertEqual(capped['control_pairs'], 4)
        self.assertEqual(capped['retained_full_training_usable_pairs'], 6)
        self.assertEqual(capped['replacement_condition_full_usable_pairs'], 6)
        _, changed_members, _ = review.extension_scenarios(train, extension, [dict(test[0], gold_expansion='OTHER')])
        self.assertEqual(members, changed_members)
        unusable = [dict(extension[0], sentence='missing target'), extension[1]]
        _, _, unusable_summary = review.extension_scenarios(train, unusable, test)
        self.assertFalse(unusable_summary['exact_candidate_count']['exact_source_row_pair_match'])
        self.assertNotEqual(unusable_summary['exact_candidate_count']['stored_candidate_pairs'],
                            unusable_summary['exact_candidate_count']['historical_pair_builder_pairs'])

    def test_pilot_contains_no_test_and_inventory_not_filled_from_gold(self):
        self.run_prepare()
        pilot = review.read_csv(self.output/'pilot_items.csv')
        self.assertFalse(any('test' in r['historical_roles'] for r in pilot))
        inv = review.read_csv(self.output/'inventory_proposed.csv')
        unsupported = [r for r in inv if r['expansion_raw']=='גן חדש']
        self.assertEqual(unsupported[0]['proposal'], 'unsupported_row_candidate_review')
        self.assertEqual(unsupported[0]['independent_export_evidence_json'], '[]')
        self.assertEqual(unsupported[0]['human_decision'], '')

    def test_pilot_quotas_are_diverse_deterministic_and_exclude_test(self):
        audits = []
        for group, category, role, risk in [
            ('natural_dev', 'knesset', 'aggregate_only', 'natural_dev_proposal'),
            ('train_wiki_natural', 'wiki_natural', 'train', 'review_not_proven'),
            ('routine_reviewed_knesset', 'knesset', 'train', 'inventory_conflict'),
            ('central_risk', 'wiki_substituted', 'dev', 'uncertain_review')]:
            for i in range(4):
                audits.append(dict(stable_item_id=f'{group}-{i}', historical_roles=role,
                                   natural_dev_proposed=str(group=='natural_dev').lower(),
                                   raw_category=category, raw_acronym=f'{group}-type-{i}',
                                   scope_proposal='', flags=risk, risk_score=review.RISK[risk],
                                   review_evidence_json='[{"verdict":"clean"}]'))
        # High-risk duplicates of one issue profile must not fill a whole stratum.
        wiki = [r for r in audits if r['raw_category']=='wiki_natural']
        audits.extend(dict(r, stable_item_id='high-'+r['stable_item_id'],
                           raw_acronym='high-'+r['raw_acronym'],
                           flags='review_not_proven|target_multiple', risk_score=13) for r in wiki)
        audits.append(dict(audits[0], stable_item_id='forbidden', historical_roles='test'))
        chosen, coverage = review.select_pilot(audits)
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len({r['raw_acronym'] for r,_ in chosen}), 16)
        self.assertEqual([r['selected'] for r in coverage[:4]], [4,4,4,4])
        self.assertEqual(coverage[-1]['selected'], 0)
        self.assertEqual(review.select_pilot(list(reversed(audits))), (chosen, coverage))
        self.assertFalse(any(r['historical_roles']=='test' for r,_ in chosen))
        selected_wiki = [r for r,g in chosen if g=='train_wiki_natural']
        self.assertLessEqual(sum('target_multiple' in r['flags'] for r in selected_wiki), 2)

    def test_additive_decision_migration_preserves_every_legacy_and_unknown_value(self):
        self.run_prepare()
        path = self.output/'review_decisions.csv'
        rows = review.read_csv(path)
        old_fields = review.DECISION_FIELDS[:-3] + ['future_human_note']
        old = [{k: r.get(k, '') for k in old_fields} for r in rows]
        old[0].update(reviewer='Fixture Reviewer', decision='uncertain', elapsed_seconds='23.5',
                      reason='Preserve all text, commas, and \nline breaks.', future_human_note='unknown field value')
        path.write_text(review.csv_text(old, old_fields), encoding='utf-8')
        self.run_prepare()
        migrated = review.read_csv(path)
        self.assertEqual([r['review_id'] for r in old], [r['review_id'] for r in migrated])
        for a,b in zip(old,migrated):
            self.assertEqual(a, {k:b[k] for k in old_fields})
            self.assertTrue(all(b[k]=='' for k in review.DECISION_FIELDS[-3:]))
        before = path.read_bytes()
        self.run_prepare()
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_decision_schema_fails_without_overwriting(self):
        self.run_prepare()
        path = self.output/'review_decisions.csv'
        path.write_text('review_id,review_id\nitem-a,item-b\n', encoding='utf-8')
        before = {p.name:p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, 'Invalid decision schema'):
            self.run_prepare()
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_empty_partial_and_complete_pilot_registration_preserves_decisions(self):
        report = self.run_prepare()
        self.assertEqual(report['pilot']['status'], 'empty_registration')
        self.assertIsNone(report['pilot']['measured_seconds'])
        path = self.output/'review_decisions.csv'
        rows = review.read_csv(path)
        pilot = review.read_csv(self.output/'pilot_items.csv')
        selected = {r['review_id'] for r in pilot}
        by_id = {r['review_id']:r for r in rows}
        by_id[pilot[0]['review_id']]['initial_blind_expansion']='fixture independent answer'
        by_id[pilot[1]['review_id']]['elapsed_seconds']='12.5'
        path.write_text(review.csv_text(rows, review.DECISION_FIELDS), encoding='utf-8')
        before = path.read_bytes()
        report = self.run_prepare()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report['pilot']['status'], 'partial_registration')
        self.assertEqual(report['pilot']['rows_with_entries'], 2)
        self.assertEqual(report['pilot']['rows_with_decision'], 0)
        self.assertEqual(report['pilot']['measured_seconds'], 12.5)
        for identifier in selected:
            by_id[identifier].update(reviewer='Fixture Reviewer', decision='uncertain',
                                     decided_at='2026-09-28T12:00:00+03:00', elapsed_seconds='10', problem_types='label')
        # An out-of-pilot inventory decision is counted separately in the full queue.
        extra = next(r for r in rows if r['review_id'] not in selected)
        extra['reason']='a partial record outside current pilot'
        path.write_text(review.csv_text(rows, review.DECISION_FIELDS), encoding='utf-8')
        before = path.read_bytes()
        report = self.run_prepare()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report['pilot']['status'], 'complete_registration')
        self.assertEqual(report['pilot']['measured_seconds'], 10*len(selected))
        self.assertEqual(report['review_registration']['rows_with_entries'], len(selected)+1)
        self.assertEqual(report['pilot']['scientific_approval'], 'not inferred; protocol pending')
        displayed = {r['review_id']:r for r in review.read_csv(self.output/'review_queue.csv')}
        self.assertTrue(all(displayed[k]['human_decision']=='uncertain' for k in selected))
        self.assertTrue(all(r['human_registration_status']=='complete_registration'
                            for r in review.read_csv(self.output/'pilot_items.csv')))
        self.assertNotIn('has not been\nperformed', (self.output/'pilot_rubric.md').read_text())

    def test_review_evidence_is_bound_to_resolved_content_not_legacy_tuple(self):
        first = item('collision', category='knesset', doc='first')
        second = item('collision', category='knesset', doc='second', sentence='משפט אחר עם א״ב.')
        for role in ('train','all'):
            path = self.root/f'data/splits/{role}_items.csv'
            write(path, FIELDS, review.read_csv(path)+[first,second])
        write(self.root/'data/mined/knesset/knesset_reviewed.csv', FIELDS, [dict(first, review_verdict='clean')])
        dev_path = self.root/'data/mined/dev_review.csv'
        dev_rows = review.read_csv(dev_path)
        dev_rows.append(dict(item_id='collision', acronym=first['acronym'], expansion=first['gold_expansion'],
                             verdict='clean', note='No sentence: ambiguous, must stay unassigned'))
        write(dev_path, list(dev_rows[0]), dev_rows)
        report = self.run_prepare()
        by_id = {r['stable_item_id']:r for r in review.read_csv(self.output/'item_audit.csv')}
        self.assertEqual(len(json.loads(by_id[review.row_id(first)]['review_evidence_json'])), 1)
        self.assertEqual(json.loads(by_id[review.row_id(second)]['review_evidence_json']), [])
        self.assertIn('review_not_proven', by_id[review.row_id(second)]['flags'])
        self.assertEqual(report['unmatched_review_records'], 2)
        unmatched = review.read_csv(self.output/'unmatched_review_evidence.csv')
        self.assertTrue(any(json.loads(r['source_record_json'])['item_id']=='collision' for r in unmatched))

    def test_import_has_no_write_or_network_side_effect(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')), \
             patch.object(Path, 'write_text', side_effect=AssertionError('Write forbidden')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('Write forbidden')):
            importlib.reload(review)


if __name__ == '__main__':
    unittest.main()
