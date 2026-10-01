"""Invented fixtures only: no repository research files or model imports."""
import csv
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.data_processing import prepare_encoder_inputs as p
from hebrew_acronyms.models.common.pairs import build_pairs


def source(acronym='א״ב', sentence='נבדק א״ב היום.', label='אור בהיר', **extra):
    row = dict(item_id='legacy-collision', acronym=acronym, sentence=sentence,
        gold_expansion=label, candidates=label+' | אפשרות אחרת', n_candidates='2',
        source='wikipedia', page_title='invented document', category='wiki_substituted',
        provenance='substituted', label_origin='substitution', label_status='weak',
        review_verdict='unreviewed', review_note='')
    row.update(extra)
    return row


def audit(row, role='train', record=1, natural=False):
    path = {'train': 'train_items.csv', 'dev': 'dev_items.csv',
            'test': 'test_items.csv', 'aggregate_only': 'all_items.csv'}[role]
    return {'stable_item_id': p.audit_id(row),
        'source_refs_json': p.canonical([{'path': 'data/splits/'+path, 'record': record}]),
        'historical_roles': role, 'natural_dev_proposed': str(natural).lower(),
        'doc_id': p.document_id(row), 'review_evidence_json': '[]', 'flags': '',
        **{'raw_'+k: v for k,v in row.items()}}


def decision(row, **extra):
    result = dict(review_id=p.audit_id(row), reviewer='שקד', decision='accept',
        label_decision='accept', corrected_label='', target_decision='accept',
        approved_span_start=str(row['sentence'].index(row['acronym'])),
        approved_span_end=str(row['sentence'].index(row['acronym'])+len(row['acronym'])),
        reason='Invented explicit approval fixture', evidence_reference='invented source record',
        decided_at='2026-01-01T00:00:00+00:00')
    result.update(extra)
    return {'source_record': 1, 'values': result}


def evidence(row):
    return {'path': 'data/mined/knesset/knesset_reviewed.csv', 'record': 1,
            'verdict': 'clean', 'source_record': deepcopy(row)}


def write_csv(path, rows, fields=None):
    fields = fields or sorted(set().union(*(r.keys() for r in rows)))
    path.write_bytes(p.csv_bytes(rows, fields))


class QualificationFixtures(unittest.TestCase):
    def test_quotes_raw_prefix_and_unicode_offsets(self):
        row = source(sentence='שלום א"ב היום.')
        out, reasons, _ = p.qualify(row, audit(row), {})
        self.assertFalse(reasons)
        self.assertEqual(out['target_raw'], 'א"ב')
        self.assertEqual(out['sentence'], row['sentence'])
        for sentence in ('שלום וא״ב היום.', 'שלום א״בים היום.', 'שלום xא״ב היום.', 'שלום א״ב2 היום.'):
            row = source(sentence=sentence)
            self.assertIn('target_boundary_ambiguous', p.qualify(row, audit(row), {})[1])
        row = source(sentence='שלום וא״ב היום.')
        d = decision(row, target_decision='correct', approved_span_start='5', approved_span_end='9')
        out, reasons, _ = p.qualify(row, audit(row), d)
        self.assertFalse(reasons)
        self.assertEqual(out['target_raw'], 'וא״ב')
        self.assertEqual(out['span_basis'], 'saved_human_span')

    def test_prefix_outside_quotes_requires_approved_boundary(self):
        row=source(sentence='היום ב"א״ב" התקיים דיון.')
        self.assertEqual(p.locate_target(row['sentence'],row['acronym'])[1],'target_boundary_ambiguous')
        d=decision(row,target_decision='correct',approved_span_start='5',approved_span_end='11')
        out,reasons,_=p.qualify(row,audit(row),d)
        self.assertFalse(reasons);self.assertEqual(out['target_raw'],'ב"א״ב"')

    def test_repeated_missing_and_overlapping_matches(self):
        self.assertEqual(p.locate_target('א״ב ואז א״ב', 'א״ב')[1], 'target_multiple')
        self.assertEqual(p.locate_target('aaaa', 'aa')[1], 'target_multiple')
        self.assertEqual(p.locate_target('ללא מטרה', 'א״ב')[1], 'target_missing')
        row=source(sentence='א״ב ואז א״ב')
        out, reasons, _ = p.qualify(row, audit(row), decision(row, approved_span_start='8', approved_span_end='11'))
        self.assertFalse(reasons)
        self.assertEqual(out['span_start'], 8)

    def test_candidates_never_repaired(self):
        for candidates, expected in [('אור בהיר | אור בהיר', 'duplicate_candidates'),
                ('אור בהיר | ', 'invalid_candidates'), ('חלופה | שנייה', 'gold_not_exactly_one_candidate'),
                ('אור בהיר', 'singleton_not_pair_loss')]:
            row=source(candidates=candidates)
            out,reasons,_=p.qualify(row,audit(row),{})
            self.assertIn(expected,reasons)
            self.assertEqual(out['candidates'],candidates)
        row=source(candidates=' אור בהיר | אפשרות אחרת ')
        self.assertFalse(p.qualify(row,audit(row),{})[1])

    def test_human_correction_missing_inventory_is_held(self):
        row=source(); original=deepcopy(row)
        out,reasons,_=p.qualify(row,audit(row),decision(row,label_decision='correct',corrected_label='תיקון חדש'))
        self.assertEqual(out['gold_expansion'],'תיקון חדש')
        self.assertIn('gold_not_exactly_one_candidate',reasons)
        self.assertEqual(out['candidates'],original['candidates'])
        self.assertEqual(row,original)

    def test_uncertainty_exclusion_unknown_author_and_invalid_span(self):
        row=source()
        for edits, reason in [({'decision':'uncertain','label_decision':'uncertain'},'human_label_unresolved'),
                ({'decision':'exclude_proposed','label_decision':'exclude_proposed'},'human_exclusion_recorded'),
                ({'reviewer':'AI'},'unverified_decision_authority'),
                ({'approved_span_start':'99','approved_span_end':'101'},'invalid_saved_human_span'),
                ({'target_decision':'uncertain'},'human_target_unresolved')]:
            self.assertIn(reason,p.qualify(row,audit(row),decision(row,**edits))[1])

    def test_evidence_requires_content_not_legacy_id_or_note(self):
        row=source(source='knesset',category='knesset',label_origin='human_review')
        a=audit(row); e=evidence(row); a['review_evidence_json']=p.canonical([e])
        self.assertEqual(len(p.exact_review_evidence(a,row)[0]),1)
        for key,value in [('sentence','different text'),('gold_expansion','different label'),('page_title','different doc'),('candidates','different inventory')]:
            bad=deepcopy(e);bad['source_record'][key]=value
            a['review_evidence_json']=p.canonical([bad])
            self.assertEqual(p.exact_review_evidence(a,row)[0],[])
        a['review_evidence_json']='[]'; row['review_note']='approved'; row['review_verdict']='clean'
        self.assertEqual(p.exact_review_evidence(a,row)[0],[])

    def test_test_and_unknown_role_gate_precedes_payload_access(self):
        class Poison(dict):
            def __getitem__(self,key):
                if key in ('raw_sentence','raw_gold_expansion','raw_candidates'):
                    raise AssertionError('Test payload touched')
                return super().__getitem__(key)
            def get(self,key,default=None):
                return self[key] if key in self else default
        row=Poison(audit(source(),role='test'))
        with patch.object(p,'read_rows',return_value=iter([(1,row)])):
            safe,types,docs,counts=p.read_scoped_audit('unused')
        self.assertEqual(safe,[])
        self.assertEqual(len(types),1);self.assertEqual(len(docs),1)
        self.assertEqual(counts['test'],1)
        for roles in ('train+test','train|test','train test','train,test'):
            row['historical_roles']=roles
            self.assertEqual(p.role_gate(row)[0],'test')
        row=audit(source());row['historical_roles']='train+dev'
        self.assertEqual(p.role_gate(row)[0],'unknown')
        row['historical_roles']='';self.assertEqual(p.role_gate(row)[0],'unknown')
        row=audit(source());row['source_refs_json']='not JSON'
        self.assertEqual(p.role_gate(row)[0],'unknown')
        row=audit(source());row['source_refs_json']=p.canonical([{'path':'data/splits/test_items.csv','record':1}])
        self.assertEqual(p.role_gate(row)[0],'test')

    def test_overall_exclusion_and_unknown_status_veto(self):
        row=source()
        for status in ('exclude_proposed','unsupported'):
            out,reasons,_=p.qualify(row,audit(row),decision(row,decision=status))
            self.assertTrue(reasons)
        a=audit(row)
        refs=json.loads(a['source_refs_json'])+[{'path':'unknown.csv','record':1}]
        a['source_refs_json']=p.canonical(refs)
        self.assertEqual(p.role_gate(a)[0],'unknown')

    def test_local_proposal_requires_exact_input_and_human_approval(self):
        row=source()
        local={'audit_item_id':p.audit_id(row),'source_row_sha256':p.digest(p.canonical(row)),
            'decision_id':'local-fixture','proposal':{'gold_expansion':'מאושר חדש',
                'candidate_replacement':{'old':'אור בהיר','new':'מאושר חדש'}},
            'human_decision':{'status':'pending'}}
        out,reasons,_=p.qualify(row,audit(row),{},local)
        self.assertIn('AI_proposal_pending_human_decision',reasons)
        self.assertEqual(out['gold_expansion'],'אור בהיר')
        local['human_decision']={'status':'approved','reviewer':'Shaked','response':'Explicit fixture approval'}
        old=decision(row,decision='uncertain',label_decision='uncertain')
        out,reasons,_=p.qualify(row,audit(row),old,local)
        self.assertFalse(reasons)
        self.assertEqual(out['gold_expansion'],'מאושר חדש')
        self.assertEqual(out['candidates'],'מאושר חדש | אפשרות אחרת')
        self.assertEqual(out['label_evidence'],'Shaked_review_with_AI_assistance')
        local['additional_human_approvals']=[{'status':'pending'}]
        self.assertIn('AI_proposal_pending_human_decision',p.qualify(row,audit(row),{},local)[1])
        local.pop('additional_human_approvals')
        local['source_row_sha256']='stale'
        with self.assertRaisesRegex(ValueError,'stale'):p.qualify(row,audit(row),{},local)

    def test_local_span_approval_does_not_claim_new_label_review(self):
        row=source(sentence='א״ב ואז א״ב',source='knesset',category='knesset',label_origin='human_review')
        a=audit(row);a['review_evidence_json']=p.canonical([evidence(row)])
        local={'audit_item_id':p.audit_id(row),'source_row_sha256':p.digest(p.canonical(row)),
            'decision_id':'local-span','proposal':{'span':[0,3]},
            'human_decision':{'status':'approved','reviewer':'Shaked','response':'First target approved'}}
        out,reasons,_=p.qualify(row,a,{},local)
        self.assertFalse(reasons);self.assertEqual(out['span_start'],0)
        self.assertEqual(out['label_evidence'],'historical_human_review_attributed_to_Ben')

    def test_reserved_markers_are_held(self):
        row=source(sentence='[ACR] א״ב היום.')
        self.assertIn('invalid_encoder_span',p.qualify(row,audit(row),{})[1])


class EndToEndFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.train=[source(),source(acronym='ג״ד',sentence='היום ג״ד נבדק.',page_title='other doc')]
        self.dev=source(acronym='ה״ו',sentence='היום ה״ו נבדק.',source='knesset',
            page_title='dev doc',category='knesset',provenance='knesset',label_origin='human_review',review_verdict='clean')
        self.audits=[audit(r,record=i) for i,r in enumerate(self.train,1)]
        dev_a=audit(self.dev,'aggregate_only',natural=True)
        dev_a['review_evidence_json']=p.canonical([evidence(self.dev)])
        self.audits.append(dev_a)
        self.decisions=[]
        self.policy={'rules_version':p.VERSION,'reference_commit':'fixture-base',
            'source_paths':{k:k for k in ('train_export','historical_dev_export','scoped_audit_source','human_decisions','local_policy')},
            'rules':['invented fixture'],'authority':{'user':'fixture'},'limits':['fixture only']}

    def run_preparation(self):
        root=self.root
        write_csv(root/'train.csv',self.train)
        write_csv(root/'old_dev.csv',[],list(self.train[0]))
        write_csv(root/'audit.csv',self.audits)
        write_csv(root/'decisions.csv',self.decisions, list(decision(self.train[0])['values']))
        (root/'policy.json').write_text(json.dumps(self.policy))
        before={f:f.read_bytes() for f in root.iterdir() if f.is_file()}
        original_open=Path.open
        def protected_open(path,*args,**kwargs):
            if path.name in ('test_items.csv','all_items.csv'):
                raise AssertionError('Forbidden source opened')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',protected_open):
            manifest=p.prepare_inputs(train_path=root/'train.csv',historical_dev_path=root/'old_dev.csv',
                audit_path=root/'audit.csv',decisions_path=root/'decisions.csv',policy_path=root/'policy.json',output_dir=root/'outputs')
        self.assertEqual(before,{f:f.read_bytes() for f in before})
        return manifest

    def outputs(self,name):
        return list(csv.DictReader(io.StringIO((self.root/'outputs'/name).read_text())))

    def test_same_legacy_id_distinct_stable_ids_contract_and_reproduction(self):
        m=self.run_preparation();rows=self.outputs('train.csv')
        self.assertEqual(len(rows),2);self.assertEqual(len({r['item_id'] for r in rows}),2)
        self.assertEqual(len(build_pairs(rows)),4)
        first={f.name:f.read_bytes() for f in (self.root/'outputs').iterdir()}
        self.assertEqual(m,self.run_preparation())
        self.assertEqual(first,{f.name:f.read_bytes() for f in (self.root/'outputs').iterdir()})

    def test_test_metadata_blocks_type_and_document_without_exposure(self):
        hidden=source(acronym=self.train[0]['acronym'],sentence='FORBIDDEN TEST TEXT',
                      label='FORBIDDEN LABEL',page_title=self.train[1]['page_title'])
        self.audits.append(audit(hidden,'test'))
        m=self.run_preparation();self.assertEqual(m['counts']['train']['rows'],0)
        blob=''.join(f.read_text() for f in (self.root/'outputs').iterdir())
        self.assertNotIn('FORBIDDEN',blob)
        reasons=[r['reasons'] for r in self.outputs('trace.csv') if r['split']=='train']
        self.assertIn('reserved_type_overlap',reasons[0]);self.assertIn('reserved_document_overlap',reasons[1])

    def test_dev_overlap_reserved_even_when_dev_label_held(self):
        self.train[0]['page_title']='dev doc'; self.train[0]['source']='knesset'
        self.audits[0]=audit(self.train[0])
        self.decisions=[decision(self.dev,decision='uncertain',label_decision='uncertain')['values']]
        m=self.run_preparation()
        self.assertEqual(m['counts']['train']['rows'],1);self.assertEqual(m['counts']['dev']['rows'],0)
        self.assertIn('dev_document_overlap',next(r for r in self.outputs('trace.csv') if r['split']=='train' and r['page_title']=='dev doc')['reasons'])

    def test_duplicate_content_records_all_held(self):
        self.train.append(deepcopy(self.train[0]))
        refs=json.loads(self.audits[0]['source_refs_json']);refs.append({'path':'data/splits/train_items.csv','record':3})
        self.audits[0]['source_refs_json']=p.canonical(refs)
        m=self.run_preparation()
        self.assertEqual(m['counts']['train']['rows'],1)
        trace=self.outputs('trace.csv');self.assertEqual(sum('duplicate_text_group_held' in r['reasons'] for r in trace),2)
        self.assertEqual(len({r['item_id'] for r in trace}),len(trace))

    def test_changed_content_or_stale_review_identity_stops(self):
        self.train[0]['sentence']='שונה א״ב היום.'
        with self.assertRaisesRegex(ValueError,'Source content differs'):self.run_preparation()

    def test_source_metadata_not_silently_replaced_by_audit(self):
        self.train[0]['label_origin']='different origin'
        m=self.run_preparation()
        self.assertEqual(m['counts']['train']['rows'],1)
        held=next(r for r in self.outputs('trace.csv') if r['raw_item_id']=='legacy-collision' and r['label_origin']=='different origin')
        self.assertIn('audit_source_metadata_mismatch',held['reasons'])
        self.assertEqual(json.loads(held['source_metadata_json'])['label_origin'],'different origin')

    def test_singletons_are_separate_from_pair_loss(self):
        self.dev.update(candidates=self.dev['gold_expansion'],n_candidates='1')
        a=audit(self.dev,'aggregate_only',natural=True);a['review_evidence_json']=p.canonical([evidence(self.dev)])
        self.audits[-1]=a
        m=self.run_preparation()
        self.assertEqual(m['counts']['dev']['rows'],0)
        self.assertEqual(m['counts']['dev_singletons']['rows'],1)
        self.assertEqual(self.outputs('dev_singletons.csv')[0]['gold_expansion'],self.dev['gold_expansion'])

    def test_unknown_membership_blocks_export(self):
        self.audits[0]['historical_roles']='unclear'
        with self.assertRaisesRegex(ValueError,'lack unambiguous'):self.run_preparation()

    def test_missing_document_and_ai_authorship_preserved(self):
        self.train[0].update(page_title='',source='declared-historical-AI',category='manual',label_origin='ai_authored')
        self.audits[0]=audit(self.train[0])
        self.run_preparation()
        row=next(r for r in self.outputs('trace.csv') if r['construction']=='authored')
        self.assertIn('missing_document_key',row['reasons']);self.assertEqual(row['label_origin'],'ai_authored')

    def test_symlink_output_cannot_overwrite_source(self):
        self.run_preparation()
        out=self.root/'outputs'/'train.csv'
        out.unlink();out.symlink_to(self.root/'train.csv')
        before=(self.root/'train.csv').read_bytes()
        with self.assertRaisesRegex(ValueError,'Symlink output file'):self.run_preparation()
        self.assertEqual((self.root/'train.csv').read_bytes(),before)

    def test_malformed_reserved_metadata_fails_without_writing(self):
        a=audit(source(acronym='ז״ח'),'test');a['source_refs_json']='malformed'
        self.audits.append(a)
        with self.assertRaisesRegex(ValueError,'reserved blockers may be incomplete'):self.run_preparation()
        self.assertFalse((self.root/'outputs').exists())

    def test_explicit_forbidden_input_fails_before_any_read(self):
        with patch.object(p,'file_hash',side_effect=AssertionError('No file may be read')):
            with self.assertRaisesRegex(ValueError,'Direct test/aggregate'):
                p.prepare_inputs(train_path=self.root/'test_items.csv',
                    historical_dev_path=self.root/'dev.csv',audit_path=self.root/'audit.csv',
                    decisions_path=self.root/'decisions.csv',policy_path=self.root/'policy.json',
                    output_dir=self.root/'outputs')


if __name__ == '__main__':
    unittest.main()
