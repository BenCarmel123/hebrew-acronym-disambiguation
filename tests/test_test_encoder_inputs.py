"""Offline checks for explicitly selected legacy target mapping."""
import csv
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.test_encoder_inputs import qualify_test, write_qualified
from hebrew_acronyms.test_encoder_inference import validate_test_inputs


class TestEncoderInputs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source.csv'
        self.rows=[{'item_id':str(i),'acronym':'אב״ג','sentence':'באב"ג ואחר כך אב״ג',
                   'gold_expansion':'אחד','candidates':'אחד|שניים'} for i in range(395)]
        self.write(self.source,self.rows)

    def write(self,path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_default_does_not_guess_but_explicit_legacy_policy_is_deterministic(self):
        _,pending=qualify_test(self.source)
        self.assertEqual(len(pending),395)
        rows,pending=qualify_test(self.source,policy='legacy_first_occurrence')
        self.assertEqual(pending,[])
        self.assertEqual((rows[0]['span_start'],rows[0]['span_end'],rows[0]['target_raw']),(1,5,'אב"ג'))
        self.assertEqual(rows[0]['span_basis'],'legacy_first_occurrence_not_human_validated')
        for original,derived in zip(self.rows,rows):
            self.assertTrue(all(derived[k]==v for k,v in original.items()))

    def test_gold_and_candidate_changes_do_not_change_spans(self):
        first,_=qualify_test(self.source,policy='legacy_first_occurrence')
        self.rows[0].update(gold_expansion='שניים',candidates='שניים|אחד')
        self.write(self.source,self.rows)
        second,_=qualify_test(self.source,policy='legacy_first_occurrence')
        self.assertEqual([r['span_start'] for r in first],[r['span_start'] for r in second])

    def test_validation_requires_original_fields_and_no_overwrite(self):
        out=self.root/'derived.csv'
        rows=write_qualified(self.source,out,policy='legacy_first_occurrence')
        self.assertEqual(len(validate_test_inputs(self.source,out)),395)
        with self.assertRaises(FileExistsError):write_qualified(self.source,out,policy='legacy_first_occurrence')
        rows[0]['sentence']='changed';self.write(out,rows)
        with self.assertRaisesRegex(ValueError,'changed original'):validate_test_inputs(self.source,out)

    def test_policy_cannot_be_implicit_or_combined_with_human_decisions(self):
        with self.assertRaises(ValueError):qualify_test(self.source,policy='arbitrary')
        with self.assertRaises(ValueError):qualify_test(self.source,{'0':{}},policy='legacy_first_occurrence')
