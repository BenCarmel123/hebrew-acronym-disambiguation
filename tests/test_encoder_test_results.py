"""Saved encoder loading checks with invented predictions, without model inference."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from hebrew_acronyms.encoder_test_results import load_encoder_test
from hebrew_acronyms.test_evaluation import _hash

class EncoderTestResults(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.test=self.root/'test.csv'
        with self.test.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['item_id','gold_expansion','candidates']);w.writeheader()
            w.writerows({'item_id':str(i),'gold_expansion':'a','candidates':'a|b'} for i in range(395))
        identity={'source_sha256':hashlib.sha256(self.test.read_bytes()).hexdigest(),
                  'weights_sha256':'c'*64,'item_ids':[str(i) for i in range(395)]}
        self.identity=_hash(identity)
        (self.root/'manifest.json').write_text(json.dumps({'run_id':'fixture','identity':identity,'identity_sha256':self.identity}))
        (self.root/'checkpoint-reconstruction.json').write_text(json.dumps({'weights_sha256':'c'*64}))
        self.records=[{'item_id':str(i),'run_id':'fixture','task':'select','status':'ok',
                       'selected_candidate':'a','correct':True} for i in range(395)]
        self.save()
    def save(self):
        p=self.root/'predictions.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in self.records))
        self.digest=hashlib.sha256(p.read_bytes()).hexdigest()
    def load(self):
        return load_encoder_test(self.root,self.test,expected_run_id='fixture',expected_identity=self.identity,
                                 expected_predictions_sha256=self.digest)
    def test_full_denominator_and_no_model_required(self):
        self.records[0].update(status='model_error',selected_candidate=None,correct=False);self.save()
        metric=self.load()['metric']
        self.assertEqual((metric['n_items'],metric['n_completed'],metric['n_correct']),(395,394,394))
    def test_changed_artifact_or_missing_id_is_rejected(self):
        (self.root/'predictions.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'identified saved'):self.load()
        self.records.pop();self.save()
        with self.assertRaisesRegex(ValueError,'ordered test IDs'):self.load()
    def test_saved_score_must_agree_with_gold(self):
        self.records[0]['selected_candidate']='b';self.save()
        with self.assertRaisesRegex(ValueError,'saved score'):self.load()
