"""Recommendation evidence remains human-provided; no silent rescoring."""
import copy
import unittest
from hebrew_acronyms.human_review_scoring_audit import enrich_summary, recommendation_rows, scoring_markdown


class ScoringAuditTests(unittest.TestCase):
    def test_masked_summary_never_enriched_with_scores_or_model_metadata(self):
        masked={'masked':True,'cases':[],'counts':{'judged_answers':0}}
        self.assertEqual(enrich_summary(masked,{'scoring_audit':{'secret':'hidden'}}),masked)
        self.assertEqual(scoring_markdown(masked),'')

    def test_tags_keep_evidence_even_when_human_and_score_agree(self):
        summary={'masked':False,'cases':[{'id':'fixture','answers':[
            {'answer_id':'original:a','label':'not_fits','auto_score':False,'tags':['gibberish']},
            {'answer_id':'original:b','label':'fits','auto_score':True,'tags':['spelling','punctuation']}]}]}
        original=copy.deepcopy(summary)
        enriched=enrich_summary(summary,{'scoring_audit':{'description':'fixture rule'}})
        proposals={p['id']:p for p in enriched['recommendations']}
        self.assertEqual(proposals['contradictory_or_extra_text']['evidence'][0]['answer_id'],'original:a')
        self.assertEqual(proposals['controlled_spelling']['evidence'][0]['label'],'fits')
        self.assertEqual(summary,original)
        self.assertIn('אין עדיין',proposals['human_approved_equivalents']['status'])

    def test_mismatch_alone_does_not_invent_mechanism_or_tag(self):
        summary={'cases':[{'id':'fixture','answers':[{'answer_id':'a','label':'fits','auto_score':False,'tags':[]}]}]}
        proposals=recommendation_rows(summary)
        self.assertTrue(all(not p['evidence'] for p in proposals))
        self.assertTrue(all('אין עדיין' in p['status'] for p in proposals))
        self.assertTrue(all(p['false_acceptance_risk'] for p in proposals))

    def test_export_explains_non_independent_rule_development(self):
        full=enrich_summary({'masked':False,'cases':[]},{'scoring_audit':{'description':'Verified fixture substring rule','saved_generation_answers_checked':4,'score_mismatches':[]}})
        text=scoring_markdown(full)
        self.assertIn('Verified fixture substring rule',text)
        self.assertIn('אינה אימות עצמאי',text)
        self.assertIn('לשני הכיוונים',text)

if __name__=='__main__':unittest.main()
