"""Presentation retains measurement layers and literal evidence, without mutating rows."""
from copy import deepcopy
import unittest
from hebrew_acronyms.saved_analysis import comparison_table, dictalm_table, failure_examples


class SavedAnalysisDisplayTests(unittest.TestCase):
    def test_comparison_keeps_counts_and_omits_generation_for_encoder(self):
        metrics = [dict(system=s, task=t, n_correct=1, n_items=3, accuracy=1/3)
                   for s in ('dictabert', 'qwen', 'qwen14', 'openai', 'anthropic', 'xai', 'gemini')
                   for t in ('generate', 'select') if s != 'dictabert' or t == 'select']
        before = deepcopy(metrics)
        similarity = {'metric': dict(n_correct=2, n_items=3, accuracy=2/3)}
        html = comparison_table(metrics, similarity)
        self.assertIn('1/3 (33.3%)', html)
        self.assertIn('2/3 (66.7%)', html)
        self.assertEqual(html.count('Not applicable'), 2)
        self.assertIn('Generation: substring match', html)
        self.assertIn('Selection: reference match', html)
        self.assertNotIn('<details>', html)
        self.assertEqual(metrics, before)

    def test_dictalm_has_four_separate_layers_including_zero(self):
        names = ('automatic_generation', 'ben_generation', 'strict_selection',
                 'diagnostic_leading_label_selection')
        rows = [dict(metric=n, n_correct=c, n_items=381, accuracy=c/381)
                for n,c in zip(names, (125,193,0,243))]
        before = deepcopy(rows)
        html = dictalm_table(rows)
        for text in ('125/381','193/381','0/381','243/381','0.0%', 'author annotation', 'diagnostic'):
            self.assertIn(text, html)
        self.assertEqual(rows,before)

    def test_failure_preview_is_literal_escaped_and_explicitly_truncated(self):
        rows = [dict(system='qwen', task='generate', item_id='invented',
                     response='<script>אב malformed\n' + '字' * 300)] * 3
        before = deepcopy(rows)
        html = failure_examples(rows, limit=2, preview_length=40)
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;אב malformed\n',html)
        self.assertIn('First 40 characters of',html)
        self.assertEqual(html.count('<blockquote'),2)
        self.assertEqual(rows,before)

if __name__ == '__main__':
    unittest.main()
