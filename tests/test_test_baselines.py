import csv
import tempfile
import unittest
from pathlib import Path

from hebrew_acronyms.test_baselines import run_test_baselines


class BaselineArtifacts(unittest.TestCase):
    def test_preserves_ties_coverage_and_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            items = root / 'items.csv'
            signals = root / 'signals.csv'
            items.write_text('item_id,acronym,candidates,gold_expansion\none,X,wrong|right,right\ntwo,X,right|wrong,right\n')
            signals.write_text('acronym,expansion,rank,mined_items\nX,wrong,1,2\nX,right,2,8\n')
            args = (items, signals, root / 'saved')
            result = run_test_baselines(*args, code_revision='a' * 40)
            self.assertEqual(result['n_items'], 2)
            self.assertEqual(result['summary']['random'], .5)
            self.assertEqual(result['summary']['most_frequent'], 0)
            self.assertEqual(result['summary']['most_mined'], 1)
            self.assertEqual(result, run_test_baselines(*args, code_revision='a' * 40))
            signals.write_text('acronym,expansion,rank,mined_items\n')
            with self.assertRaises(ValueError):
                run_test_baselines(*args, code_revision='a' * 40)

    def test_never_silently_skips_invalid_item(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'items.csv').write_text('item_id,acronym,candidates,gold_expansion\none,X,only,only\n')
            (root / 'signals.csv').write_text('acronym,expansion,rank,mined_items\n')
            with self.assertRaises(ValueError):
                run_test_baselines(root / 'items.csv', root / 'signals.csv', root / 'saved', code_revision='a' * 40)


if __name__ == '__main__':
    unittest.main()
