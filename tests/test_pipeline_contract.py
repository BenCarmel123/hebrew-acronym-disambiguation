"""Local environment fixture and installed-package checks."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from hebrew_acronyms.pipelines import check_environment
ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_environment_pairs_have_explicit_exact_targets(self):
        with patch.object(check_environment, 'check_model') as model, redirect_stdout(io.StringIO()):
            pairs = check_environment.check_pairs()
        model.assert_not_called()
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs.identity['item_ids'], ['environment-1', 'environment-2'])
        for row in pairs.rows:
            self.assertEqual(row['sentence'][row['span_start']:row['span_end']], row['target_raw'])
        self.assertEqual(pairs.rows[1]['target_raw'], 'ב"ד')
        self.assertEqual([p[0] for p in pairs], [
            'היום נערך מפגש של [ACR]ב״ד[/ACR] בכיתה.',
            'היום נערך מפגש של [ACR]ב״ד[/ACR] בכיתה.',
            'מחר נציג [ACR]ב"ד[/ACR] קטן.',
            'מחר נציג [ACR]ב"ד[/ACR] קטן.',
        ])
        self.assertEqual([p[2] for p in pairs], [1, 0, 0, 1])

    def test_installed_package_imports_in_fresh_isolated_process(self):
        code = """
from pathlib import Path
import hebrew_acronyms.models.dictabert_cross_encoder.workflow as workflow
from hebrew_acronyms.pipelines.check_environment import check_pairs
print(Path(workflow.__file__).resolve())
assert len(check_pairs()) == 4
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / 'src/hebrew_acronyms/models/dictabert_cross_encoder/workflow.py'), result.stdout)
        self.assertIn('Pairs: PASS', result.stdout)
