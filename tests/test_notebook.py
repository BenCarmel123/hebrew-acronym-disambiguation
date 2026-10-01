"""Structural checks for the training code appendix, without executing a model."""
import ast
import json
from pathlib import Path
import unittest
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class NotebookStructureTests(unittest.TestCase):
    def test_source_imports_do_not_write_or_use_network(self):
        code = """
import os, sys

def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('Network during import')
    if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise RuntimeError('File write during import')

sys.addaudithook(guard)
import hebrew_acronyms.models.dictabert_similarity.model
import hebrew_acronyms.models.dictabert_cross_encoder.model
import hebrew_acronyms.models.dictabert_cross_encoder.encoding
import hebrew_acronyms.models.dictabert_cross_encoder.training
import hebrew_acronyms.models.dictabert_cross_encoder.workflow
import hebrew_acronyms.models.dictabert_cross_encoder.eval
import hebrew_acronyms.pipelines.check_environment
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_cells_run_on_tiny_fixtures_without_cache(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.run_notebook", "--tiny-sanity"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-2000:])
        self.assertIn("Notebook: PASS (sanity, all cells, offline, invented inputs)", result.stdout)
        notebook = json.loads((ROOT / "notebooks/train_dictabert.ipynb").read_text())
        self.assertEqual(result.stdout.count("Running notebook cell"),
                         sum(c["cell_type"] == "code" for c in notebook["cells"]))

    def test_missing_snapshot_reports_not_run_without_download(self):
        code = """
from unittest.mock import patch
from tests.run_notebook import main
with patch('hebrew_acronyms.models.dictabert_cross_encoder.workflow.find_snapshot', return_value=None):
    raise SystemExit(main())
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Notebook: NOT RUN", result.stdout)

    def test_thin_notebook_and_single_definitions(self):
        notebook = json.loads((ROOT / "notebooks/train_dictabert.ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(cells):
            tree = ast.parse("".join(cell["source"]))
            forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For,
                         ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
            self.assertFalse(any(isinstance(node, forbidden) for node in ast.walk(tree)))
            if i:
                self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                                     for node in ast.walk(tree)))
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
        setup = "".join(cells[0]["source"])
        self.assertIn('MODE = "smoke"', setup)
        self.assertNotIn("manual_seed", setup)
        definitions = []
        for path in (ROOT / "src/hebrew_acronyms/models").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ClassDef) and node.name == "CrossEncoder":
                    definitions.append(path)
        self.assertEqual(definitions, [ROOT / "src/hebrew_acronyms/models/dictabert_cross_encoder/model.py"])
        self.assertIn("from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs",
                      (ROOT / "src/hebrew_acronyms/models/dictabert_cross_encoder/training.py").read_text())
        self.assertIn("encode_pairs", (ROOT / "src/hebrew_acronyms/models/dictabert_cross_encoder/eval.py").read_text())


if __name__ == "__main__":
    unittest.main()
