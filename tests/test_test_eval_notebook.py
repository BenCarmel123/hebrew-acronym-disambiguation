"""Offline notebook contracts; no credentials, research data or model calls."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/run_test_eval_colab.ipynb"


class TestEvaluationNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = ["".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"]

    def cell_containing(self, marker):
        return next(source for source in self.cells if marker in source)

    def test_python_cells_compile_and_outputs_are_empty(self):
        for index, source in enumerate(self.cells):
            with self.subTest(cell=index):
                compile(source, f"colab-cell-{index}", "exec")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_setup_requires_commit_and_archive_hash_without_import_hacks(self):
        source = self.cells[0]
        self.assertIn('r"[0-9a-f]{40}"', source)
        self.assertIn('r"[0-9a-f]{64}"', source)
        self.assertIn('"get-tar-commit-id"', source)
        self.assertIn('archive_revision != CODE_REVISION', source)
        self.assertIn('hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256', source)
        self.assertIn('"pip", "install", "-q", str(REPO_DIR)', source)
        self.assertNotIn('"-e"', source)
        all_source = "\n".join(self.cells)
        for prohibited in ('sys.path', 'sys.modules', 'spec_from_file_location', 'raw.githubusercontent.com'):
            self.assertNotIn(prohibited, all_source)
        # A portable stable source path keeps the runner identity resumable.
        self.assertIn('Path("/content/hebrew-eval-" + CODE_REVISION)', source)

    def test_project_pins_are_required_but_global_pip_check_is_advisory(self):
        source = self.cells[0]
        self.assertIn('importlib.metadata.version(package_name) != required_version', source)
        self.assertIn('raise RuntimeError(f"Required package version mismatch:', source)
        tree = ast.parse(source)
        checks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'run'
                  and node.args and isinstance(node.args[0], ast.List)
                  and any(isinstance(value, ast.Constant) and value.value == 'check'
                          for value in node.args[0].elts)]
        self.assertEqual(len(checks), 1)
        self.assertFalse(any(keyword.arg == 'check' and isinstance(keyword.value, ast.Constant)
                             and keyword.value.value for keyword in checks[0].keywords))

    def test_secrets_use_userdata_and_outputs_are_persistent(self):
        source = self.cell_containing('drive.mount(')
        self.assertIn('userdata.get(secret_name)', source)
        self.assertNotIn('print(secret_value)', source)
        self.assertIn('/content/drive/MyDrive/NLP/evaluation-runs', source)
        self.assertIn('raise RuntimeError', source)
        self.assertIn('from None', source)

    def test_full_test_cannot_call_provider_before_gate(self):
        source = self.cell_containing('test_manifest = prepare_evaluation(')
        for passed, cost in ((False, 1.0), (True, 100.01)):
            prepare, run = Mock(), Mock()
            namespace = {'PILOT_PASSED': passed, 'projected_total_ils': cost,
                         'BUDGET_ILS': 100.0, 'prepare_evaluation': prepare,
                         'run_evaluation': run}
            with self.subTest(passed=passed, cost=cost), self.assertRaises(RuntimeError):
                exec(compile(source, '<full-test>', 'exec'), namespace)
            prepare.assert_not_called()
            run.assert_not_called()

    def test_pilot_gate_requires_every_model_completion_and_identity(self):
        source = self.cell_containing('PILOT_INSPECTED = False')
        gate = source[source.index('PILOT_PASSED ='):]
        valid = {'n_items': 10, 'n_records': 80, 'n_completed': 80,
                 'n_pending': 0, 'n_ambiguous': 0,
                 'records': [{'response_metadata': {'identity_status': 'verified'}}]}
        for field, value in (('n_completed', 79), ('n_pending', 1), ('n_ambiguous', 1),
                             ('n_items', 9), ('n_records', 60),
                             ('records', [{'response_metadata': {'identity_status': 'unverified'}}])):
            namespace = {'pilot_summary': dict(valid, **{field: value}),
                         'PILOT_INSPECTED': True, 'projected_total_ils': 5, 'BUDGET_ILS': 100}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                exec(compile(gate, '<pilot-gate>', 'exec'), namespace)
        namespace = {'pilot_summary': valid, 'PILOT_INSPECTED': False,
                     'projected_total_ils': 5, 'BUDGET_ILS': 100}
        with self.assertRaises(RuntimeError):
            exec(compile(gate, '<pilot-gate>', 'exec'), namespace)
        namespace['PILOT_INSPECTED'] = True
        exec(compile(gate, '<pilot-gate>', 'exec'), namespace)
        self.assertTrue(namespace['PILOT_PASSED'])

    def test_notebook_settings_pass_installed_provider_validation(self):
        from hebrew_acronyms.test_evaluation import _validate_system
        source = self.cell_containing('SYSTEMS = [')
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "SYSTEMS" for t in node.targets))
        namespace = {"QWEN_MODEL": "qwen2.5:7b", "QWEN_IDENTITY": {"digest": "fixture-digest"},
                     "OLLAMA_URL": "http://localhost:11434",
                     "QWEN_OPTIONS": {"temperature": 0, "seed": 42, "num_predict": 512}}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<settings>", "exec"), namespace)
        for system in namespace['SYSTEMS']:
            with self.subTest(system=system['name']):
                _validate_system(system, fixture=False)

    def test_exact_approved_models_and_two_distinct_cohorts(self):
        source = "\n".join(self.cells)
        for model in ('qwen2.5:7b', 'gemini-3.8-flash', 'gpt-4.1-mini-2025-04-14', 'claude-haiku-5-5'):
            self.assertIn(model, source)
        self.assertIn('cohort="dev_pilot"', source)
        self.assertIn('cohort="full_test"', source)
        self.assertIn('max_calls=160', source)
        self.assertIn('max_calls=6320', source)
        self.assertIn('"num_predict": 512', source)
        self.assertIn('"seed": 42', source)


if __name__ == '__main__':
    unittest.main()
