"""Offline notebook contracts; no credentials, research data or model calls."""
import ast
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
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

    def full_fixture(self, directory):
        folder = Path(directory)
        pilot = folder / "pilot"
        pilot.mkdir()
        dev = folder / "dev.csv"
        dev.write_text("invented fixed data")
        inspection = {"model": "qwen2.5:7b", "digest": "fixture-digest",
                      "server_version": "fixture-version", "options": {"seed": 42},
                      "template": "fixture-template", "model_parameters": "stop fixture",
                      "details": {"quantization_level": "Q4_K_M"}, "model_info": {"architecture": "qwen2"},
                      "installed_model": {"modified_at": "old-session-time"}}
        source = self.cell_containing('QWEN_STABLE_FIELDS =')
        assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in
                               {'QWEN_STABLE_FIELDS', 'QWEN_FROZEN_IDENTITY'} for target in node.targets)]
        state = {"QWEN_IDENTITY": deepcopy(inspection)}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<qwen-identity>', 'exec'), state)
        rates = {"fixture": {"input": .4, "output": 1.6}}
        systems = [{"name": "fixture", "model": "exact-fixture", "settings": {"max_output_tokens": 512}}]
        metadata = {"qwen_inspection": state['QWEN_FROZEN_IDENTITY'],
                    "rates_usd_per_million": rates, "conversion_allowance_ils_per_usd": 4.0,
                    "prior_spend_ils": 0.0}
        identity = {"cohort": "dev_pilot", "systems": systems, "code_revision": "a" * 40,
                    "code_sha256": {"test_evaluation.py": "fixture-source-hash"},
                    "source_sha256": hashlib.sha256(dev.read_bytes()).hexdigest(),
                    "seed": 42, "max_attempts": 2, "max_calls": 160,
                    "rates_usd_per_million": rates, "reserve_per_call_usd": {"fixture": .02},
                    "metadata": metadata, "budget_usd": 25.0}
        (pilot / "manifest.json").write_text(json.dumps({"identity": identity}))
        summary = {"n_items": 10, "n_records": 80, "n_completed": 80, "n_pending": 0,
                   "n_ambiguous": 0, "n_identity_unverified": 0, "charged_or_reserved_usd": .2,
                   "n_calls": 80}
        state.update(json=json, hashlib=hashlib, PILOT_DIR=pilot, DEV_PATH=dev,
                     TEST_PATH=folder / "test.csv", TEST_DIR=folder / "full",
                     CODE_REVISION="a" * 40, SYSTEMS=deepcopy(systems), RATES=deepcopy(rates),
                     BUDGET_ILS=100.0, PRIOR_SPEND_ILS=0.0, ILS_PER_USD_ALLOWANCE=4.0,
                     RESERVE_PER_CALL_USD={"fixture": .02}, PILOT_INSPECTED=True,
                     PILOT_PASSED=True, projected_total_ils=1.0,
                     pilot_summary={"charged_or_reserved_usd": 0.0},
                     summarize_evaluation=Mock(return_value=summary),
                     estimate_cost=Mock(return_value={"pilot_usd": .2, "projected_full_with_reserve_usd": 2.0}),
                     prepare_evaluation=Mock(return_value={"identity": {"code_sha256": {"test_evaluation.py": "fixture-source-hash"}}}),
                     run_evaluation=Mock(return_value=summary))
        return state

    def execute_full_cell(self, state):
        source = self.cell_containing('test_manifest = prepare_evaluation(')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, '<full-test>', 'exec'), state)

    def test_full_test_rejects_changed_settings_despite_stale_success_flag(self):
        for changed in ('system', 'code', 'rates', 'digest', 'template', 'source'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                state = self.full_fixture(directory)
                if changed == 'system':
                    state['SYSTEMS'][0]['settings']['max_output_tokens'] = 1024
                elif changed == 'code':
                    state['CODE_REVISION'] = 'b' * 40
                elif changed == 'rates':
                    state['RATES']['fixture']['output'] = .1
                elif changed in {'digest', 'template'}:
                    state['QWEN_IDENTITY'][changed] = 'changed'
                else:
                    state['DEV_PATH'].write_text('changed input')
                with self.assertRaisesRegex(RuntimeError, 'Pilot configuration changed'):
                    self.execute_full_cell(state)
                state['prepare_evaluation'].assert_not_called()
                state['run_evaluation'].assert_not_called()

    def test_full_test_reloads_coverage_cost_and_remaining_budget(self):
        for changed in ('incomplete', 'expensive', 'uninspected', 'complete'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                state = self.full_fixture(directory)
                if changed == 'incomplete':
                    state['summarize_evaluation'].return_value['n_completed'] = 79
                elif changed == 'expensive':
                    state['estimate_cost'].return_value['projected_full_with_reserve_usd'] = 30.0
                elif changed == 'uninspected':
                    state['PILOT_INSPECTED'] = False
                if changed == 'complete':
                    self.execute_full_cell(state)
                    state['summarize_evaluation'].assert_called_once_with(state['PILOT_DIR'])
                    self.assertEqual(state['prepare_evaluation'].call_args.kwargs['budget_usd'], 24.8)
                    state['run_evaluation'].assert_called_once()
                else:
                    with self.assertRaises(RuntimeError):
                        self.execute_full_cell(state)
                    state['prepare_evaluation'].assert_not_called()
                    state['run_evaluation'].assert_not_called()

    def test_changed_installed_source_blocks_calls_despite_same_commit_label(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.full_fixture(directory)
            state['prepare_evaluation'].return_value['identity']['code_sha256'] = {
                "test_evaluation.py": "changed-installed-source"}
            with self.assertRaisesRegex(RuntimeError, 'Installed package source differs'):
                self.execute_full_cell(state)
            state['prepare_evaluation'].assert_called_once()
            state['run_evaluation'].assert_not_called()

    def test_same_qwen_digest_with_new_pull_timestamp_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.full_fixture(directory)
            state['QWEN_IDENTITY']['installed_model']['modified_at'] = 'new-session-time'
            self.execute_full_cell(state)
            metadata = state['prepare_evaluation'].call_args.kwargs['metadata']
            self.assertNotIn('installed_model', metadata['qwen_inspection'])
            self.assertEqual(metadata['qwen_inspection']['digest'], 'fixture-digest')
            state['run_evaluation'].assert_called_once()

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
