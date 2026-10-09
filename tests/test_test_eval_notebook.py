"""Offline notebook contracts; no credentials, research data or model calls."""
import ast
import contextlib
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
        self.assertIn('missing_secret', source)
        self.assertIn('PRIOR_SPEND_ILS = None', source)
        self.assertIn('SELECTED_SYSTEMS', source)

    def test_settings_validate_and_qwen_can_be_unselected(self):
        from hebrew_acronyms.test_evaluation import _validate_system
        source = next(source for source in self.cells if source.startswith('SYSTEMS = ['))
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'SYSTEMS' for t in node.targets))
        for qwen in (None, {'digest': 'fixture-digest'}):
            state = {'QWEN_MODEL': 'qwen2.5:7b', 'QWEN_IDENTITIES': {'qwen': qwen or {}, 'qwen14': qwen or {}}, 'QWEN14_MODEL': 'qwen2.5:14b',
                     'OLLAMA_URL': 'http://localhost:11434',
                     'QWEN_OPTIONS': {'temperature': 0, 'seed': 42, 'num_predict': 512}}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<settings>', 'exec'), state)
            for system in state['SYSTEMS']:
                if system['provider'] != 'qwen' or qwen:
                    _validate_system(system, fixture=False)
        source = self.cell_containing('QWEN_MODEL =')
        self.assertIn('if qwen_name not in SELECTED_SYSTEMS:', source)

    def test_qwen14_preload_has_no_prompt_and_rejects_generated_output(self):
        source = self.cell_containing('preload_started =')
        tree = ast.parse(source)
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(child, ast.Name) and child.id == 'preload_started'
                             for child in ast.walk(node))
                     and isinstance(node.test, ast.Compare))
        for payload, valid in (({'done': True, 'response': ''}, True),
                               ({'done': False}, False),
                               ({'done': True, 'response': 'unexpected'}, False)):
            request = Mock()
            request.post.return_value.json.return_value = payload
            with tempfile.TemporaryDirectory() as directory:
                state = {'qwen_name': 'qwen14', 'qwen_model': 'qwen2.5:14b',
                         'OLLAMA_URL': 'http://localhost:11434', 'QWEN_OPTIONS': {'seed': 42},
                         'inspection': {'digest': 'fixture'}, 'OUTPUT_ROOT': Path(directory),
                         'time': Mock(perf_counter=Mock(side_effect=[0, 73]), time_ns=Mock(return_value=1)),
                         'requests': request, 'json': json}
                code = compile(ast.Module(body=[block], type_ignores=[]), '<preload>', 'exec')
                if valid:
                    exec(code, state)
                    self.assertEqual(len(list(Path(directory).glob('*preload*.json'))), 1)
                else:
                    with self.assertRaises(RuntimeError):
                        exec(code, state)
                body = request.post.call_args.kwargs['json']
                self.assertNotIn('prompt', body)
                self.assertEqual(body['model'], 'qwen2.5:14b')
                request.post.assert_called_once()

    def test_selected_unavailable_provider_does_not_block_others(self):
        source = next(source for source in self.cells if source.startswith('SYSTEMS = ['))
        state = {'QWEN_MODEL': 'qwen2.5:7b', 'QWEN_IDENTITIES': {}, 'QWEN14_MODEL': 'qwen2.5:14b',
                 'OLLAMA_URL': 'http://localhost:11434',
                 'QWEN_OPTIONS': {'temperature': 0, 'seed': 42, 'num_predict': 512},
                 'SELECTED_SYSTEMS': ['openai', 'anthropic'],
                 'SYSTEM_NAMES': ('qwen', 'gemini', 'openai', 'anthropic'),
                 'AVAILABILITY': {'qwen': {'status': 'not_selected'}, 'gemini': {'status': 'not_selected'},
                                  'openai': {'status': 'credentials_loaded'}, 'anthropic': {'status': 'unavailable'}},
                 'inspect_openai_model': Mock(return_value={'status': 'available'}),
                 'inspect_anthropic_model': Mock(), 'json': json,
                 'time': Mock(time_ns=Mock(return_value=1))}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            state.update(OUTPUT_ROOT=Path(directory), REPO_DIR=Path(directory))
            exec(compile(source, '<availability>', 'exec'), state)
        state['inspect_anthropic_model'].assert_not_called()
        self.assertEqual([system['name'] for system in state['available_selected_systems']()], ['openai'])
        # Selection and credential readiness can change after this cell ran.
        state['SELECTED_SYSTEMS'] = ['anthropic']
        self.assertEqual(state['available_selected_systems'](), [])
        state['SELECTED_SYSTEMS'] = ['openai']
        state['AVAILABILITY']['openai']['status'] = 'credentials_loaded'
        self.assertEqual(state['available_selected_systems'](), [])

    def test_full_rechecks_current_session_before_any_cached_authorization(self):
        source = self.cell_containing('INSPECTED_PILOTS = {}')
        state = {'current_session': Mock(side_effect=ValueError('Session identity mismatch')),
                 'available_selected_systems': Mock(), 'run_system_full': Mock()}
        with self.assertRaisesRegex(ValueError, 'Session identity mismatch'):
            exec(compile(source, '<full>', 'exec'), state)
        state['available_selected_systems'].assert_not_called()
        state['run_system_full'].assert_not_called()

    def test_full_cell_only_calls_systems_with_explicit_inspection_identity(self):
        source = self.cell_containing('INSPECTED_PILOTS = {}')
        state = {'available_selected_systems': lambda: [{'name': 'openai'}], 'current_session': Mock(), 'OUTPUT_ROOT': Path('/invented'),
                 'run_system_full': Mock(), 'session_summary': Mock(return_value={}), 'json': json}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, '<full>', 'exec'), state)
        state['run_system_full'].assert_not_called()
        state['current_session'].assert_called_once()
        self.assertIn('inspected_pilot_identity=INSPECTED_PILOTS[name]', source)

    def test_two_tasks_and_budget_are_delegated_to_installed_shared_workflow(self):
        source = '\n'.join(self.cells)
        self.assertIn('run_system_pilot(', source)
        self.assertIn('review_system_pilot(', source)
        self.assertIn('run_system_full(', source)
        self.assertIn('prepare_session(', source)
        self.assertIn('rates=RATES, reserves=RESERVE_PER_CALL_USD', source)
        self.assertNotIn('PILOT_PASSED', source)
        for model in ('qwen2.5:7b', 'gemini-3.8-flash', 'gpt-4.1-mini-2025-04-14', 'claude-haiku-5-5', 'grok-4.7', 'qwen2.5:14b'):
            self.assertIn(model, source)


if __name__ == '__main__':
    unittest.main()
