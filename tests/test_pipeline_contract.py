"""E1-I integration checks: invented pair inputs and fully mocked pipeline arms."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hebrew_acronyms.pipelines import check_environment, run_all
from hebrew_acronyms.models.dictabert_cross_encoder import eval as cross_eval
from hebrew_acronyms.models.dictabert_cross_encoder import model as cross_model

ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ('connect', 'connect_ex', 'sendto'):
            self.stack.enter_context(patch.object(socket.socket, name,
                                    side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch('socket.getaddrinfo', side_effect=AssertionError('network forbidden')))
        self.qwen, self.gemini = Mock(), Mock()
        self.stack.enter_context(patch.dict(sys.modules, {
            'hebrew_acronyms.models.qwen.eval': SimpleNamespace(ollama_generate=self.qwen),
            'hebrew_acronyms.models.gemini.eval': SimpleNamespace(gemini_generate=self.gemini),
        }))

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

    def assert_checkpoint_rejected_without_effects(self, invoke):
        with ExitStack() as stack:
            calls = [stack.enter_context(patch.object(run_all, name,
                     side_effect=AssertionError(f'Unexpected call: {name}')))
                     for name in ('load_rows', 'load_signals', 'evaluate_baselines',
                                  'build_model', 'evaluate_dictabert_zeroshot', 'evaluate_llm')]
            calls += [stack.enter_context(patch.object(cross_model, 'load_finetuned')),
                      stack.enter_context(patch.object(cross_eval, 'evaluate')),
                      stack.enter_context(patch.object(run_all.torch.cuda, 'is_available')),
                      stack.enter_context(patch('builtins.open', side_effect=AssertionError('file I/O forbidden'))),
                      stack.enter_context(patch.object(Path, 'open', side_effect=AssertionError('file I/O forbidden')))]
            with self.assertRaisesRegex(ValueError, 'benchmark metric aggregation.*pending') as error:
                invoke()
            self.assertIn('hebrew_acronyms.models.dictabert_cross_encoder.eval', str(error.exception))
            for call in calls + [self.qwen, self.gemini]:
                call.assert_not_called()

    def test_checkpoint_rejected_before_inputs_models_or_services(self):
        for checkpoint in ('fixture.pt', '', Path('missing-checkpoint.pt')):
            for skip_llm in (True, False):
                with self.subTest(checkpoint=checkpoint, skip_llm=skip_llm):
                    self.assert_checkpoint_rejected_without_effects(
                        lambda: run_all.run('unread-items.csv', 'unread-candidates.csv', checkpoint, skip_llm))

    def test_cli_checkpoint_rejected_before_reading_or_writing(self):
        with patch.object(sys, 'argv', ['run_all', '--checkpoint', 'fixture.pt',
                                       '--items', 'unread.csv', '--out', 'untouched.md']):
            self.assert_checkpoint_rejected_without_effects(run_all.main)

    def test_no_checkpoint_preserves_results_and_optional_llm_dispatch(self):
        for skip_llm in (True, False):
            with self.subTest(skip_llm=skip_llm), ExitStack() as stack:
                rows, ranks, mined = [dict(item_id='invented')], object(), object()
                tok, model = object(), Mock()
                model.to.return_value = model
                readers = stack.enter_context(patch.object(run_all, 'load_rows', return_value=rows))
                signals = stack.enter_context(patch.object(run_all, 'load_signals', return_value=(ranks, mined)))
                baseline = stack.enter_context(patch.object(run_all, 'evaluate_baselines',
                    return_value=dict(random=0.1, most_frequent=0.2, most_mined=0.3, oracle=1.0, n_items=1)))
                stack.enter_context(patch.object(run_all.torch.cuda, 'is_available', return_value=False))
                builder = stack.enter_context(patch.object(run_all, 'build_model', return_value=(tok, model)))
                similarity = stack.enter_context(patch.object(run_all, 'evaluate_dictabert_zeroshot',
                                                              return_value=dict(accuracy=0.4, n_items=1)))
                llm = stack.enter_context(patch.object(run_all, 'evaluate_llm',
                             return_value=dict(accuracy=0.5, invalid_rate=0.0, n_items=1)))
                loader = stack.enter_context(patch.object(cross_model, 'load_finetuned'))
                with patch('builtins.open', side_effect=AssertionError('unmocked file access')):
                    result = run_all.run('invented-items.csv', 'invented-candidates.csv', None, skip_llm)
                expected = [dict(arm=name, accuracy=score, invalid_rate=None, n_items=1)
                            for name, score in [('random', 0.1), ('most_frequent', 0.2),
                                                ('most_mined', 0.3), ('oracle', 1.0),
                                                ('dictabert (untrained)', 0.4)]]
                expected.append(dict(arm='dictabertX (fine-tuned)', accuracy=None, invalid_rate=None, n_items=None))
                if not skip_llm:
                    expected += [dict(arm=f'{name} ({mode})', accuracy=0.5, invalid_rate=0.0, n_items=1)
                                 for name in ('qwen', 'gemini') for mode in ('generate', 'select')]
                    self.assertEqual(llm.call_count, 4)
                    self.assertEqual([(c.kwargs['generate_fn'], c.kwargs['mode']) for c in llm.call_args_list],
                                     [(fn, mode) for fn in (self.qwen, self.gemini) for mode in ('generate', 'select')])
                else:
                    llm.assert_not_called()
                self.assertEqual(result, expected)
                readers.assert_called_once_with('invented-items.csv')
                signals.assert_called_once_with('invented-candidates.csv')
                baseline.assert_called_once_with(rows, ranks, mined)
                builder.assert_called_once_with(run_all.DICTABERT_MODEL_ID)
                model.to.assert_called_once_with('cpu')
                model.eval.assert_called_once_with()
                similarity.assert_called_once_with(rows, tok, model, 'cpu')
                loader.assert_not_called()
                self.qwen.assert_not_called()
                self.gemini.assert_not_called()

    def test_installed_package_imports_in_fresh_isolated_process(self):
        code = """
from pathlib import Path
import hebrew_acronyms.pipelines.run_all as runner
from hebrew_acronyms.pipelines.check_environment import check_pairs
print(Path(runner.__file__).resolve())
assert len(check_pairs()) == 4
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / 'src/hebrew_acronyms/pipelines/run_all.py'), result.stdout)
        self.assertIn('Pairs: PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
