"""Loading equivalence against accepted S3b sources; invented inputs and fake models only.

The pipeline's run/main functions are never called. Only its isolated model-loading
statements execute, and the remaining function is compared structurally to Git.
"""

import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import csv
import io
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model import baselines
from model.common import pairs
from model.dictabert import model as base_model
from tests.reference import reference_git

BASE = "f528183164dde019351ad5b00d3f60f354c69989"
ROOT = Path(__file__).resolve().parents[1]
EVAL = "model/dictabert/eval.py"
PIPELINE = "pipeline/run_all.py"


def source(path, baseline=False):
    if baseline:
        return reference_git(BASE, "show", f"{BASE}:{path}")
    return (ROOT / path).read_text(encoding="utf-8")


def definition(path, name, baseline=False):
    return next(node for node in ast.parse(source(path, baseline)).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "isolated-loading", "exec"), scope)


def loading_block(function):
    """Select from device choice through model.eval(), not pipeline evaluation."""
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "device")
    end = next(i for i in range(start, len(function.body))
               if ast.unparse(function.body[i]) == "model.eval()") + 1
    return function.body[start:end], function.body[:start] + function.body[end:]


class LoadingEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("connect", "connect_ex"):
            self.stack.enter_context(patch.object(
                socket.socket, name, side_effect=AssertionError("network forbidden")))
        self.stack.enter_context(patch(
            "socket.create_connection", side_effect=AssertionError("network forbidden")))
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_rng)
        self.addCleanup(torch.set_rng_state, self.torch_rng)

    def old_rows(self):
        scope = {"csv": csv}
        execute([definition("model/baselines.py", "load_rows", True)], scope)
        return scope["load_rows"]

    def test_csv_bom_order_quoted_newlines_and_permissive_shapes(self):
        self.assertIs(baselines.load_rows, pairs.load_rows)
        cases = [
            "", "acronym,sentence\r\n",
            'acronym,sentence\r\nב,שני\r\nא,"ראשון, עם פסיק\r\nושורה"\r\n',
            'acronym,sentence\nא,קצר,עודף\nב\n',
            'acronym,sentence\nא,"ציטוט שלא נסגר\n',
            'acronym,acronym\nא,ב\n',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invented.csv"
            for encoding in ("utf-8", "utf-8-sig"):
                for content in cases:
                    with self.subTest(encoding=encoding, content=content):
                        path.write_bytes(content.encode(encoding))
                        expected = self.old_rows()(path)
                        self.assertEqual(pairs.load_rows(path), expected)
                        self.assertEqual(baselines.load_rows(str(path)), expected)
                        self.assertEqual([list(row) for row in pairs.load_rows(path)],
                                         [list(row) for row in expected])

    def test_csv_read_errors_match(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.csv"
            invalid.write_bytes(b"acronym\n\xff")
            for path in (invalid, Path(directory) / "missing.csv", Path(directory)):
                errors = []
                for loader in (self.old_rows(), pairs.load_rows, baselines.load_rows):
                    try:
                        loader(path)
                    except (OSError, UnicodeError) as error:
                        errors.append((type(error), str(error)))
                self.assertEqual(len(errors), 3)
                self.assertEqual(errors, [errors[0]] * 3)

    def fake_loading(self, events, raw_calls, *, cuda=False, fail=None, training=True):
        """Factories consume RNG so reordering or extra calls cannot hide in a spy."""
        token = object()
        moved = SimpleNamespace(training=training)

        def event(name, *args):
            events.append((name, *args))
            if fail == name:
                raise RuntimeError(f"fixture {name} failure")

        def set_eval():
            event("eval")
            moved.training = False
            return moved

        moved.eval = set_eval

        def move(device):
            event("to", device)
            return moved

        encoder = SimpleNamespace(to=move)

        def factory(kind, result):
            def from_pretrained(model_id, **kwargs):
                raw_calls.append((kind, model_id, kwargs.copy()))
                # Installed Transformers uses None when revision is omitted;
                # Hugging Face Hub resolves None to its DEFAULT_REVISION (main).
                effective = {"revision": None, **kwargs}
                event(kind, model_id, effective, random.random(), torch.rand(()).item())
                return result
            return SimpleNamespace(from_pretrained=from_pretrained)

        def is_available():
            event("cuda_available")
            return cuda

        transformers = SimpleNamespace(AutoTokenizer=factory("tokenizer", token),
                                       AutoModel=factory("model", encoder))
        return transformers, SimpleNamespace(cuda=SimpleNamespace(is_available=is_available)), token, moved

    def run_isolated(self, path, baseline, *, device=None, cuda=False, fail=None,
                     training=True, input_bytes=b"acronym,sentence\na,fixture\n",
                     default_items=False):
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng)
        events, raw_calls = [], []
        transformers, fake_torch, token, moved = self.fake_loading(
            events, raw_calls, cuda=cuda, fail=fail, training=training)
        output, error = io.StringIO(), None
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "invented.csv"
            fixture.write_bytes(input_bytes)
            requested_path = "data/splits/dev_items.csv" if default_items else str(fixture)
            original_open = open

            def fixture_open(path, *args, **kwargs):
                self.assertEqual(str(path), requested_path)
                events.append(("read", "<fixture>", args, kwargs))
                return original_open(fixture, *args, **kwargs)

            def evaluate(rows, tok, model, target):
                self.assertIs(tok, token)
                self.assertIs(model, moved)
                self.assertFalse(model.training)
                events.append(("evaluate", rows, target))
                return {"n_items": len(rows), "accuracy": 0.25}

            scope = {"argparse": argparse, "csv": csv, "torch": fake_torch,
                     "AutoTokenizer": transformers.AutoTokenizer,
                     "AutoModel": transformers.AutoModel, "build_model": base_model.build_model,
                     "MODEL_ID": base_model.MODEL_ID, "DICTABERT_MODEL_ID": base_model.MODEL_ID,
                     "load_rows": pairs.load_rows, "evaluate": evaluate, "__doc__": "fixture"}
            with (patch.dict(sys.modules, {"transformers": transformers}),
                  patch("builtins.open", fixture_open), redirect_stdout(output)):
                try:
                    if path == EVAL:
                        argv = ["fixture"] if default_items else ["fixture", "--items", str(fixture)]
                        if device is not None:
                            argv += ["--device", device]
                        execute([definition(path, "main", baseline)], scope)
                        with patch.object(sys, "argv", argv):
                            scope["main"]()
                    else:
                        nodes, _ = loading_block(definition(path, "run", baseline))
                        execute(nodes, scope)
                        self.assertIs(scope["tok"], token)
                        self.assertIs(scope["model"], moved)
                        self.assertFalse(moved.training)
                except (RuntimeError, UnicodeError) as exc:
                    error = (type(exc), str(exc))
        return (events, output.getvalue().replace(str(fixture), "<fixture>"), error,
                random.getstate(), torch.get_rng_state().tolist()), raw_calls

    def compare_loading(self, path, **kwargs):
        old, old_raw = self.run_isolated(path, True, **kwargs)
        new, new_raw = self.run_isolated(path, False, **kwargs)
        self.assertEqual(new, old)
        self.assertTrue(all(call[2] == {} for call in old_raw))
        self.assertTrue(all(call[2] == {"revision": None} for call in new_raw))
        return new

    def test_eval_defaults_explicit_device_order_and_output(self):
        for cuda, device, training in ((False, None, True), (True, None, False),
                                      (True, "cpu", True), (False, "fixture-device", False)):
            with self.subTest(cuda=cuda, device=device, training=training):
                result = self.compare_loading(EVAL, cuda=cuda, device=device, training=training)
                names = [event[0] for event in result[0]]
                self.assertEqual(names, (["cuda_available"] if device is None else [])
                                 + ["tokenizer", "model", "to", "eval", "read", "evaluate"])
        self.compare_loading(EVAL, default_items=True)

    def test_eval_reads_bom_and_preserves_decode_error_after_loading(self):
        self.compare_loading(EVAL, input_bytes='\ufeffacronym,sentence\r\nא,"ב\r\nג"\r\n'.encode())
        result = self.compare_loading(EVAL, input_bytes=b"acronym\n\xff")
        self.assertIs(result[2][0], UnicodeDecodeError)
        self.assertEqual(result[0][-1][0], "read")

    def test_pipeline_loading_order_device_and_rng(self):
        for cuda in (False, True):
            result = self.compare_loading(PIPELINE, cuda=cuda)
            self.assertEqual([event[0] for event in result[0]],
                             ["cuda_available", "tokenizer", "model", "to", "eval"])

    def test_loading_failures_stop_at_same_stage(self):
        for path in (EVAL, PIPELINE):
            for fail in ("tokenizer", "model", "to", "eval"):
                with self.subTest(path=path, fail=fail):
                    result = self.compare_loading(path, fail=fail)
                    self.assertEqual(result[0][-1][0], fail)
                    self.assertEqual(result[2], (RuntimeError, f"fixture {fail} failure"))

    def test_pipeline_outside_loading_is_unchanged(self):
        old, new = [definition(PIPELINE, "run", baseline) for baseline in (True, False)]
        _, old.body = loading_block(old)
        _, new.body = loading_block(new)
        self.assertEqual(ast.dump(new), ast.dump(old))
        for path, names in ((PIPELINE, ("main", "to_markdown")),
                            (EVAL, ("embed", "evaluate")),
                            ("model/dictabertX/eval.py", ("evaluate", "main")),
                            ("model/baselines.py", ("load_signals", "evaluate", "main"))):
            for name in names:
                self.assertEqual(ast.dump(definition(path, name)),
                                 ast.dump(definition(path, name, True)))

    def test_cross_encoder_selects_highest_score_and_first_tie(self):
        from model.dictabertX import eval as cross_eval

        rows = [dict(sentence="דוגמה א״ב", acronym="א״ב", candidates="אלף|בית",
                     gold_expansion=gold) for gold in ("בית", "אלף")]
        scores = iter((torch.tensor([-2.0, 4.0]), torch.tensor([5.0, 5.0])))
        with (patch.object(cross_eval, "encode_batch", return_value={}),
              patch.object(cross_eval, "tqdm", side_effect=lambda values, **_: values)):
            result = cross_eval.evaluate(
                rows, None, lambda **_: next(scores), 1, 2, "cpu")
        self.assertEqual(result, {"n_items": 2, "accuracy": 1.0})

    def test_fresh_imports_do_not_write_read_research_inputs_or_use_network(self):
        code = """
import os, sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
root = Path.cwd()
protected = [root / name for name in ('data', 'results', 'weights')]

def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('Network during import')
    if event in {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir'}:
        raise RuntimeError('Filesystem mutation during import')
    if event == 'open':
        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise RuntimeError('File write during import')
        if isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if any(path.is_relative_to(directory) for directory in protected):
                raise RuntimeError('Research input read during import')

sys.addaudithook(guard)
import model.baselines
import model.common.pairs
import model.dictabert.model
import model.dictabert.eval
import pipeline.run_all
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_callers_use_the_same_shared_functions(self):
        from model.dictabert import eval as dictabert_eval
        from pipeline import run_all

        self.assertIs(dictabert_eval.load_rows, pairs.load_rows)
        self.assertIs(run_all.load_rows, pairs.load_rows)
        self.assertIs(dictabert_eval.build_model, base_model.build_model)
        self.assertIs(run_all.build_model, base_model.build_model)

    def test_installed_library_treats_omitted_and_none_revision_equally(self):
        from huggingface_hub import hf_hub_url
        from transformers import PretrainedConfig, configuration_utils
        from transformers.models.auto import tokenization_auto

        # Exercise real dependency argument handling, intercepting the cache access
        # before any configuration, tokenizer, model or network data can be read.
        for module, function in (
            (tokenization_auto, tokenization_auto.get_tokenizer_config),
            (configuration_utils, PretrainedConfig._get_config_dict),
        ):
            calls = []
            for kwargs in ({}, {"revision": None}):
                with patch.object(module, "cached_file", return_value=None) as cached:
                    result = function("invented-loading-fixture", **kwargs)
                calls.append((result, cached.call_args))
            self.assertEqual(calls[0], calls[1])
            self.assertIsNone(calls[0][1].kwargs["revision"])
        self.assertEqual(hf_hub_url("fixture/model", "config.json"),
                         hf_hub_url("fixture/model", "config.json", revision=None))
        self.assertIn("/resolve/main/", hf_hub_url("fixture/model", "config.json"))

    def test_shared_builder_defaults_custom_revision_and_no_mode_change(self):
        self.assertEqual(base_model.build_model.__defaults__, (base_model.MODEL_ID, None))
        self.assertEqual(source("model/dictabert/model.py"), source("model/dictabert/model.py", True))
        for args, expected_id, revision in (((), base_model.MODEL_ID, None),
                                           (("fixture-model",), "fixture-model", None),
                                           (("fixture-model", "fixture-revision"),
                                            "fixture-model", "fixture-revision")):
            events, raw_calls = [], []
            transformers, _, token, _ = self.fake_loading(events, raw_calls)
            with patch.dict(sys.modules, {"transformers": transformers}):
                tok, encoder = base_model.build_model(*args)
            self.assertIs(tok, token)
            self.assertTrue(callable(encoder.to))
            self.assertEqual([event[0] for event in events], ["tokenizer", "model"])
            self.assertEqual(raw_calls, [(kind, expected_id, {"revision": revision})
                                         for kind in ("tokenizer", "model")])


if __name__ == "__main__":
    unittest.main()
