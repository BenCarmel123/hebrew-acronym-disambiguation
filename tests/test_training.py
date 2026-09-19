"""Engineering equivalence against isolated training code at the accepted baseline."""

import ast
from contextlib import redirect_stdout
import copy
import io
import random
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hebrew_acronyms.models.common import pairs
from hebrew_acronyms.models.dictabert_cross_encoder import model as model_module
from hebrew_acronyms.models.dictabert_cross_encoder import training
from hebrew_acronyms.models.dictabert_cross_encoder import workflow
from tests.fixtures.tiny import TRAINING_PAIRS, tiny_base_model
from tests.reference import BASE, baseline_notebook_cell, baseline_notebook_namespace


def baseline_training_code():
    """Retain the baseline seed/optimizer/loop, excluding settings and definitions.

    Cell 13 contains only local training operations. Earlier installation, download,
    data-loading and model-loading cells are never executed. Supplied fixtures replace
    its globals; torch.save is intercepted by each caller.
    """
    tree = ast.parse(baseline_notebook_cell(13))
    settings = {"BATCH_SIZE", "EPOCHS", "LR", "SEED"}
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.FunctionDef)
                 and not (isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id in settings
                                  for target in node.targets))]
    return compile(tree, f"git:{BASE}:notebook-cell-13", "exec")


class TrainingEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_rng)
        self.addCleanup(torch.set_rng_state, self.torch_rng)
        torch.manual_seed(187)
        with patch.object(model_module, "build_base_model", tiny_base_model):
            self.tok, self.model, open_id, close_id = model_module.build_cross_encoder()
        self.scope = baseline_notebook_namespace(
            {"CrossEncoder", "encode_batch", "batches", "evaluate"},
            {"torch": torch, "nn": nn, "random": random, "time": time,
             "pairs": pairs, "tok": self.tok, "device": "cpu", "MAX_LEN": 256,
             "ACR_OPEN_ID": open_id, "ACR_CLOSE_ID": close_id})
        self.old_model = self.scope["CrossEncoder"](
            copy.deepcopy(self.model.encoder), hidden=self.model.encoder.config.hidden_size)
        self.old_model.load_state_dict(self.model.state_dict(), strict=True)
        self.scope.update(model=self.old_model, loss_fn=nn.BCEWithLogitsLoss())

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for actual_value, expected_value in zip(actual, expected):
                self.assert_nested_equal(actual_value, expected_value)
        else:
            self.assertEqual(actual, expected)

    def run_baseline(self, config, fixture_pairs, evaluation=None):
        self.scope.update(BATCH_SIZE=config.batch_size, EPOCHS=config.epochs,
                          LR=config.lr, SEED=config.seed, train_pairs=fixture_pairs,
                          dev_pairs=TRAINING_PAIRS)
        if evaluation is not None:
            self.scope["evaluate"] = evaluation
        saved, order = [], []
        original_encode = self.scope["encode_batch"]

        def record_encoding(batch):
            order.append(copy.deepcopy(batch))
            return original_encode(batch)

        def record_save(state, path):
            self.assertEqual(path, "best.pt")
            saved.append(copy.deepcopy(state))

        self.scope["encode_batch"] = record_encoding
        with patch.object(torch, "save", record_save), redirect_stdout(io.StringIO()):
            exec(baseline_training_code(), self.scope)
        return saved, order

    def run_extracted(self, config, fixture_pairs, evaluation=None):
        saved, order, optimizers = [], [], []
        original_encode = training.encode_pairs
        original_adamw = torch.optim.AdamW

        def record_encoding(tok, batch, device, **kwargs):
            order.append(copy.deepcopy(batch))
            return original_encode(tok, batch, device, **kwargs)

        def record_optimizer(*args, **kwargs):
            optimizer = original_adamw(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        def record_save(state, path):
            self.assertEqual(path, "unused-fixture-checkpoint.pt")
            saved.append(copy.deepcopy(state))

        original_evaluate = training.evaluate_pairs
        with (patch.object(training, "encode_pairs", record_encoding),
              patch.object(torch.optim, "AdamW", record_optimizer),
              patch.object(torch, "save", record_save),
              patch.object(training, "evaluate_pairs", evaluation or original_evaluate),
              redirect_stdout(io.StringIO())):
            history = training.train(
                self.model, self.tok, fixture_pairs, TRAINING_PAIRS,
                "unused-fixture-checkpoint.pt", config=config)
        return saved, order, optimizers[0], history

    def test_batch_order_matches_baseline_with_and_without_shuffle(self):
        for shuffle in (False, True):
            for batch_size in (1, 2, 16):
                with self.subTest(shuffle=shuffle, batch_size=batch_size):
                    random.seed(42)
                    expected = list(self.scope["batches"](TRAINING_PAIRS, batch_size, shuffle))
                    random.seed(42)
                    actual = list(training.batches(TRAINING_PAIRS, batch_size, shuffle))
                    self.assertEqual(actual, expected)

    def test_default_configuration_matches_baseline_notebook(self):
        names = {"max_len": "MAX_LEN", "batch_size": "BATCH_SIZE", "epochs": "EPOCHS",
                 "lr": "LR", "seed": "SEED", "pooling": "POOLING"}
        constants = {}
        for index in (9, 13):
            for node in ast.parse(baseline_notebook_cell(index)).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in names.values():
                            constants[target.id] = ast.literal_eval(node.value)
        expected = {field: constants[name] for field, name in names.items()}
        evaluate = next(node for node in ast.parse(baseline_notebook_cell(13)).body
                        if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        batch_call = next(node for node in ast.walk(evaluate)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == "batches")
        expected["eval_batch_size"] = next(ast.literal_eval(keyword.value)
                                           for keyword in batch_call.keywords
                                           if keyword.arg == "batch_size")
        self.assertEqual(vars(training.TrainingConfig()), expected)

    def test_pair_loss_and_accuracy_match_baseline(self):
        # 33 pairs exercise a partial evaluation batch under the existing size 32.
        fixture_pairs = (TRAINING_PAIRS * 11)[:33]
        expected = self.scope["evaluate"](fixture_pairs)
        actual = training.evaluate_pairs(self.model, self.tok, fixture_pairs, "cpu")
        self.assertEqual(actual, expected)
        self.assertFalse(self.model.training)

    def test_one_optimizer_update_matches_baseline(self):
        self.check_optimizer_equivalence(TRAINING_PAIRS[:2])

    def test_two_optimizer_updates_match_baseline_including_partial_batch(self):
        self.check_optimizer_equivalence(TRAINING_PAIRS)

    def check_optimizer_equivalence(self, fixture_pairs):
        config = training.TrainingConfig(batch_size=2)
        initial = copy.deepcopy(self.model.state_dict())
        saved_old, order_old = self.run_baseline(config, fixture_pairs)
        saved_new, order_new, optimizer, history = self.run_extracted(config, fixture_pairs)
        self.assertEqual(order_new, order_old)
        self.assert_nested_equal(self.model.state_dict(), self.old_model.state_dict())
        self.assert_nested_equal(optimizer.state_dict(), self.scope["optimizer"].state_dict())
        self.assert_nested_equal(saved_new, saved_old)
        self.assertEqual(history[0]["train_loss"], self.scope["train_loss"])
        self.assertEqual(history[0]["dev_loss"], self.scope["dev_loss"])
        self.assertEqual(history[0]["dev_pair_accuracy"], self.scope["dev_pair_acc"])
        self.assertTrue(history[0]["checkpoint_saved"])
        self.assertTrue(any(not torch.equal(initial[key], value)
                            for key, value in self.model.state_dict().items()))

    def test_checkpoint_selection_matches_strict_baseline_loss_improvement(self):
        config = training.TrainingConfig(batch_size=16, epochs=4)
        controlled_losses = (0.5, 0.5, 0.6, 0.4)
        old_losses = iter(controlled_losses)
        new_losses = iter(controlled_losses)
        saved_old, _ = self.run_baseline(
            config, TRAINING_PAIRS, lambda _: (next(old_losses), 0.25))
        saved_new, _, _, history = self.run_extracted(
            config, TRAINING_PAIRS, lambda *args: (next(new_losses), 0.25))
        self.assertEqual([row["checkpoint_saved"] for row in history],
                         [True, False, False, True])
        self.assertEqual(len(saved_new), 2)
        self.assert_nested_equal(saved_new, saved_old)

    def test_workflow_initializes_before_applying_training_seed(self):
        events = []
        original_python_seed, original_torch_seed = random.seed, torch.manual_seed

        def initialize(*args, **kwargs):
            events.append("initialize")
            return tiny_base_model(*args, **kwargs)

        def python_seed(seed):
            events.append(("python_seed", seed))
            return original_python_seed(seed)

        def torch_seed(seed):
            events.append(("torch_seed", seed))
            return original_torch_seed(seed)

        # No configure_run call: the fixture factory needs no actual model cache.
        run = {"mode": "train", "model_id": "tiny-fixture", "device": "cpu",
               "checkpoint_path": "unused-fixture-checkpoint.pt"}
        config = training.TrainingConfig()
        with (patch.object(model_module, "build_base_model", initialize),
              patch.object(random, "seed", python_seed),
              patch.object(torch, "manual_seed", torch_seed),
              patch.object(torch, "save"), redirect_stdout(io.StringIO())):
            tok, model, _, _ = workflow.load_model(run, config)
            self.assertEqual(events, ["initialize"])
            result = workflow.run_action(
                run, model, tok, TRAINING_PAIRS, TRAINING_PAIRS, config)
        self.assertEqual(events, ["initialize", ("python_seed", 42), ("torch_seed", 42)])
        self.assertEqual(len(result["history"]), 1)


if __name__ == "__main__":
    unittest.main()
