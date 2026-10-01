"""Training defaults, BCE/AdamW updates and strict checkpoint selection."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from hebrew_acronyms.models.common.pairs import build_pairs
from hebrew_acronyms.models.dictabert_cross_encoder import model as models, training
from tests.fixtures.tiny import TRAINING_ROWS, tiny_base_model


class TrainingTests(unittest.TestCase):
    def test_defaults_and_strict_development_loss_selection(self):
        config = training.TrainingConfig()
        self.assertEqual(asdict(config), dict(epochs=1, batch_size=16, lr=2e-5, seed=42,
                                             max_len=256, eval_batch_size=32, pooling="cls"))
        config = training.TrainingConfig(epochs=4, batch_size=4, max_len=64)
        with patch.object(models, "build_base_model", tiny_base_model):
            tok, model, _, _ = models.build_cross_encoder(model_id="tiny-fixture", revision="fixture-v1")
        pairs = build_pairs(TRAINING_ROWS)
        initial = model.score.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with (patch.object(training, "evaluate_pairs", side_effect=[(.5, .25), (.5, 1.), (.6, 1.), (.4, .25)]),
                  patch.object(training, "save_checkpoint") as save,
                  patch.object(torch.optim, "AdamW", wraps=torch.optim.AdamW) as optimizer,
                  patch.object(training.nn, "BCEWithLogitsLoss", wraps=torch.nn.BCEWithLogitsLoss) as loss):
                history = training.train(model, tok, pairs, pairs, Path(directory) / "best.pt", config=config)
        self.assertEqual([h["checkpoint_saved"] for h in history], [True, False, False, True])
        self.assertEqual([call.args[5] for call in save.call_args_list], [1, 4])
        self.assertEqual(optimizer.call_args.kwargs, {"lr": 2e-5})
        self.assertEqual(loss.call_count, 1)
        self.assertFalse(torch.equal(initial, model.score.weight))
