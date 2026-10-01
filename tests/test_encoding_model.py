"""Encoding, pooling and reconstruction contracts without historical source."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from hebrew_acronyms.models.common import pairs
from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs
from hebrew_acronyms.models.dictabert_cross_encoder.model import (
    CrossEncoder, build_cross_encoder, load_finetuned, save_checkpoint, file_digest, metadata_path,
)
from hebrew_acronyms.models.dictabert_cross_encoder.training import TrainingConfig
from tests.fixtures.tiny import TinyEncoder, marked_tokenizer, tiny_base_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tok = marked_tokenizer()
        self.open_id, self.close_id = self.tok.convert_tokens_to_ids([pairs.ACR_OPEN, pairs.ACR_CLOSE])
        self.batch = encode_pairs(self.tok, [("[ACR]ב״ד[/ACR]", "דגם")], "cpu")

    def test_padding_masks_and_marker_errors(self):
        batch = encode_pairs(self.tok, [("[ACR]א[/ACR]", "ב"),
                                        ("לפני [ACR]א[/ACR]", "בב")], "cpu", max_len=32)
        first = [1, self.open_id, 3 + ord("א") % 61, self.close_id, 2, 3 + ord("ב") % 61, 2]
        padding = batch["input_ids"].shape[1] - len(first)
        self.assertEqual(batch["input_ids"][0].tolist(), first + [0] * padding)
        self.assertEqual(batch["attention_mask"][0].tolist(), [1] * len(first) + [0] * padding)
        self.assertEqual(batch["token_type_ids"][0].tolist(), [0] * 5 + [1] * 2 + [0] * padding)
        for context in ("ללא סימון", "[ACR][/ACR]", "[/ACR]א[ACR]", "[ACR]א[/ACR][ACR]ב[/ACR]"):
            with self.subTest(context=context), self.assertRaises(ValueError):
                encode_pairs(self.tok, [(context, "ב")], "cpu")
        with self.assertRaises(ValueError):
            encode_pairs(self.tok, [], "cpu")

    def test_pooling_uses_expected_vectors(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]]])
        class FixedEncoder(nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(last_hidden_state=hidden)
        ids = torch.tensor([[1, self.open_id, 8, 9, self.close_id]])
        expected = {"cls": [1., 2.], "marker": [3., 4.], "span_mean": [6., 7.],
                    "concat": [1., 2., 6., 7.]}
        for pooling, vector in expected.items():
            model = CrossEncoder(FixedEncoder(), hidden=2, pooling=pooling, dropout=0).eval()
            model.score = nn.Identity()
            output = model(ids, torch.ones_like(ids), acr_open_id=self.open_id, acr_close_id=self.close_id)
            torch.testing.assert_close(output, torch.tensor([vector]))

    def test_factory_does_not_resize_when_markers_already_exist(self):
        encoder = TinyEncoder(66)
        with patch.object(encoder, "resize_token_embeddings") as resize:
            with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", return_value=(self.tok, encoder)):
                build_cross_encoder()
            resize.assert_not_called()

    def test_checkpoint_loader_retains_cpu_load_then_device_order(self):
        events = []
        with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", side_effect=tiny_base_model):
            _, model, _, _ = build_cross_encoder(model_id="fixture", revision="fixture-revision")
        original_to = model.to
        original_load = model.load_state_dict

        def construct(**kwargs):
            self.assertIsNone(kwargs["device"])
            events.append("construct_cpu")
            return self.tok, model, self.open_id, self.close_id

        def load(state, strict=True):
            events.append("load_state")
            return original_load(state, strict=strict)

        def move(device):
            events.append("move_device")
            return original_to(device)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(model, self.tok, checkpoint, TrainingConfig(), {}, 1, 0.5)
            with (patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_cross_encoder", side_effect=construct),
                  patch.object(model, "load_state_dict", side_effect=load),
                  patch.object(model, "to", side_effect=move)):
                load_finetuned(checkpoint, device="cpu")
        self.assertEqual(events, ["construct_cpu", "load_state", "move_device"])

    def test_checkpoint_round_trip_via_real_loader(self):
        for pooling in ("cls", "marker", "span_mean", "concat"):
            with self.subTest(pooling=pooling), tempfile.TemporaryDirectory() as directory:
                with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", side_effect=tiny_base_model):
                    _, model, _, _ = build_cross_encoder(pooling=pooling, model_id="fixture", revision="fixture-revision")
                    model.eval()
                    checkpoint = Path(directory) / "best.pt"
                    save_checkpoint(model, self.tok, checkpoint, TrainingConfig(pooling=pooling), {}, 1, 0.5)
                    _, loaded, opened, closed = load_finetuned(
                        checkpoint, pooling=pooling, model_id="fixture", revision="fixture-revision")
                    self.assertFalse(loaded.training)
                    self.assertEqual((opened, closed), (self.open_id, self.close_id))
                    with torch.no_grad():
                        before = model(**self.batch, acr_open_id=opened, acr_close_id=closed)
                        after = loaded(**self.batch, acr_open_id=opened, acr_close_id=closed)
                    self.assertTrue(torch.equal(before, after))
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
                    broken = deepcopy(model.state_dict())
                    broken.pop("score.bias")
                    torch.save(broken, checkpoint)
                    # With matching digest, strict state loading must still reject missing weights.
                    manifest = json.loads(metadata_path(checkpoint).read_text())
                    manifest["weights_sha256"] = file_digest(checkpoint)
                    metadata_path(checkpoint).write_text(json.dumps(manifest))
                    with self.assertRaises(RuntimeError):
                        load_finetuned(checkpoint, pooling=pooling)
