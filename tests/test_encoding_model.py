"""Compare shared code to exact baseline definitions, using tiny fixtures only."""

import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hebrew_acronyms.models.common import pairs
from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs
from hebrew_acronyms.models.dictabert_cross_encoder.eval import encode_batch
from hebrew_acronyms.models.dictabert_cross_encoder.model import CrossEncoder, build_cross_encoder, load_finetuned
from tests.fixtures.tiny import TinyEncoder, TinyTokenizer, marked_tokenizer, tiny_base_model
from tests.reference import baseline_module_namespace, baseline_notebook_cell, baseline_notebook_namespace


class EncodingEquivalence(unittest.TestCase):
    def setUp(self):
        self.tok = marked_tokenizer()
        self.open_id, self.close_id = self.tok.convert_tokens_to_ids(
            [pairs.ACR_OPEN, pairs.ACR_CLOSE])
        self.old_notebook = baseline_notebook_namespace(
            ["encode_batch"], {"torch": torch, "pairs": pairs, "tok": self.tok,
                               "device": "cpu", "MAX_LEN": 256})["encode_batch"]
        self.old_eval = baseline_module_namespace(
            "model/dictabertX/eval.py", ["encode_batch"],
            {"torch": torch, "MAX_LEN": 256})["encode_batch"]

    def assert_batch_equal(self, left, right):
        self.assertEqual(set(left), {"input_ids", "attention_mask", "token_type_ids"})
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertEqual(left[key].dtype, torch.long)
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def test_short_pairs_padding_and_masks(self):
        batch = [("לפני [ACR]ב״ד[/ACR] אחרי", "בדיקת דוגמה"),
                 ("[ACR]ב״ד[/ACR]", "דגם"),
                 ("מחר [ACR]ב״ד[/ACR] חדש", "")]
        actual = encode_pairs(self.tok, batch, "cpu")
        self.assert_batch_equal(self.old_notebook(batch), actual)
        self.assertTrue((actual["attention_mask"] == 0).any().item())
        for context, _ in batch:
            candidates = ["דגם", "בדיקת דוגמה", ""]
            self.assert_batch_equal(
                self.old_eval(self.tok, "cpu", self.open_id, self.close_id, context, candidates),
                encode_batch(self.tok, "cpu", self.open_id, self.close_id, context, candidates))

    def test_long_context_near_beginning_and_end(self):
        for context in ("[ACR]ב״ד[/ACR] " + "מילה " * 120,
                        "מילה " * 120 + " [ACR]ב״ד[/ACR]",
                        "מילה " * 60 + " [ACR]ב״ד[/ACR] " + "מילה " * 60):
            with self.subTest(context_start=context[:20]):
                batch = [(context, "בדיקת דוגמה"), (context, "דגם")]
                actual = encode_pairs(self.tok, batch, "cpu")
                self.assert_batch_equal(self.old_notebook(batch), actual)
                self.assertEqual(actual["input_ids"].shape[1], 256)
                self.assertTrue((actual["input_ids"] == self.open_id).any(1).all())
                self.assertTrue((actual["input_ids"] == self.close_id).any(1).all())
                self.assert_batch_equal(
                    self.old_eval(self.tok, "cpu", self.open_id, self.close_id, context,
                                  [candidate for _, candidate in batch]),
                    encode_batch(self.tok, "cpu", self.open_id, self.close_id, context,
                                 [candidate for _, candidate in batch]))

    def test_over_budget_span_or_candidate_is_preserved(self):
        for batch in ([('[ACR]יעד[/ACR]', 'מ' * 300)],
                      [('[ACR]' + 'מ' * 300 + '[/ACR]', 'ד')]):
            self.assert_batch_equal(self.old_notebook(batch), encode_pairs(self.tok, batch, "cpu"))
            self.assertGreater(encode_pairs(self.tok, batch, "cpu")["input_ids"].shape[1], 256)

    def test_edge_marker_arrangements_and_custom_budget(self):
        for context in ("[ACR][/ACR]", "[/ACR]ד[ACR]", "[ACR]א[/ACR][ACR]ב[/ACR]"):
            for budget in (0, 10, 256):
                with self.subTest(context=context, max_len=budget):
                    batch = [(context, "מועמד")]
                    self.assert_batch_equal(self.old_notebook(batch, budget),
                                            encode_pairs(self.tok, batch, "cpu", budget))

    def test_explicit_and_inferred_marker_ids(self):
        batch = [("[ACR]ב״ד[/ACR]", "דגם")]
        expected = self.old_notebook(batch)
        for ids in ((None, None), (self.open_id, None), (None, self.close_id),
                    (self.open_id, self.close_id)):
            self.assert_batch_equal(expected, encode_pairs(
                self.tok, batch, "cpu", acr_open_id=ids[0], acr_close_id=ids[1]))

    def test_missing_markers_and_empty_batch_errors_match(self):
        for batch in ([], [("ללא סימון", "דגם")], [("[ACR]ב״ד", "דגם")],
                      [("ב״ד[/ACR]", "דגם")]):
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError) as before:
                    self.old_notebook(batch)
                with self.assertRaises(ValueError) as after:
                    encode_pairs(self.tok, batch, "cpu")
                self.assertEqual(str(before.exception), str(after.exception))
        for context in ("ללא סימון", "[ACR]ב״ד", "[ACR]ב״ד[/ACR]"):
            for candidates in ([], ["דגם"]):
                if candidates and context.endswith("[/ACR]"):
                    continue
                with self.subTest(context=context, candidates=candidates):
                    with self.assertRaises(ValueError) as before:
                        self.old_eval(self.tok, "cpu", self.open_id, self.close_id,
                                      context, candidates)
                    with self.assertRaises(ValueError) as after:
                        encode_batch(self.tok, "cpu", self.open_id, self.close_id,
                                     context, candidates)
                    self.assertEqual(str(before.exception), str(after.exception))

    def test_eval_wrapper_keeps_module_max_len(self):
        context = "מילה " * 20 + " [ACR]ב״ד[/ACR]"
        old = baseline_module_namespace("model/dictabertX/eval.py", ["encode_batch"],
                                        {"torch": torch, "MAX_LEN": 20})["encode_batch"]
        with patch("hebrew_acronyms.models.dictabert_cross_encoder.eval.MAX_LEN", 20):
            self.assert_batch_equal(old(self.tok, "cpu", self.open_id, self.close_id,
                                        context, ["דגם"]),
                                    encode_batch(self.tok, "cpu", self.open_id, self.close_id,
                                                 context, ["דגם"]))


class ModelEquivalence(unittest.TestCase):
    def setUp(self):
        self.tok = marked_tokenizer()
        self.open_id, self.close_id = self.tok.convert_tokens_to_ids(
            [pairs.ACR_OPEN, pairs.ACR_CLOSE])
        scope = {"torch": torch, "nn": nn, "ACR_OPEN_ID": self.open_id,
                 "ACR_CLOSE_ID": self.close_id}
        self.old_notebook = baseline_notebook_namespace(["CrossEncoder"], scope)["CrossEncoder"]
        self.old_module = baseline_module_namespace("model/dictabertX/model.py",
                                                     ["CrossEncoder"], scope)["CrossEncoder"]
        self.batch = {
            "input_ids": torch.tensor([[1, 64, 8, 9, 65, 2], [1, 64, 65, 2, 0, 0],
                                        [1, 7, 8, 2, 0, 0], [1, 64, 8, 64, 9, 65]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0],
                                             [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]),
            "token_type_ids": torch.zeros(4, 6, dtype=torch.long),
        }

    def test_all_poolings_logits_state_shapes_and_strict_loading(self):
        for pooling in ("cls", "marker", "span_mean", "concat"):
            with self.subTest(pooling=pooling):
                old = self.old_notebook(TinyEncoder(66), 8, pooling).eval()
                new = CrossEncoder(TinyEncoder(66), 8, pooling).eval()
                previous = self.old_module(TinyEncoder(66), 8, pooling).eval()
                expected = {key: value.shape for key, value in old.state_dict().items()}
                self.assertEqual(expected, {key: value.shape for key, value in new.state_dict().items()})
                new.load_state_dict(old.state_dict(), strict=True)
                previous.load_state_dict(old.state_dict(), strict=True)
                for with_types in (True, False):
                    batch = self.batch if with_types else {k: v for k, v in self.batch.items()
                                                          if k != "token_type_ids"}
                    with torch.no_grad():
                        actual = new(**batch, acr_open_id=self.open_id, acr_close_id=self.close_id)
                        self.assertTrue(torch.equal(old(**batch), actual))
                        self.assertTrue(torch.equal(previous(**batch, acr_open_id=self.open_id,
                                                               acr_close_id=self.close_id), actual))
                with self.assertRaises(RuntimeError):
                    new.load_state_dict({**old.state_dict(), "unexpected": torch.tensor(0)}, strict=True)

    def test_factory_matches_baseline_initialization_and_rng(self):
        state = torch.get_rng_state()
        # Execute only the baseline's initialization assignments and resize guard.
        # The download-capable constructors are replaced before AST execution.
        nodes = ast.parse(baseline_notebook_cell(9)).body
        names = {"tok", "encoder", "n_added", "ACR_OPEN_ID", "ACR_CLOSE_ID", "model"}
        selected = []
        for node in nodes:
            if isinstance(node, ast.Assign):
                assigned = {item.id for target in node.targets for item in ast.walk(target)
                            if isinstance(item, ast.Name)}
                if assigned & names:
                    selected.append(node)
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                if node.test.id == "n_added":
                    selected.append(node)
        scope = {"AutoTokenizer": SimpleNamespace(from_pretrained=lambda _: TinyTokenizer()),
                 "AutoModel": SimpleNamespace(from_pretrained=lambda _: TinyEncoder()),
                 "MODEL_ID": "fixture", "POOLING": "cls", "pairs": pairs, "device": "cpu",
                 "CrossEncoder": self.old_notebook}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "baseline-initialization", "exec"), scope)
        expected = scope["model"]
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", side_effect=tiny_base_model) as loader:
            with patch("torch.manual_seed", side_effect=AssertionError("factory must not seed")):
                _, actual, opened, closed = build_cross_encoder(model_id="fixture", revision="fixture-revision")
        loader.assert_called_once_with(model_id="fixture", revision="fixture-revision")
        self.assertEqual((opened, closed), (self.open_id, self.close_id))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[key]), key)

    def test_factory_does_not_resize_when_markers_already_exist(self):
        encoder = TinyEncoder(66)
        with patch.object(encoder, "resize_token_embeddings") as resize:
            with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", return_value=(self.tok, encoder)):
                build_cross_encoder()
            resize.assert_not_called()

    def test_checkpoint_loader_retains_cpu_load_then_device_order(self):
        events = []
        with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", side_effect=tiny_base_model):
            _, model, _, _ = build_cross_encoder()
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

        with (patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_cross_encoder", side_effect=construct),
              patch("torch.load", return_value=model.state_dict()),
              patch.object(model, "load_state_dict", side_effect=load),
              patch.object(model, "to", side_effect=move)):
            load_finetuned("fixture.pt", device="cpu")
        self.assertEqual(events, ["construct_cpu", "load_state", "move_device"])

    def test_checkpoint_round_trip_via_real_loader(self):
        for pooling in ("cls", "marker", "span_mean", "concat"):
            with self.subTest(pooling=pooling), tempfile.TemporaryDirectory() as directory:
                with patch("hebrew_acronyms.models.dictabert_cross_encoder.model.build_base_model", side_effect=tiny_base_model):
                    _, model, _, _ = build_cross_encoder(pooling=pooling)
                    model.eval()
                    checkpoint = Path(directory) / "best.pt"
                    torch.save(model.state_dict(), checkpoint)
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
                    with self.assertRaises(RuntimeError):
                        load_finetuned(checkpoint, pooling=pooling)


if __name__ == "__main__":
    unittest.main()
