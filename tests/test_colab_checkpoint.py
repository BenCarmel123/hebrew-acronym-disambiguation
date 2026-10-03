"""Focused offline checks for the separately authorized Colab loader."""
from pathlib import Path
import io
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hebrew_acronyms.models.dictabert_cross_encoder import colab
from hebrew_acronyms.models.dictabert_cross_encoder.model import CrossEncoder, file_digest


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.weight = nn.Parameter(torch.empty(2, 2))
        self.embeddings.register_buffer('position_ids', torch.arange(512).expand(1, -1), persistent=False)
        self.embeddings.register_buffer('token_type_ids', torch.zeros((1, 512), dtype=torch.long), persistent=False)


class Tokenizer:
    def add_special_tokens(self, tokens):
        assert tokens == {'additional_special_tokens': ['[ACR]', '[/ACR]']}
        return 2

    def __len__(self):
        return 128002

    def convert_tokens_to_ids(self, tokens):
        return [128000, 128001]


class ColabCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'):
            (self.root / name).write_text('{}')
        self.checkpoint = self.root / 'weights.pt'
        self.config = SimpleNamespace(model_type='bert', hidden_size=768, num_hidden_layers=12,
                                      num_attention_heads=12, intermediate_size=3072,
                                      vocab_size=128000, type_vocab_size=2,
                                      max_position_embeddings=512)
        self.state = {k: torch.ones_like(v).half() for k, v in
                      CrossEncoder(TinyEncoder(), 768).state_dict().items()}
        torch.save(self.state, self.checkpoint)

    def load(self):
        with patch.object(colab.AutoConfig, 'from_pretrained', return_value=self.config), \
             patch.object(colab.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
             patch.object(colab.AutoModel, 'from_config', side_effect=lambda _: TinyEncoder()), \
             patch.object(colab, 'tokenizer_identity', return_value={'fixture': True}), \
             patch.object(colab, 'library_versions', return_value={'fixture': True}):
            return colab.load_colab_finetuned(self.checkpoint, snapshot_path=self.root,
                                             expected_sha256=file_digest(self.checkpoint))

    def test_full_fp16_state_and_head_load_into_fp32(self):
        _, model, opened, closed = self.load()
        self.assertEqual((opened, closed), (128000, 128001))
        self.assertTrue(torch.equal(model.score.weight, self.state['score.weight'].float()))
        self.assertEqual(model.score.weight.dtype, torch.float32)
        self.assertFalse(model.encoder.embeddings.position_ids.is_meta)
        self.assertEqual(model.encoder.embeddings.position_ids[0, -1].item(), 511)
        self.assertEqual(model.colab_reconstruction['strict_load']['missing_keys'], [])
        self.assertIsNone(model.colab_reconstruction['original_manifest'])

    def test_missing_head_rejected(self):
        del self.state['score.weight']
        torch.save(self.state, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'score.weight'):
            self.load()

    def test_extra_key_rejected(self):
        self.state['unused'] = torch.ones(1, dtype=torch.float16)
        torch.save(self.state, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'unused'):
            self.load()

    def test_nonfinite_state_rejected(self):
        self.state['score.bias'][0] = float('nan')
        torch.save(self.state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.load()

    def test_hash_mismatch_precedes_model_access(self):
        with patch.object(colab.AutoModel, 'from_config') as construct:
            with self.assertRaisesRegex(ValueError, 'authorized SHA'):
                colab.load_colab_finetuned(self.checkpoint, snapshot_path=self.root,
                                          expected_sha256='0' * 64)
            construct.assert_not_called()

    def test_dataless_read_rejected(self):
        with patch.object(Path, "open", return_value=io.BytesIO(b"")):
            with self.assertRaisesRegex(ValueError, "read 0/"):
                colab._readable_digest(self.checkpoint)

    def test_study_helper_uses_validation_once_and_attaches_provenance(self):
        from hebrew_acronyms.experimental_study import fixture_rows, new_artifact
        rows = fixture_rows()
        artifact = new_artifact(rows, "colab-fixture", {"checkpoint_format": "colab_state_dict"}, {})
        model = SimpleNamespace(colab_reconstruction={"weights_sha256": "a" * 64})
        predictions = []
        for row in rows:
            candidates = [c.strip() for c in row["candidates"].split("|")]
            predictions.append({"item_id": row["item_id"], "status": "ok", "selected_index": 0,
                                "selected_candidate": candidates[0], "error": None,
                                "candidate_scores": [{"candidate": c, "score": float(-i)}
                                                     for i, c in enumerate(candidates)]})
        with patch.object(colab, "load_colab_finetuned", return_value=(None, model, 1, 2)), \
             patch("hebrew_acronyms.models.dictabert_cross_encoder.eval.evaluate",
                   side_effect=[predictions, []]) as evaluate, \
             patch("hebrew_acronyms.experimental_study.save_study") as save:
            actual = colab.predict_colab_study(artifact, checkpoint="fixture.pt", snapshot_path="fixture",
                                              expected_sha256="a" * 64)
        self.assertEqual(actual, predictions)
        self.assertEqual(evaluate.call_args_list[0].args[0], rows)
        self.assertEqual(evaluate.call_args_list[1].args[0], [])
        origin = artifact["records"][0]["encoder_origin"]
        self.assertEqual(origin["checkpoint_format"], "colab_state_dict")
        self.assertNotIn("manifest_sha256", origin)
        save.assert_called_once_with(artifact)
