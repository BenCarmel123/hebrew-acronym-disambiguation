"""DictaLM LoRA selection examples, training loop and inference, on a tiny invented model."""
from contextlib import redirect_stdout
import io
import json
import string
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hebrew_acronyms.models.common.eval import build_select_prompt
from hebrew_acronyms.models.dictalm_lora import examples, training
from hebrew_acronyms.models.dictalm_lora.eval import make_generate_fn
from tests.fixtures.tiny import TRAINING_ROWS

ROWS = [dict(row, acronym=row["target_raw"]) for row in TRAINING_ROWS]


class ByteTokenizer:
    """UTF-8 bytes shifted past three special IDs; no chat template."""

    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    chat_template = None
    vocab_size = 259

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + b for b in text.encode("utf-8")]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}

    def decode(self, ids, skip_special_tokens=False):
        return bytes(int(i) - 3 for i in ids if int(i) >= 3).decode("utf-8", errors="replace")


def tiny_causal_lm():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=ByteTokenizer.vocab_size, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=1024, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return LlamaForCausalLM(config)


TINY_CONFIG = training.LoraTrainingConfig(epochs=40, batch_size=2, grad_accum=1, lr=1e-2,
                                          lora_r=8, lora_alpha=16, lora_dropout=0.0,
                                          target_modules=("q_proj", "v_proj", "o_proj", "down_proj"))


class ExampleTests(unittest.TestCase):
    def test_prompt_and_letter_match_the_shared_selection_format(self):
        built = examples.build_selection_examples(ROWS, seed=42)
        self.assertEqual(built, examples.build_selection_examples(ROWS, seed=42))
        for row, example in zip(ROWS, built):
            shown = example["shown_order"]
            self.assertEqual(sorted(shown), sorted(row["candidates"].split("|")))
            self.assertEqual(example["prompt"], build_select_prompt(row["acronym"], row["sentence"], shown))
            self.assertEqual(shown[string.ascii_uppercase.index(example["answer"])], row["gold_expansion"])

    def test_invalid_training_rows_are_rejected(self):
        bad = [dict(ROWS[0], gold_expansion="לא ברשימה"),
               dict(ROWS[0], candidates="|".join(f"מועמד {i}" for i in range(27)), gold_expansion="מועמד 0"),
               dict(ROWS[0], candidates="בדיקת דוגמה|בדיקת דוגמה")]
        for row in bad:
            with self.assertRaises(ValueError):
                examples.build_selection_examples([row], seed=42)

    def test_only_answer_and_eos_are_trained_and_padding_is_masked(self):
        tok = ByteTokenizer()
        example = {"item_id": "x", "prompt": "שאלה", "answer": "B"}
        encoded = examples.encode_example(tok, example, max_len=64)
        prompt_len = len(tok("שאלה")["input_ids"])
        self.assertEqual(encoded["labels"][:prompt_len], [examples.IGNORE_INDEX] * prompt_len)
        self.assertEqual(encoded["labels"][prompt_len:], [3 + ord("B"), tok.eos_token_id])
        with self.assertRaises(ValueError):
            examples.encode_example(tok, example, max_len=prompt_len)
        batch = examples.collate([encoded, {"input_ids": [5], "labels": [5]}], pad_token_id=0)
        self.assertEqual(batch["attention_mask"][1][1:], [0] * (len(encoded["input_ids"]) - 1))
        self.assertEqual(set(batch["labels"][1][1:]), {examples.IGNORE_INDEX})


class TrainingTests(unittest.TestCase):
    def test_lora_initialization_follows_the_seed(self):
        first = training.add_lora(tiny_causal_lm(), TINY_CONFIG, gradient_checkpointing=False)
        base = tiny_causal_lm()
        torch.rand(100)  # Advance the generator, so only the seed in add_lora can match.
        second = training.add_lora(base, TINY_CONFIG, gradient_checkpointing=False)
        for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
            if "lora_" in name:
                self.assertTrue(torch.equal(a, b), name)

    def test_strict_development_loss_selection(self):
        tok = ByteTokenizer()
        model = training.add_lora(tiny_causal_lm(), training.LoraTrainingConfig(
            epochs=4, batch_size=2, target_modules=("q_proj",)), gradient_checkpointing=False)
        config = training.LoraTrainingConfig(epochs=4, batch_size=2, target_modules=("q_proj",))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            with (patch.object(training, "evaluate_loss", side_effect=[.5, .5, .6, .4]),
                  patch.object(training, "save_adapter") as save):
                history = training.train(model, tok, ROWS, ROWS, output, config)
            self.assertEqual([h["checkpoint_saved"] for h in history], [True, False, False, True])
            self.assertEqual([call.args[2]["epoch"] for call in save.call_args_list], [1, 4])
            with self.assertRaises(FileExistsError):
                training.train(model, tok, ROWS, ROWS, output, config)

    def test_training_learns_the_answers_and_saves_the_adapter(self):
        tok = ByteTokenizer()
        model = training.add_lora(tiny_causal_lm(), TINY_CONFIG, gradient_checkpointing=False)
        encoded = [examples.encode_example(tok, e, TINY_CONFIG.max_len)
                   for e in examples.build_selection_examples(ROWS, TINY_CONFIG.seed)]
        initial = training.evaluate_loss(model, tok, encoded, TINY_CONFIG)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            history = training.train(model, tok, ROWS, ROWS, output, TINY_CONFIG, {"model_id": "tiny"})
            self.assertTrue((output / "adapter" / "adapter_config.json").is_file())
            saved = json.loads((output / "training_run.json").read_text(encoding="utf-8"))
        self.assertLess(history[-1]["dev_loss"], initial)
        self.assertEqual(saved["run_info"], {"model_id": "tiny"})
        self.assertEqual(saved["inputs"]["train"]["item_ids"], ["tiny-1", "tiny-2"])
        generate = make_generate_fn(model, tok, max_new_tokens=1)
        for example in examples.build_selection_examples(ROWS, TINY_CONFIG.seed):
            self.assertEqual(generate(example["prompt"]), example["answer"])

    def test_training_requires_trainable_parameters(self):
        model = tiny_causal_lm()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            training.train(model, ByteTokenizer(), ROWS, ROWS, Path(directory) / "run", TINY_CONFIG)


if __name__ == "__main__":
    unittest.main()
