"""E1 contracts on invented inputs and a tiny encoder only; no research claims."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hebrew_acronyms.models.common.pairs import (
    ACR_OPEN, ACR_CLOSE, build_pairs, explicit_span, mark_span,
)
from hebrew_acronyms.models.dictabert_cross_encoder import model as models
from hebrew_acronyms.models.dictabert_cross_encoder.encoding import encode_pairs
from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate
from hebrew_acronyms.models.dictabert_cross_encoder.training import TrainingConfig, train, sanity_overfit
from tests.fixtures.tiny import TRAINING_ROWS, marked_tokenizer, tiny_base_model

ROOT = Path(__file__).resolve().parents[1]


def build(config=TrainingConfig()):
    return models.build_cross_encoder(model_id="tiny-fixture", revision="fixture-v1",
                                      pooling=config.pooling, seed=config.seed)


class InputContractTests(unittest.TestCase):
    def test_selected_second_occurrence_keeps_prefix_and_original_text(self):
        row = dict(item_id="repeated", sentence='בב״ד וגם בב״ד.', target_raw='בב״ד',
                   span_start=9, span_end=13, candidates='בדיקת דוגמה|בניית דגם',
                   gold_expansion='בניית דגם')
        values = build_pairs([row])
        self.assertEqual(values[0][0], 'בב״ד וגם [ACR]בב״ד[/ACR].')
        self.assertEqual([label for _, _, label in values], [0, 1])
        self.assertEqual(values.identity['item_ids'], ['repeated'])
        self.assertEqual(len(values.identity['sha256']), 64)

    def test_missing_invalid_and_nonmatching_spans_rejected(self):
        original = TRAINING_ROWS[0]
        changes = [{"span_start": None}, {"span_start": -1}, {"span_end": 999},
                   {"span_end": 5}, {"span_start": True}, {"span_start": 5.0},
                   {"span_start": "5.0"}, {"target_raw": 'ב"ד'}, {"target_raw": "ב"},
                   {"sentence": original['sentence'] + ACR_OPEN}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_pairs([{**original, **change}])
        self.assertEqual(explicit_span({**original, 'span_start': '5', 'span_end': '8'}), (5, 8))

    def test_missing_and_duplicate_ids_reject_training_and_prediction(self):
        for rows in ([{**TRAINING_ROWS[0], 'item_id': ''}],
                     [{k: v for k, v in TRAINING_ROWS[0].items() if k != 'item_id'}],
                     [TRAINING_ROWS[0], TRAINING_ROWS[0]]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    build_pairs(rows)
                with self.assertRaises(ValueError):
                    evaluate(rows, None, None, 1, 2, 'cpu')

    def test_exactly_one_positive_no_alias_or_inventory_repair(self):
        row = TRAINING_ROWS[0]
        for changes in ({'gold_expansion': 'not in inventory'},
                        {'candidates': 'בדיקת דוגמה|בדיקת דוגמה'},
                        {'gold_expansion': ''}, {'candidates': 'בדיקת דוגמה'},
                        {'candidates': 'בדיקת דוגמה|'},
                        {'candidates': 'בדיקת דוגמה|בניית דגם|בניית דגם'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_pairs([{**row, **changes}])

    def test_prepared_inputs_detect_tampering(self):
        prepared = build_pairs(TRAINING_ROWS)
        prepared.rows[0]['item_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepared.verify()

    def test_exact_budget_preserves_full_target_markers_and_candidate(self):
        tok = marked_tokenizer()
        context = 'הקשר ' * 100 + '[ACR]בב״ד[/ACR]' + ' המשך' * 100
        candidate = 'דוגמה'
        target_ids = tok.encode('[ACR]בב״ד[/ACR]')
        candidate_ids = tok.encode(candidate)
        budget = len(target_ids) + len(candidate_ids) + 3
        encoded = encode_pairs(tok, [(context, candidate)], 'cpu', budget)
        self.assertEqual(encoded['input_ids'].tolist()[0],
                         [tok.cls_token_id] + target_ids + [tok.sep_token_id] + candidate_ids + [tok.sep_token_id])
        with self.assertRaisesRegex(ValueError, 'require'):
            encode_pairs(tok, [(context, candidate)], 'cpu', budget - 1)
        with self.assertRaises(ValueError):
            encode_pairs(tok, [(context, ACR_OPEN + candidate)], 'cpu', 256)


class LearningAndPredictionTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        patcher = patch.object(models, 'build_base_model', tiny_base_model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cpu_initialization_reproducible_before_all_construction(self):
        states = []
        for seed in (71, 71, 72):
            torch.manual_seed(9999)  # Surrounding RNG state must not determine initialization.
            _, model, _, _ = build(TrainingConfig(seed=seed))
            states.append(deepcopy(model.state_dict()))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        for key in ('encoder.embeddings.weight', 'encoder.projection.weight', 'score.weight'):
            self.assertFalse(torch.equal(states[0][key], states[2][key]))

    def test_identified_records_singleton_errors_and_saved_length(self):
        tok, model, opened, closed = build()
        rows = deepcopy(TRAINING_ROWS)
        rows += [{**rows[0], 'item_id': 'singleton', 'candidates': 'אחד'},
                 {**rows[0], 'item_id': 'bad-span', 'span_start': -1},
                 {**rows[0], 'item_id': 'too-long', 'candidates': 'א' * 100},
                 {**rows[0], 'item_id': 'empty', 'candidates': ''}]
        model.training_config = TrainingConfig(max_len=40)
        results = evaluate(rows, tok, model, opened, closed, 'cpu')
        self.assertEqual([r['item_id'] for r in results], [r['item_id'] for r in rows])
        self.assertEqual([r['status'] for r in results],
                         ['ok', 'ok', 'ok', 'invalid_input', 'encoding_error', 'invalid_input'])
        self.assertEqual(results[2]['selected_candidate'], 'אחד')
        self.assertEqual(len(results[2]['candidate_scores']), 1)
        self.assertTrue(all(r['error'] for r in results[3:]))
        self.assertFalse(model.training)
        with self.assertRaisesRegex(ValueError, 'max_len override'):
            evaluate(rows, tok, model, opened, closed, 'cpu', max_len=256)

    def test_invalid_scores_and_runtime_errors_keep_identified_records(self):
        tok, model, opened, closed = build()
        for logits in (torch.tensor([float('nan'), 0.]), torch.tensor([1.])):
            with patch.object(model, 'forward', return_value=logits):
                results = evaluate(TRAINING_ROWS[:1], tok, model, opened, closed, 'cpu')
            self.assertEqual(results[0]['status'], 'invalid_scores')
        with patch.object(model, 'forward', side_effect=[torch.tensor([0., 1.]), RuntimeError('fixture failure')]):
            result = evaluate(TRAINING_ROWS, tok, model, opened, closed, 'cpu')
        self.assertEqual([r['item_id'] for r in result], [r['item_id'] for r in TRAINING_ROWS])
        self.assertEqual([r['status'] for r in result], ['ok', 'model_error'])
        self.assertEqual(result[1]['error'], 'fixture failure')

    def test_train_preflight_rejects_bad_inputs_without_updates(self):
        tok, model, _, _ = build()
        initial = deepcopy(model.state_dict())
        pairs = build_pairs(TRAINING_ROWS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unused.pt'
            for tr, dev, cfg in ((pairs, [], TrainingConfig()),
                                 (list(pairs), pairs, TrainingConfig()),
                                 (pairs, pairs, TrainingConfig(max_len=3)),
                                 (pairs, pairs, TrainingConfig(seed=99))):
                with self.assertRaises(ValueError):
                    train(model, tok, tr, dev, path, config=cfg)
            self.assertFalse(path.exists())
        for key, value in initial.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_tiny_overfit_and_checkpoint_roundtrip(self):
        rows = json.loads((ROOT / 'tests/fixtures/training_rows.json').read_text())
        config = TrainingConfig(epochs=180, lr=0.03, batch_size=4, max_len=64)
        tok, model, opened, closed = build(config)
        initial = deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / 'best.pt'
            result = sanity_overfit(model, tok, rows, path, config=config)
            self.assertEqual(result['status'], 'PASS')
            self.assertLess(result['final_loss'], result['initial_loss'] * 0.5)
            self.assertEqual([r['selected_candidate'] for r in result['predictions']],
                             [r['gold_expansion'] for r in rows])
            self.assertFalse(torch.equal(initial['encoder.embeddings.weight'], model.encoder.embeddings.weight))
            self.assertFalse(torch.equal(initial['score.weight'], model.score.weight))
            manifest = json.loads(models.metadata_path(path).read_text())
            self.assertEqual(manifest['inputs']['train']['item_ids'], [r['item_id'] for r in rows])
            self.assertEqual(manifest['selection']['dev_pair_loss'], min(h['dev_loss'] for h in result['history']))
            loaded_tok, loaded, lo, lc = models.load_finetuned(path, expected_inputs=manifest['inputs'])
            self.assertEqual(loaded.training_config, config)
            self.assertEqual([r['selected_candidate'] for r in evaluate(rows, loaded_tok, loaded, lo, lc, 'cpu')],
                             [r['gold_expansion'] for r in rows])
            # Roundtrip the actual final model separately: selected best can precede final epoch.
            final = Path(directory) / 'final-fixture.pt'
            models.save_checkpoint(model, tok, final, config, manifest['inputs'], config.epochs, result['final_loss'])
            tok2, model2, o2, c2 = models.load_finetuned(final)
            before = evaluate(rows, tok, model, opened, closed, 'cpu')
            after = evaluate(rows, tok2, model2, o2, c2, 'cpu')
            self.assertFalse(model2.training)
            for left, right in zip(before, after):
                self.assertEqual(left['selected_candidate'], right['selected_candidate'])
                torch.testing.assert_close(torch.tensor([p['score'] for p in left['candidate_scores']]),
                                           torch.tensor([p['score'] for p in right['candidate_scores']]),
                                           rtol=0, atol=1e-7)
            for overrides in ({'pooling': 'marker'}, {'revision': 'wrong'}, {'model_id': 'wrong'},
                              {'config': TrainingConfig()}, {'expected_inputs': {}}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    models.load_finetuned(final, **overrides)
            with patch.object(models, 'tokenizer_identity', return_value={'wrong': True}):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    models.load_finetuned(final)
            with open(final, 'ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'weights do not match'):
                models.load_finetuned(final)

    def test_local_source_changes_detected_and_metadata_required(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / 'snapshot'
            snapshot.mkdir()
            source = snapshot / 'config.json'
            source.write_text('{"fixture":1}')
            tok, model, _, _ = models.build_cross_encoder(model_id=str(snapshot))
            path = Path(directory) / 'best.pt'
            models.save_checkpoint(model, tok, path, TrainingConfig(), {}, 1, 0.5)
            source.write_text('{"fixture":2}')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                models.load_finetuned(path)
            with self.assertRaisesRegex(ValueError, 'metadata is required'):
                models.load_finetuned(Path(directory) / 'legacy.pt')


if __name__ == '__main__':
    unittest.main()
