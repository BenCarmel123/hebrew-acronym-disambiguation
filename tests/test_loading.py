"""CSV and prediction behavior on explicit invented inputs."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from hebrew_acronyms.models.common.pairs import load_rows


class LoadingTests(unittest.TestCase):
    def test_csv_bom_quoted_newline_and_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.csv"
            path.write_text('item_id,sentence\r\nב,"טקסט, ושורה\nנוספת"\r\nא,שני\r\n', encoding="utf-8-sig")
            self.assertEqual(load_rows(path), [dict(item_id="ב", sentence="טקסט, ושורה\nנוספת"),
                                               dict(item_id="א", sentence="שני")])
            path.write_bytes(b"item_id\n\xff")
            with self.assertRaises(UnicodeDecodeError):
                load_rows(path)
            with self.assertRaises(FileNotFoundError):
                load_rows(path.parent / "missing.csv")

    def test_cross_encoder_selects_highest_score_and_first_tie(self):
        from hebrew_acronyms.models.dictabert_cross_encoder import eval as cross_eval

        rows = [dict(item_id=f"item-{i}", sentence="דוגמה א״ב", target_raw="א״ב",
                     span_start=6, span_end=9, candidates="אלף|בית") for i in range(2)]
        scores = iter((torch.tensor([-2.0, 4.0]), torch.tensor([5.0, 5.0])))
        model = unittest.mock.Mock(side_effect=lambda **_: next(scores))
        model.training_config = None
        with patch.object(cross_eval, "encode_batch", return_value={}):
            result = cross_eval.evaluate(rows, None, model, 1, 2, "cpu")
        self.assertEqual([r["item_id"] for r in result], ["item-0", "item-1"])
        self.assertEqual([r["selected_candidate"] for r in result], ["בית", "אלף"])
        self.assertTrue(all(r["status"] == "ok" for r in result))
        self.assertEqual(result[0]["candidate_scores"],
                         [{"candidate": "אלף", "score": -2.0}, {"candidate": "בית", "score": 4.0}])
