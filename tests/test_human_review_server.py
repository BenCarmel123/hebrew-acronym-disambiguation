"""Offline review persistence/HTTP checks using invented data only."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
from hebrew_acronyms.human_review_server import ReviewStore, SCHEMA, make_server


def fixture():
    return {"dataset_id": "invented-fixture", "provenance": {"files": []},
            "systems": [{"id": "test", "blind_id": "מערכת א", "name": "Invented"}],
            "queues": {"calibration": ["fiction-1"], "evaluation": ["fiction-1"], "diagnosis": []},
            "items": [{"id": "fiction-1", "sentence": 'אב״ג וגם אב"ג', "acronym": 'אב״ג',
                       "answers": [{"id": "answer-1", "system_id": "test", "raw": ""}]}]}


def annotation():
    return {"schema_version": SCHEMA, "annotator": "QA only", "interpretation": 'אב״ג',
            "prior_exposure": "unknown", "item_problems": [], "answers": {
                "answer-1": {"system_id": "test", "quality": "", "format_ok": "", "disagrees_auto": ""}}}


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "annotations.json"
        self.store = ReviewStore(fixture(), self.path)

    def tearDown(self):
        self.directory.cleanup()

    def update(self, **extra):
        return self.store.update({"item_id": "fiction-1", "revision": self.store.state["revision"], **extra})

    def test_csv_roundtrip_preserves_blank_labels_and_review_status(self):
        self.update(action="review", annotation=annotation())
        text = self.store.export_csv()
        self.assertIn("reviewed", text)
        imported = self.store.csv_bundle(text)
        self.assertEqual(imported["records"], self.store.state["records"])
        reloaded = ReviewStore(fixture(), self.path)
        self.assertEqual(reloaded.state, self.store.state)

    def test_csv_does_not_silently_ignore_summary_edits(self):
        self.update(action="review", annotation=annotation())
        text = self.store.export_csv().replace(",reviewed,,,", ",reviewed,correct,,")
        with self.assertRaises(ValueError):
            self.store.csv_bundle(text)

    def test_write_failure_leaves_current_state_unchanged(self):
        before = self.store.snapshot()
        with patch("hebrew_acronyms.human_review_server.atomic_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.update(action="draft", annotation=annotation())
        self.assertEqual(before, self.store.state)

    def test_exposure_order_and_import_timestamps(self):
        with self.assertRaises(ValueError):
            self.update(action="expose", stage="responses")
        self.update(action="expose", stage="candidates")
        incoming = self.store.snapshot()
        incoming["records"]["fiction-1"]["exposure"]["candidates"] = ""
        with self.assertRaises(ValueError):
            self.store.import_bundle(incoming, self.store.state["revision"])

    def test_wrong_source_and_schema_do_not_mutate(self):
        before = self.store.snapshot()
        for field in ("dataset_id", "schema_version", "provenance"):
            incoming = copy.deepcopy(before)
            incoming[field] = "wrong"
            with self.assertRaises(ValueError):
                self.store.import_bundle(incoming, 0)
            self.assertEqual(before, self.store.state)

    def test_unattributed_annotation_is_rejected(self):
        a = annotation()
        a["annotator"] = " "
        with self.assertRaises(ValueError):
            self.update(action="draft", annotation=a)
        self.assertFalse(self.path.exists())


class HttpTests(unittest.TestCase):
    def test_loopback_http_asset_and_origin_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            server = make_server(fixture(), Path(folder) / "annotations.json", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/") as response:
                    self.assertIn(b'dir="rtl"', response.read())
                with urlopen(base + "/api/state") as response:
                    self.assertEqual(json.load(response)["records"], {})
                for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}):
                    with self.assertRaises(HTTPError) as raised:
                        urlopen(Request(base + "/api/state", headers=headers))
                    self.assertEqual(raised.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
