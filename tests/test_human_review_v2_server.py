"""V2 completion, immutable interpretation, migration and resampling regressions."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from hebrew_acronyms.human_review_server import ReviewStore, SCHEMA, now


def fixture():
    return {"dataset_id": "source-v2", "source_identity": "source-v2", "sampling_plan_id": "plan-20",
            "provenance": {"repository": "fixture-repository", "files": [
                {"path": "invented.csv", "sha256": "a" * 64, "bytes": 7}], "commit": "one"},
            "queues": {"calibration": ["one"], "evaluation": []},
            "items": [{"id": "one", "answers": [{"id": "a", "system_id": "A", "auto_score": True},
                                                   {"id": "b", "system_id": "B", "auto_score": False}]},
                      {"id": "two", "answers": [{"id": "c", "system_id": "A", "auto_score": True}]}]}


def annotation(**updates):
    result = {"schema_version": SCHEMA, "annotator": "QA", "interpretation": "first idea",
              "interpretation_kind": "interpretation", "item_problems": [], "answers": {}}
    result.update(updates)
    return result


def full():
    return annotation(item_problems=["valid"], answers={
        "a": {"system_id": "A", "quality": "correct"}, "b": {"system_id": "B", "quality": "wrong"}})


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "annotations.json"
        self.store = ReviewStore(fixture(), self.path)

    def tearDown(self):
        self.directory.cleanup()

    def update(self, action="draft", **kwargs):
        return self.store.update({"item_id": "one", "revision": self.store.state["revision"], "action": action, **kwargs})

    def record(self):
        return self.store.snapshot()["records"]["one"]

    def test_identity_only_and_revisit_never_complete(self):
        for a in (annotation(), annotation(revisit=True), annotation(item_problems=["valid"])):
            self.update("review", annotation=a)
            self.assertNotEqual("complete", self.record()["completion"]["status"])
            self.assertEqual(0, self.store.snapshot()["summary"]["complete_items"])
            self.assertNotIn("reviewed", self.record())
            before = self.store.snapshot()
            with self.assertRaisesRegex(ValueError, "Full completion"):
                self.update("complete", annotation=a)
            self.assertEqual(before, self.store.snapshot())

    def test_explicit_partial_complete_and_edit_downgrade(self):
        a = full()
        self.update("partial", annotation=a)
        self.assertEqual("partial", self.record()["completion"]["status"])
        self.update("complete", annotation=a)
        reviewed = copy.deepcopy(self.record()["reviewed"])
        self.assertEqual("complete", self.record()["completion"]["status"])
        self.update("draft", annotation=a)
        self.assertEqual("complete", self.record()["completion"]["status"])
        a["answers"]["b"]["quality"] = ""
        self.update(annotation=a)
        self.assertEqual("partial", self.record()["completion"]["status"])
        self.assertEqual(1, self.record()["completion"]["judged_answers"])
        self.assertEqual(reviewed, self.record()["reviewed"])
        self.assertEqual(0, self.store.snapshot()["summary"]["complete_items"])

    def test_explicit_inability_completes_without_semantic_guess(self):
        a = full()
        a["item_problems"] = ["undecidable"]
        a["answers"]["b"]["quality"] = "undecidable"
        self.update("complete", annotation=a)
        c = self.record()["completion"]
        self.assertEqual(("complete", 2, 1, 1), (c["status"], c["decided_answers"], c["semantic_judged_answers"], c["undecidable_answers"]))
        self.assertEqual("unknown", self.record()["draft"]["answers"]["b"]["disagrees_auto"])

    def test_initial_snapshot_required_immutable_and_roundtrips(self):
        with self.assertRaises(ValueError):
            self.update("expose", stage="candidates")
        self.update(annotation=annotation())
        self.update("expose", stage="candidates")
        initial = copy.deepcopy(self.record()["initial_interpretation"])
        self.update(annotation=annotation(interpretation="attempted overwrite", updated_interpretation="second idea"))
        self.assertEqual(initial, self.record()["initial_interpretation"])
        self.assertEqual("first idea", initial["text"])
        self.assertEqual("QA", initial["annotator"])
        self.assertEqual("recorded", initial["availability"])
        self.assertTrue(initial["captured_at"])
        roundtrip = self.store.csv_bundle(self.store.export_csv())
        self.assertEqual(self.store.snapshot()["records"], roundtrip["records"])
        other = ReviewStore(fixture(), Path(self.directory.name) / "other.json")
        other.import_bundle(roundtrip, 0)
        self.assertEqual(self.store.snapshot()["records"], other.snapshot()["records"])
        forged = self.store.snapshot()
        forged["records"]["one"]["initial_interpretation"]["text"] = "forged"
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.import_bundle(forged, self.store.state["revision"])
        self.assertEqual(initial, ReviewStore(fixture(), self.path).snapshot()["records"]["one"]["initial_interpretation"])

    def test_no_interpretation_or_no_context_explicit_choices(self):
        for kind in ("none", "no_context", "needs_check"):
            store = ReviewStore(fixture(), Path(self.directory.name) / (kind + ".json"))
            store.update({"revision": 0, "item_id": "one", "action": "draft", "annotation": annotation(interpretation="", interpretation_kind=kind)})
            result = store.update({"revision": 1, "item_id": "one", "action": "expose", "stage": "candidates"})
            self.assertEqual(kind, result["records"]["one"]["initial_interpretation"]["kind"])

    def test_separate_partial_and_undecidable_derived_disagreement(self):
        a = full()
        a["answers"]["a"]["quality"] = "wrong"
        self.update(annotation=a)
        self.assertEqual("yes", self.record()["draft"]["answers"]["a"]["disagrees_auto"])
        self.assertEqual("no", self.record()["draft"]["answers"]["b"]["disagrees_auto"])
        for quality in ("partial", "undecidable"):
            a["answers"]["a"]["quality"] = quality
            a["answers"]["a"]["disagrees_auto"] = "yes"
            self.update("complete", annotation=a)
            self.assertEqual(quality, self.record()["draft"]["answers"]["a"]["quality"])
            self.assertEqual("unknown", self.record()["draft"]["answers"]["a"]["disagrees_auto"])

    def legacy(self):
        a = full()
        a["schema_version"] = "human-review-v1"
        a["updated_at"] = now()
        a["answers"]["a"]["quality"] = "partial"
        return {"schema_version": "human-review-v1", "dataset_id": "old-sampling-dependent-id", "provenance": fixture()["provenance"],
                "revision": 2, "records": {"one": {"draft": a, "reviewed": copy.deepcopy(a), "exposure": {"candidates": now()}}}}

    def test_legacy_migration_preserves_combined_label_with_backup(self):
        legacy = self.legacy()
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = ReviewStore(fixture(), self.path)
        record = loaded.snapshot()["records"]["one"]
        self.assertEqual("legacy_partial", record["draft"]["answers"]["a"]["quality"])
        self.assertEqual("partial", record["draft"]["answers"]["a"]["legacy_quality"])
        self.assertEqual("partial", record["completion"]["status"])
        self.assertEqual("unavailable", record["initial_interpretation"]["availability"])
        self.assertEqual(["one"], loaded.snapshot()["migration_recheck_items"])
        backups = list(self.path.parent.glob("*.before-migration.*.bak"))
        self.assertEqual(1, len(backups))
        self.assertEqual(legacy, json.loads(backups[0].read_text()))

    def test_only_unambiguous_first_reveal_history_recovers_initial(self):
        legacy = self.legacy()
        first = copy.deepcopy(legacy["records"]["one"])
        first["draft"]["interpretation"] = "saved before reveal"
        first["reviewed"] = None
        events = [{"revision": 1, "action": "draft", "item_id": "one", "dataset_id": legacy["dataset_id"], "records": {"one": {"draft": first["draft"], "exposure": {}}}},
                  {"revision": 2, "action": "expose", "item_id": "one", "dataset_id": legacy["dataset_id"], "records": {"one": first}}]
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        self.path.with_suffix(".json.history.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        loaded = ReviewStore(fixture(), self.path)
        initial = loaded.snapshot()["records"]["one"]["initial_interpretation"]
        self.assertEqual("recorded", initial["availability"])
        self.assertEqual("saved before reveal", initial["text"])
        self.assertEqual("first idea", loaded.snapshot()["records"]["one"]["draft"]["updated_interpretation"])

    def test_truncated_or_later_exposure_history_is_not_independent(self):
        for revision, exposure in ((2, {"candidates": now()}), (1, {"candidates": now(), "gold": now()})):
            legacy = self.legacy()
            record = copy.deepcopy(legacy["records"]["one"])
            record["exposure"] = exposure
            record["draft"]["interpretation"] = "post exposure edit"
            event = {"revision": revision, "action": "expose", "item_id": "one", "dataset_id": legacy["dataset_id"], "records": {"one": record}}
            self.path.write_text(json.dumps(legacy), encoding="utf-8")
            self.path.with_suffix(".json.history.jsonl").write_text(json.dumps(event), encoding="utf-8")
            loaded = ReviewStore(fixture(), self.path)
            self.assertEqual("unavailable", loaded.snapshot()["records"]["one"]["initial_interpretation"]["availability"])

    def test_continuation_expanded_plan_preserves_records_and_exposures(self):
        self.update(annotation=annotation())
        self.update("expose", stage="candidates")
        self.update("complete", annotation=full())
        old = self.store.snapshot()
        expanded = fixture()
        expanded["sampling_plan_id"] = "plan-30"
        expanded["queues"]["calibration"].append("two")
        expanded["provenance"]["commit"] = "changed-builder-only"
        continued = ReviewStore(expanded, self.path)
        self.assertEqual(old["records"], continued.snapshot()["records"])
        self.assertEqual("plan-30", continued.snapshot()["sampling_plan_id"])
        continued.import_bundle(old, continued.state["revision"])
        self.assertEqual(old["records"], continued.snapshot()["records"])
        for key in ("sha256", "path", "bytes"):
            changed = copy.deepcopy(old)
            changed["provenance"]["files"][0][key] = 9 if key == "bytes" else "changed"
            with self.assertRaises(ValueError):
                continued.import_bundle(changed, continued.state["revision"])

    def test_legacy_csv_migration_and_provenance_validation(self):
        legacy = self.legacy()
        fields = ["dataset_id", "schema_version", "item_id", "answer_id", "system_id", "annotator", "updated_at", "status", "quality", "format_ok", "disagrees_auto", "provenance_json", "record_json"]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        record = legacy["records"]["one"]
        for answer in fixture()["items"][0]["answers"]:
            j = record["draft"]["answers"][answer["id"]]
            writer.writerow({"dataset_id": legacy["dataset_id"], "schema_version": legacy["schema_version"], "item_id": "one", "answer_id": answer["id"], "system_id": answer["system_id"], "annotator": "QA", "updated_at": record["draft"]["updated_at"], "status": "reviewed", "quality": j["quality"], "format_ok": "", "disagrees_auto": "", "provenance_json": json.dumps(legacy["provenance"]), "record_json": json.dumps(record)})
        prepared = self.store.csv_bundle(out.getvalue())
        self.assertEqual(SCHEMA, prepared["schema_version"])
        self.assertEqual("legacy_partial", prepared["records"]["one"]["draft"]["answers"]["a"]["quality"])
        self.store.import_bundle(prepared, 0)
        self.assertEqual("unavailable", self.record()["initial_interpretation"]["availability"])


if __name__ == "__main__":
    unittest.main()
