"""Offline human review, with durable separate annotations and auditable exposure.

Run ``python -m hebrew_acronyms.human_review_server --data bundle.json
--annotations annotations.json``. No model execution or original-data writes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SCHEMA = "human-review-v2"
LEGACY_SCHEMA = "human-review-v1"
DECISIVE_QUALITIES = {"correct", "wrong", "partial", "undecidable", "no_answer"}
QUALITIES = {"", "legacy_partial"} | DECISIVE_QUALITIES
INTERPRETATION_KINDS = {"interpretation", "multiple", "no_context", "none", "needs_check"}
PROBLEMS = {"valid", "ambiguous", "suspect_gold", "missing_candidate", "target_occurrence", "other", "undecidable"}
EXPOSURES = {"candidates", "gold", "responses", "identities", "auto_scores"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ReviewStore:
    def __init__(self, dataset, path):
        self.dataset = copy.deepcopy(dataset)
        self.path = Path(path)
        self.lock = threading.RLock()
        self.items = {str(item["id"]): item for item in dataset["items"]}
        if len(self.items) != len(dataset["items"]):
            raise ValueError("Duplicate item IDs in dataset")
        for item in self.items.values():
            ids = [a["id"] for a in item["answers"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate answer IDs")
        self.state = self.empty_bundle()
        self.state["revision"] = 0
        self.refresh_summary(self.state)
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            prepared = self.prepare_bundle(loaded, recover_history=True)
            self.validate_bundle(prepared)
            if prepared != loaded:
                self.backup("before-migration")
                atomic_json(self.path, prepared)
            self.state = prepared

    def empty_bundle(self):
        return {"schema_version": SCHEMA, "dataset_id": self.dataset["dataset_id"],
                "source_identity": self.dataset.get("source_identity", self.dataset["dataset_id"]),
                "sampling_plan_id": self.dataset.get("sampling_plan_id"),
                "provenance": copy.deepcopy(self.dataset["provenance"]), "records": {}}

    def backup(self, reason):
        if self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copy2(self.path, self.path.with_name(self.path.name + "." + reason + "." + stamp + ".bak"))

    @staticmethod
    def manifest(provenance):
        if not isinstance(provenance, dict):
            raise ValueError("Invalid source provenance")
        files = provenance.get("files", [])
        if not isinstance(files, list):
            raise ValueError("Invalid source manifest")
        result = []
        for entry in files:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                    or not isinstance(entry.get("sha256"), str) or type(entry.get("bytes")) is not int):
                raise ValueError("Incomplete source manifest")
            result.append({key: entry[key] for key in ("path", "sha256", "bytes")})
        if len({entry["path"] for entry in result}) != len(result):
            raise ValueError("Duplicate source manifest path")
        return {"repository": provenance.get("repository"), "files": sorted(result, key=lambda entry: entry["path"])}

    def validate_sources(self, bundle):
        incoming = self.manifest(bundle.get("provenance"))
        expected = self.manifest(self.dataset["provenance"])
        if incoming != expected:
            raise ValueError("Source provenance mismatch: manifest")
        if not expected["files"] and bundle.get("dataset_id") != self.dataset["dataset_id"]:
            raise ValueError("Source provenance mismatch: dataset_id")
        if expected["files"] and not expected["repository"]:
            raise ValueError("Source repository identity required")

    def completion(self, item_id, record, requested=False):
        annotation = record.get("draft") or record.get("reviewed") or {}
        judged = sum(j.get("quality") in DECISIVE_QUALITIES for j in annotation.get("answers", {}).values())
        total = len(self.items[item_id]["answers"])
        item_decided = bool(annotation.get("item_problems"))
        status = "partial" if judged or item_decided or any(j.get("quality") for j in annotation.get("answers", {}).values()) else "draft"
        if requested and item_decided and judged == total:
            status = "complete"
        undecidable = sum(j.get("quality") == "undecidable" for j in annotation.get("answers", {}).values())
        return {"status": status, "judged_answers": judged, "decided_answers": judged,
                "undecidable_answers": undecidable, "semantic_judged_answers": judged - undecidable,
                "total_answers": total, "item_decided": item_decided}

    def normalize_record(self, item_id, record, requested=None):
        if requested is None:
            requested = record.get("completion", {}).get("status") == "complete"
        record["completion"] = self.completion(item_id, record, requested)
        annotation = record.get("draft") or record.get("reviewed") or {}
        record["needs_label_recheck"] = any(j.get("quality") == "legacy_partial" for j in annotation.get("answers", {}).values())

    def recover_initial(self, item_id, loaded):
        """Only an intact first-reveal event can establish a historical snapshot."""
        history_path = self.path.with_suffix(self.path.suffix + ".history.jsonl")
        if not history_path.exists():
            return None
        try:
            events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            previous_revision = 0
            for event in events:
                revision = event.get("revision")
                if (type(revision) is not int or revision != previous_revision + 1
                        or revision > loaded.get("revision", 0) or event.get("dataset_id") != loaded.get("dataset_id")):
                    return None
                previous_revision = revision
                record = event.get("records", {}).get(item_id, {})
                if "candidates" not in record.get("exposure", {}):
                    continue
                annotation = record.get("draft") or {}
                if (set(record.get("exposure", {})) != {"candidates"}
                        or event.get("action") != "expose" or event.get("item_id") != item_id
                        or not annotation.get("annotator") or not annotation.get("updated_at")
                        or annotation["updated_at"] > record["exposure"]["candidates"]):
                    return None
                return {"text": annotation.get("interpretation", ""), "kind": annotation.get("interpretation_kind", ""),
                        "annotator": annotation["annotator"], "captured_at": record["exposure"]["candidates"],
                        "availability": "recorded", "recovered_from": "first_candidate_exposure_history"}
        except (ValueError, TypeError, KeyError):
            return None
        return None

    def prepare_bundle(self, bundle, recover_history=False):
        if not isinstance(bundle, dict) or bundle.get("schema_version") not in {SCHEMA, LEGACY_SCHEMA}:
            raise ValueError("Incompatible schema version; annotations require explicit migration")
        self.validate_sources(bundle)
        prepared = copy.deepcopy(bundle)
        legacy = bundle["schema_version"] == LEGACY_SCHEMA
        if not isinstance(prepared.get("records"), dict):
            raise ValueError("Invalid records")
        for item_id, record in prepared["records"].items():
            if item_id not in self.items or not isinstance(record, dict):
                raise ValueError("Unknown item ID or invalid record: " + item_id)
            if legacy:
                record["migration"] = {"from_schema": LEGACY_SCHEMA, "original_dataset_id": bundle.get("dataset_id"),
                                       "original_provenance": copy.deepcopy(bundle["provenance"])}
                for status in ("draft", "reviewed"):
                    annotation = record.get(status)
                    if annotation:
                        annotation["schema_version"] = SCHEMA
                        if record.get("exposure"):
                            annotation.setdefault("updated_interpretation", annotation.get("interpretation", ""))
                        for judgment in annotation.get("answers", {}).values():
                            if judgment.get("quality") == "partial":
                                judgment["quality"] = "legacy_partial"
                                judgment["legacy_quality"] = "partial"
                            if "disagrees_auto" in judgment:
                                judgment["legacy_disagrees_auto"] = judgment["disagrees_auto"]
                        self.derive_judgments(item_id, annotation)
                if record.get("exposure"):
                    record["initial_interpretation"] = (self.recover_initial(item_id, bundle) if recover_history else None) or {
                        "availability": "unavailable", "text": "", "kind": "", "annotator": "", "captured_at": None,
                        "reason": "Legacy exposure has no unambiguous saved initial interpretation"}
                # Legacy review clicks were not completeness validation.
                record.pop("completion", None)
            self.normalize_record(item_id, record)
        for key, value in self.empty_bundle().items():
            if key != "records":
                prepared[key] = value
        self.refresh_summary(prepared)
        return prepared

    def derive_judgments(self, item_id, annotation):
        answers = {a["id"]: a for a in self.items[item_id]["answers"]}
        for answer_id, judgment in annotation.get("answers", {}).items():
            if answer_id not in answers:
                continue
            quality = judgment.get("quality")
            score = answers[answer_id].get("auto_score")
            judgment["disagrees_auto"] = ("yes" if (quality == "correct") != score else "no") if quality in {"correct", "wrong", "no_answer"} and type(score) is bool else "unknown"

    def validate_annotation(self, item_id, annotation):
        if not isinstance(annotation, dict):
            raise ValueError("Invalid annotation")
        if annotation.get("schema_version") != SCHEMA:
            raise ValueError("Schema changed: review/migrate annotations explicitly")
        if not isinstance(annotation.get("annotator"), str) or not annotation["annotator"].strip():
            raise ValueError("Explicit annotator identity required")
        if annotation.get("prior_exposure", "unknown") not in {"unknown", "yes", "no"}:
            raise ValueError("Invalid prior exposure")
        for field in ("interpretation", "updated_interpretation", "proposal_expansion", "proposed_alternatives"):
            if not isinstance(annotation.get(field, ""), str):
                raise ValueError("Invalid interpretation/proposal text")
        if not isinstance(annotation.get("item_problems", []), list) or not all(isinstance(p, str) for p in annotation.get("item_problems", [])):
            raise ValueError("Invalid item problem list")
        if not set(annotation.get("item_problems", [])).issubset(PROBLEMS):
            raise ValueError("Invalid item label")
        if "valid" in annotation.get("item_problems", []) and len(annotation["item_problems"]) > 1:
            raise ValueError("Valid cannot be combined with item problems")
        answers = {a["id"]: a for a in self.items[item_id]["answers"]}
        if not isinstance(annotation.get("answers", {}), dict):
            raise ValueError("Invalid answer judgments")
        for answer_id, judgment in annotation.get("answers", {}).items():
            if not isinstance(judgment, dict):
                raise ValueError("Invalid answer judgment")
            if answer_id not in answers or judgment.get("system_id") != answers[answer_id]["system_id"]:
                raise ValueError("Answer/system does not match the source item")
            if judgment.get("quality", "") not in QUALITIES:
                raise ValueError("Invalid answer label")
            for field in ("format_ok", "disagrees_auto"):
                if judgment.get(field, "") not in {"", "yes", "no", "unknown"}:
                    raise ValueError("Invalid answer flag")
            if judgment.get("disagreement_reason", "") not in {"", "gold", "protocol", "other"}:
                raise ValueError("Invalid disagreement reason")

    def validate_bundle(self, bundle):
        if not isinstance(bundle, dict):
            raise ValueError("Invalid annotation bundle")
        if "revision" in bundle and (type(bundle["revision"]) is not int or bundle["revision"] < 0):
            raise ValueError("Invalid revision")
        if bundle.get("schema_version") != SCHEMA:
            raise ValueError("Incompatible schema version")
        self.validate_sources(bundle)
        if not isinstance(bundle.get("records"), dict):
            raise ValueError("Invalid records")
        for item_id, record in bundle["records"].items():
            if item_id not in self.items or not isinstance(record, dict):
                raise ValueError("Unknown item ID or invalid record: " + item_id)
            if not isinstance(record.get("exposure", {}), dict) or not set(record.get("exposure", {})).issubset(EXPOSURES):
                raise ValueError("Invalid exposure record")
            for timestamp in record.get("exposure", {}).values():
                if not isinstance(timestamp, str) or not timestamp:
                    raise ValueError("Invalid exposure timestamp")
                datetime.fromisoformat(timestamp)
            initial = record.get("initial_interpretation")
            if record.get("exposure") and initial is None:
                raise ValueError("Exposed record must preserve initial interpretation or mark it unavailable")
            if initial is not None:
                if not isinstance(initial, dict) or initial.get("availability") not in {"recorded", "unavailable"}:
                    raise ValueError("Invalid initial interpretation snapshot")
                if initial["availability"] == "recorded":
                    if not all(isinstance(initial.get(key), str) for key in ("text", "kind", "annotator", "captured_at")) or not initial["annotator"].strip():
                        raise ValueError("Invalid initial interpretation identity or time")
                    datetime.fromisoformat(initial["captured_at"])
            for status in ("draft", "reviewed"):
                if record.get(status) is not None:
                    self.validate_annotation(item_id, record[status])
                    if not isinstance(record[status].get("updated_at"), str) or not record[status]["updated_at"]:
                        raise ValueError("Annotation timestamp required")
                    datetime.fromisoformat(record[status]["updated_at"])
            if record.get("completion") != self.completion(item_id, record, record.get("completion", {}).get("status") == "complete"):
                raise ValueError("Invalid completion counters")
            if record["completion"]["status"] == "complete" and record.get("draft") != record.get("reviewed"):
                raise ValueError("Complete record must match its explicitly reviewed snapshot")

    def refresh_summary(self, bundle):
        summary = {"complete_items": 0, "partial_items": 0, "draft_items": 0, "judged_answers": 0, "decided_answers": 0, "undecidable_answers": 0, "semantic_judged_answers": 0, "total_answers": 0}
        for record in bundle["records"].values():
            c = record["completion"]
            summary[c["status"] + "_items"] += 1
            for key in ("judged_answers", "decided_answers", "undecidable_answers", "semantic_judged_answers"):
                summary[key] += c[key]
            summary["total_answers"] += c["total_answers"]
        bundle["summary"] = summary
        bundle["migration_recheck_items"] = [item_id for item_id, record in bundle["records"].items() if record.get("needs_label_recheck")]

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            self.refresh_summary(result)
            return result

    def _commit(self, candidate, action, item_id=None):
        candidate["revision"] = self.state["revision"] + 1
        candidate["saved_at"] = now()
        self.refresh_summary(candidate)
        self.validate_bundle(candidate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {"time": now(), "action": action, "item_id": item_id, "revision": candidate["revision"],
                 "dataset_id": candidate["dataset_id"], "schema_version": SCHEMA,
                 "records": {item_id: candidate["records"][item_id]} if item_id else candidate["records"]}
        with self.path.with_suffix(self.path.suffix + ".history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(event, ensure_ascii=False) + "\n")
            history.flush()
            os.fsync(history.fileno())
        atomic_json(self.path, candidate)
        self.state = candidate
        return self.snapshot()

    def update(self, payload):
        with self.lock:
            if payload.get("revision") != self.state["revision"]:
                raise ValueError("Another window changed the annotations. Reload before saving.")
            item_id = str(payload.get("item_id"))
            if item_id not in self.items:
                raise ValueError("Unknown item ID")
            candidate = self.snapshot()
            record = candidate["records"].setdefault(item_id, {"exposure": {}})
            action = payload.get("action")
            if "initial_interpretation" in payload and payload["initial_interpretation"] != record.get("initial_interpretation"):
                raise ValueError("Initial interpretation snapshot is immutable")
            if action == "expose":
                stage = payload.get("stage")
                if stage not in EXPOSURES:
                    raise ValueError("Invalid exposure stage")
                exposure = record.setdefault("exposure", {})
                prerequisite = {"gold": "candidates", "responses": "gold", "identities": "responses", "auto_scores": "responses"}.get(stage)
                if prerequisite and prerequisite not in exposure:
                    raise ValueError("Reveal earlier stages first")
                if stage == "candidates" and "candidates" not in exposure:
                    annotation = record.get("draft") or {}
                    kind = annotation.get("interpretation_kind", annotation.get("initial_interpretation_kind"))
                    if (not annotation.get("annotator") or kind not in INTERPRETATION_KINDS
                            or (kind in {"interpretation", "multiple"} and not annotation.get("interpretation", "").strip())):
                        raise ValueError("Save an initial interpretation or an explicit no-interpretation/context decision before revealing candidates")
                    record["initial_interpretation"] = {"availability": "recorded", "text": annotation.get("interpretation", ""),
                                                        "kind": kind, "annotator": annotation["annotator"], "captured_at": now()}
                if stage == "identities" and record.get("completion", {}).get("status") != "complete":
                    raise ValueError("Complete the explicit review before unblinding")
                exposure.setdefault(stage, now())
                self.normalize_record(item_id, record)
            elif action in {"draft", "partial", "review", "complete"}:
                annotation = copy.deepcopy(payload["annotation"])
                annotation["updated_at"] = now()
                self.validate_annotation(item_id, annotation)
                self.derive_judgments(item_id, annotation)
                previous = record.get("draft") or {}
                semantic = lambda value: {key: val for key, val in value.items() if key != "updated_at"}
                unchanged_complete = record.get("completion", {}).get("status") == "complete" and semantic(previous) == semantic(annotation)
                if unchanged_complete:
                    annotation["updated_at"] = previous["updated_at"]
                record["draft"] = annotation
                self.normalize_record(item_id, record, requested=action in {"review", "complete"} or (action == "draft" and unchanged_complete))
                if action == "complete" and record["completion"]["status"] != "complete":
                    raise ValueError("Full completion requires an item decision and a judgment or explicit inability for every answer")
                if record["completion"]["status"] == "complete":
                    record["reviewed"] = copy.deepcopy(annotation)
                    record["reviewed_at"] = now()
            else:
                raise ValueError("Unknown action")
            return self._commit(candidate, action, item_id)

    def import_bundle(self, bundle, revision):
        with self.lock:
            if revision != self.state["revision"]:
                raise ValueError("State changed. Reload before import.")
            prepared = self.prepare_bundle(bundle)
            self.validate_bundle(prepared)
            candidate = self.snapshot()
            for item_id, incoming in prepared["records"].items():
                record = copy.deepcopy(incoming)
                existing = candidate["records"].get(item_id, {})
                old_initial = existing.get("initial_interpretation")
                new_initial = record.get("initial_interpretation")
                if old_initial is not None and new_initial is not None and old_initial != new_initial:
                    raise ValueError("Import cannot replace an immutable initial interpretation")
                if old_initial is not None:
                    record["initial_interpretation"] = copy.deepcopy(old_initial)
                exposure = record.setdefault("exposure", {})
                for stage, timestamp in existing.get("exposure", {}).items():
                    exposure[stage] = min(timestamp, exposure.get(stage, timestamp))
                if existing.get("reviewed") and not record.get("reviewed"):
                    record["reviewed"] = copy.deepcopy(existing["reviewed"])
                    record["reviewed_at"] = existing.get("reviewed_at")
                candidate["records"][item_id] = record
            self.validate_bundle(candidate)
            self.backup("before-import")
            return self._commit(candidate, "import")

    def export_csv(self):
        output = io.StringIO(newline="")
        fields = ["dataset_id", "source_identity", "sampling_plan_id", "schema_version", "item_id", "answer_id", "system_id", "annotator", "updated_at", "status", "judged_answers", "total_answers", "initial_interpretation", "initial_kind", "initial_availability", "initial_annotator", "initial_captured_at", "updated_interpretation", "quality", "format_ok", "disagrees_auto", "provenance_json", "record_json"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item_id, record in self.snapshot()["records"].items():
            annotation = record.get("draft") or record.get("reviewed") or {}
            initial = record.get("initial_interpretation") or {}
            for answer in self.items[item_id]["answers"]:
                judgment = annotation.get("answers", {}).get(answer["id"], {})
                writer.writerow({"dataset_id": self.dataset["dataset_id"], "source_identity": self.dataset.get("source_identity", self.dataset["dataset_id"]),
                                 "sampling_plan_id": self.dataset.get("sampling_plan_id"), "schema_version": SCHEMA,
                                 "item_id": item_id, "answer_id": answer["id"], "system_id": answer["system_id"],
                                 "annotator": annotation.get("annotator", ""), "updated_at": annotation.get("updated_at", ""),
                                 "status": record["completion"]["status"], "judged_answers": record["completion"]["judged_answers"], "total_answers": record["completion"]["total_answers"],
                                 "initial_interpretation": initial.get("text", ""), "initial_kind": initial.get("kind", ""),
                                 "initial_availability": initial.get("availability", ""), "initial_annotator": initial.get("annotator", ""),
                                 "initial_captured_at": initial.get("captured_at") or "", "updated_interpretation": annotation.get("updated_interpretation", ""),
                                 "quality": judgment.get("quality", ""), "format_ok": judgment.get("format_ok", ""),
                                 "disagrees_auto": judgment.get("disagrees_auto", ""),
                                 "provenance_json": json.dumps(self.dataset["provenance"], ensure_ascii=False),
                                 "record_json": json.dumps(record, ensure_ascii=False)})
        return "\ufeff" + output.getvalue()

    def csv_bundle(self, text):
        bundle = None
        seen = set()
        for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
            if bundle is None:
                bundle = {"schema_version": row["schema_version"], "dataset_id": row["dataset_id"], "provenance": json.loads(row["provenance_json"]), "records": {}}
                self.validate_sources(bundle)
            if row["dataset_id"] != bundle["dataset_id"] or row["schema_version"] != bundle["schema_version"] or json.loads(row["provenance_json"]) != bundle["provenance"]:
                raise ValueError("CSV provenance/schema mismatch")
            item_id = row["item_id"]
            if item_id not in self.items:
                raise ValueError("Unknown CSV item")
            answer = next((a for a in self.items[item_id]["answers"] if a["id"] == row["answer_id"]), None)
            if answer is None or answer["system_id"] != row["system_id"] or (item_id, answer["id"]) in seen:
                raise ValueError("CSV answer mapping or duplicate row error")
            seen.add((item_id, answer["id"]))
            record = json.loads(row["record_json"])
            if item_id in bundle["records"] and bundle["records"][item_id] != record:
                raise ValueError("Conflicting CSV item records")
            annotation = record.get("draft") or record.get("reviewed") or {}
            judgment = annotation.get("answers", {}).get(answer["id"], {})
            if bundle["schema_version"] == LEGACY_SCHEMA:
                expected_status = "reviewed" if record.get("reviewed") == record.get("draft") and record.get("reviewed") else ("draft" if annotation else "exposure_only")
            else:
                expected_status = record.get("completion", {}).get("status")
                initial = record.get("initial_interpretation") or {}
                expected = {"initial_interpretation": initial.get("text", ""), "initial_kind": initial.get("kind", ""), "initial_availability": initial.get("availability", ""), "initial_annotator": initial.get("annotator", ""), "initial_captured_at": initial.get("captured_at") or "", "updated_interpretation": annotation.get("updated_interpretation", ""), "judged_answers": str(record["completion"]["judged_answers"]), "total_answers": str(record["completion"]["total_answers"])}
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError("CSV interpretation/count columns do not match payload")
            if row["status"] != expected_status:
                raise ValueError("CSV status does not match payload")
            for key in ("quality", "format_ok", "disagrees_auto"):
                if row[key] != judgment.get(key, ""):
                    raise ValueError("CSV summary columns edited; edit annotations in the review tool")
            if row["annotator"] != annotation.get("annotator", "") or row["updated_at"] != annotation.get("updated_at", ""):
                raise ValueError("CSV identity/time columns do not match payload")
            bundle["records"][item_id] = record
        if bundle is None:
            bundle = self.empty_bundle()
        for item_id in bundle["records"]:
            expected = {(item_id, a["id"]) for a in self.items[item_id]["answers"]}
            if not expected.issubset(seen):
                raise ValueError("CSV has missing answer rows")
        prepared = self.prepare_bundle(bundle)
        self.validate_bundle(prepared)
        return prepared


def make_server(dataset, annotations, port=8765, qa=False):
    identified_protocol = dataset.get("short_protocol") == "identified-test-review-v1"
    continuation_protocol = dataset.get("short_protocol") == "qualitative-generation-v3"
    masked_protocol = identified_protocol or continuation_protocol or dataset.get("short_protocol") == "qualitative-generation-v2"
    store = None if masked_protocol else ReviewStore(dataset, annotations)
    web = Path(__file__).with_name("human_review_web")
    short = None
    if identified_protocol:
        from .human_review_identified import IdentifiedStore
        short = IdentifiedStore(dataset, annotations)
        if qa:
            short.state["reviewer"] = "QA_IDENTIFIED_NOT_HUMAN"
    elif continuation_protocol:
        from .human_review_masked import ContinuationStore
        previous_path = Path(annotations).with_name(Path(annotations).stem + ".short-v2.json")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_history = previous_path.with_suffix(".history.jsonl")
        short = ContinuationStore(dataset, Path(annotations).with_name(Path(annotations).stem + ".continuation-v1.json"),
                                  previous, previous_history.read_text(encoding="utf-8") if previous_history.exists() else "")
        if qa:
            short.state["reviewer"] = "QA_CONTINUATION_NOT_HUMAN"
    elif masked_protocol:
        from .human_review_masked import MaskedStore
        previous_path = Path(annotations).with_name(Path(annotations).stem + ".short-v1.json")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_history = previous_path.with_suffix(".history.jsonl")
        short = MaskedStore(dataset, Path(annotations).with_name(Path(annotations).stem + ".short-v2.json"),
                            previous, previous_history.read_text(encoding="utf-8") if previous_history.exists() else "")
        if qa:
            short.state["reviewer"] = "QA_MASKED_NOT_HUMAN"
    elif dataset.get("short_plan"):
        from .human_review_short import ShortStore
        legacy_history = Path(annotations).with_suffix(Path(annotations).suffix + ".history.jsonl")
        short = ShortStore(dataset, Path(annotations).with_name(Path(annotations).stem + ".short-v1.json"),
                           store.snapshot(), legacy_history.read_text() if legacy_history.exists() else "")
        if qa:
            short.state["reviewer"] = "QA_SHORT_NOT_HUMAN"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, body, kind="application/json; charset=utf-8", status=200):
            if not isinstance(body, bytes):
                body = (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            expected = "127.0.0.1:" + str(self.server.server_port)
            return host == expected and (not origin or origin == "http://" + expected)

        def do_GET(self):
            if not self.allowed():
                return self.reply({"error": "Loopback origin required"}, status=403)
            path = urlparse(self.path).path
            if masked_protocol and path in {"/api/data", "/api/state", "/api/export.json", "/api/export.csv", "/legacy", "/app.js", "/review_logic.js"}:
                return self.reply({"error": "המסלול המוסתר אינו מציג מידע מלא; חשיפה אפשרית רק בפעולת הסיום המפורשת."}, status=403)
            if masked_protocol and path in {"/api/short/full-export.json", "/api/short/full-summary.md"}:
                try:
                    from .human_review_scoring_audit import enrich_summary, scoring_markdown
                    if path.endswith(".json"):
                        result = short.export(masked=False)
                        result["summary"] = short.summary(masked=False) if identified_protocol else enrich_summary(short.summary(masked=False), dataset)
                        return self.reply(result)
                    summary = short.summary(masked=False) if identified_protocol else enrich_summary(short.summary(masked=False), dataset)
                    return self.reply(short.markdown(masked=False) + ("" if identified_protocol else scoring_markdown(summary)), "text/markdown; charset=utf-8")
                except ValueError as error:
                    return self.reply({"error": str(error)}, status=403)
            if path == "/api/short/state" and short:
                return self.reply(short.snapshot())
            if path == "/api/short/export.json" and short:
                return self.reply(short.export())
            if path == "/api/short/summary/export.md" and short:
                return self.reply(short.markdown(), "text/markdown; charset=utf-8")
            if path == "/api/session":
                return self.reply({"qa": qa, "annotations_path": str(Path(annotations).resolve()), "short_plan_id": dataset.get("continuation_plan" if continuation_protocol else "short_plan", {}).get("plan_id"), "short_protocol": dataset.get("short_protocol", "qualitative-generation-v1")})
            if path == "/api/data":
                return self.reply(dataset)
            if path == "/api/state":
                return self.reply(store.snapshot())
            if path == "/api/export.json":
                return self.reply(store.snapshot())
            if path == "/api/export.csv":
                return self.reply(store.export_csv(), "text/csv; charset=utf-8")
            names = {"/": "short.html" if short else "index.html", "/legacy": "index.html", "/short.js": "short.js", "/short.css": "short.css", "/app.js": "app.js", "/review_logic.js": "review_logic.js", "/style.css": "style.css"}
            if path not in names:
                return self.reply({"error": "Not found"}, status=404)
            kind = {"/": "text/html", "/legacy": "text/html", "/short.js": "text/javascript", "/short.css": "text/css", "/app.js": "text/javascript", "/review_logic.js": "text/javascript", "/style.css": "text/css"}[path]
            return self.reply((web / names[path]).read_bytes(), kind + "; charset=utf-8")

        def do_POST(self):
            if not self.allowed():
                return self.reply({"error": "Loopback origin required"}, status=403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 30_000_000:
                    raise ValueError("Import too large")
                payload = json.loads(self.rfile.read(length))
                if self.path.startswith("/api/short/") and short:
                    if self.path.rsplit("/", 1)[-1] in {"group-open", "group-save", "group-details"}:
                        raise ValueError("הבדיקה חזרה למשפטים נפרדים. יש לרענן את העמוד; העבודה השמורה נשמרה.")
                    result = short.transact(self.path.rsplit("/", 1)[-1], payload)
                    if masked_protocol and not identified_protocol and "summary" in result:
                        from .human_review_scoring_audit import enrich_summary
                        result["summary"] = enrich_summary(result["summary"], dataset)
                elif masked_protocol:
                    return self.reply({"error": "פעולה זו אינה זמינה במסלול המוסתר"}, status=403)
                elif self.path == "/api/update":
                    result = store.update(payload)
                elif self.path == "/api/import":
                    bundle = store.csv_bundle(payload["text"]) if payload["format"] == "csv" else json.loads(payload["text"])
                    result = store.import_bundle(bundle, payload.get("revision"))
                else:
                    return self.reply({"error": "Not found"}, status=404)
                self.reply(result)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.reply({"error": str(error)}, status=400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.store = store
    server.short_store = short
    if identified_protocol:
        def deadline_backup():
            import time
            while not short.stop_if_due():
                time.sleep(1)
        threading.Thread(target=deadline_backup, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--qa", action="store_true", help="Mark this isolated instance as engineering QA")
    args = parser.parse_args()
    if args.data.resolve() == args.annotations.resolve():
        parser.error("Annotations must be a separate file")
    dataset = json.loads(args.data.read_text(encoding="utf-8"))
    source_root = Path(dataset.get("provenance", {}).get("source_root", "."))
    protected = {(source_root / f["path"]).resolve() for f in dataset.get("provenance", {}).get("files", [])}
    outputs = {args.annotations.resolve(), args.annotations.with_suffix(args.annotations.suffix + ".history.jsonl").resolve()}
    if outputs & protected:
        parser.error("Annotations cannot overwrite a source")
    server = make_server(dataset, args.annotations, args.port, args.qa)
    print(f"Human review: http://127.0.0.1:{server.server_port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
