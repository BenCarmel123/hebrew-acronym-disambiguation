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
from pathlib import Path
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SCHEMA = "human-review-v1"
QUALITIES = {"", "correct", "wrong", "partial", "no_answer"}
PROBLEMS = {"valid", "ambiguous", "suspect_gold", "missing_candidate", "target_occurrence", "other"}
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
        self.state = {"schema_version": SCHEMA, "dataset_id": dataset["dataset_id"],
                      "provenance": dataset["provenance"], "revision": 0, "records": {}}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.validate_bundle(loaded)
            self.state = loaded

    def validate_annotation(self, item_id, annotation):
        if not isinstance(annotation, dict):
            raise ValueError("Invalid annotation")
        if annotation.get("schema_version") != SCHEMA:
            raise ValueError("Schema changed: review/migrate annotations explicitly")
        if not isinstance(annotation.get("annotator"), str) or not annotation["annotator"].strip():
            raise ValueError("Explicit annotator identity required")
        if annotation.get("prior_exposure", "unknown") not in {"unknown", "yes", "no"}:
            raise ValueError("Invalid prior exposure")
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

    def validate_bundle(self, bundle):
        if not isinstance(bundle, dict):
            raise ValueError("Invalid annotation bundle")
        if "revision" in bundle and (type(bundle["revision"]) is not int or bundle["revision"] < 0):
            raise ValueError("Invalid revision")
        if bundle.get("schema_version") != SCHEMA:
            raise ValueError("Incompatible schema version; annotations require explicit migration")
        for key in ("dataset_id", "provenance"):
            if bundle.get(key) != self.dataset.get(key):
                raise ValueError("Source provenance mismatch: " + key)
        if not isinstance(bundle.get("records"), dict):
            raise ValueError("Invalid records")
        for item_id, record in bundle["records"].items():
            if item_id not in self.items:
                raise ValueError("Unknown item ID: " + item_id)
            if not isinstance(record, dict):
                raise ValueError("Invalid record")
            if not isinstance(record.get("exposure", {}), dict):
                raise ValueError("Invalid exposure record")
            if not set(record.get("exposure", {})).issubset(EXPOSURES):
                raise ValueError("Unknown exposure stage")
            for timestamp in record.get("exposure", {}).values():
                if not isinstance(timestamp, str) or not timestamp:
                    raise ValueError("Invalid exposure timestamp")
                datetime.fromisoformat(timestamp)
            for status in ("draft", "reviewed"):
                if record.get(status) is not None:
                    self.validate_annotation(item_id, record[status])
                    if not isinstance(record[status].get("updated_at"), str) or not record[status]["updated_at"]:
                        raise ValueError("Annotation timestamp required")
                    datetime.fromisoformat(record[status]["updated_at"])

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _commit(self, candidate, action, item_id=None):
        candidate["revision"] = self.state["revision"] + 1
        candidate["saved_at"] = now()
        self.validate_bundle(candidate)
        # Append the changed record before atomic replacement. The current JSON
        # is authoritative; an event ahead of its revision marks an interrupted write.
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
            if action == "expose":
                stage = payload.get("stage")
                if stage not in EXPOSURES:
                    raise ValueError("Invalid exposure stage")
                exposure = record.setdefault("exposure", {})
                prerequisite = {"gold": "candidates", "responses": "gold", "identities": "responses", "auto_scores": "responses"}.get(stage)
                if prerequisite and prerequisite not in exposure:
                    raise ValueError("Reveal earlier stages first")
                if stage == "identities" and (not record.get("reviewed") or record.get("draft") != record.get("reviewed")):
                    raise ValueError("Save an explicit review before unblinding")
                exposure.setdefault(stage, now())
            elif action in {"draft", "review"}:
                annotation = copy.deepcopy(payload["annotation"])
                annotation["updated_at"] = now()
                self.validate_annotation(item_id, annotation)
                record["draft"] = annotation
                if action == "review":
                    record["reviewed"] = copy.deepcopy(annotation)
                    record["reviewed_at"] = now()
            else:
                raise ValueError("Unknown action")
            return self._commit(candidate, action, item_id)

    def import_bundle(self, bundle, revision):
        with self.lock:
            if revision != self.state["revision"]:
                raise ValueError("State changed. Reload before import.")
            self.validate_bundle(bundle)
            candidate = self.snapshot()
            for item_id, record in bundle["records"].items():
                record = copy.deepcopy(record)
                # Import can never erase recorded exposures.
                exposure = record.setdefault("exposure", {})
                for stage, timestamp in candidate["records"].get(item_id, {}).get("exposure", {}).items():
                    exposure[stage] = min(timestamp, exposure.get(stage, timestamp))
                candidate["records"][item_id] = record
            return self._commit(candidate, "import")

    def export_csv(self):
        output = io.StringIO(newline="")
        fields = ["dataset_id", "schema_version", "item_id", "answer_id", "system_id", "annotator", "updated_at", "status", "quality", "format_ok", "disagrees_auto", "provenance_json", "record_json"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item_id, record in self.snapshot()["records"].items():
            annotation = record.get("draft") or record.get("reviewed") or {}
            for answer in self.items[item_id]["answers"]:
                judgment = annotation.get("answers", {}).get(answer["id"], {})
                writer.writerow({"dataset_id": self.dataset["dataset_id"], "schema_version": SCHEMA,
                                 "item_id": item_id, "answer_id": answer["id"], "system_id": answer["system_id"],
                                 "annotator": annotation.get("annotator", ""), "updated_at": annotation.get("updated_at", ""),
                                 "status": "reviewed" if record.get("reviewed") and record.get("reviewed") == record.get("draft") else ("draft" if annotation else "exposure_only"),
                                 "quality": judgment.get("quality", ""), "format_ok": judgment.get("format_ok", ""),
                                 "disagrees_auto": judgment.get("disagrees_auto", ""),
                                 "provenance_json": json.dumps(self.dataset["provenance"], ensure_ascii=False),
                                 "record_json": json.dumps(record, ensure_ascii=False)})
        return "\ufeff" + output.getvalue()

    def csv_bundle(self, text):
        bundle = {"schema_version": SCHEMA, "dataset_id": self.dataset["dataset_id"], "provenance": self.dataset["provenance"], "records": {}}
        seen = set()
        for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
            if row["dataset_id"] != bundle["dataset_id"] or row["schema_version"] != SCHEMA or json.loads(row["provenance_json"]) != bundle["provenance"]:
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
            # record_json is the lossless, versioned payload; visible columns are
            # checked to prevent silently ignoring spreadsheet edits.
            annotation = record.get("draft") or record.get("reviewed") or {}
            judgment = annotation.get("answers", {}).get(answer["id"], {})
            expected_status = "reviewed" if record.get("reviewed") == record.get("draft") and record.get("reviewed") else ("draft" if annotation else "exposure_only")
            if row["status"] != expected_status:
                raise ValueError("CSV status does not match payload")
            for key in ("quality", "format_ok", "disagrees_auto"):
                if row[key] != judgment.get(key, ""):
                    raise ValueError("CSV summary columns edited; edit annotations in the review tool")
            if row["annotator"] != annotation.get("annotator", "") or row["updated_at"] != annotation.get("updated_at", ""):
                raise ValueError("CSV identity/time columns do not match payload")
            bundle["records"][item_id] = record
        for item_id in bundle["records"]:
            expected = {(item_id, a["id"]) for a in self.items[item_id]["answers"]}
            if not expected.issubset(seen):
                raise ValueError("CSV has missing answer rows")
        self.validate_bundle(bundle)
        return bundle


def make_server(dataset, annotations, port=8765, qa=False):
    store = ReviewStore(dataset, annotations)
    web = Path(__file__).with_name("human_review_web")

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
            if path == "/api/session":
                return self.reply({"qa": qa, "annotations_path": str(Path(annotations).resolve())})
            if path == "/api/data":
                return self.reply(dataset)
            if path == "/api/state":
                return self.reply(store.snapshot())
            if path == "/api/export.json":
                return self.reply(store.snapshot())
            if path == "/api/export.csv":
                return self.reply(store.export_csv(), "text/csv; charset=utf-8")
            names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path not in names:
                return self.reply({"error": "Not found"}, status=404)
            kind = {"/": "text/html", "/app.js": "text/javascript", "/style.css": "text/css"}[path]
            return self.reply((web / names[path]).read_bytes(), kind + "; charset=utf-8")

        def do_POST(self):
            if not self.allowed():
                return self.reply({"error": "Loopback origin required"}, status=403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 30_000_000:
                    raise ValueError("Import too large")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/api/update":
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
