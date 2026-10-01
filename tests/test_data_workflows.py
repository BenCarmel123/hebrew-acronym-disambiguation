"""Offline data-command contracts on invented temporary inputs and mocked sources."""
import argparse
import bz2
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import importlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TYPES = ('א״ב', 'ג״ד', 'ו״ז')
EXPANSIONS = ('אור בהיר', 'אור בחוץ', 'גן דשא', 'גן דק', 'ורד זהוב')
CANDIDATE_FIELDS = ["acronym", "page_title", "expansion", "hits", "script",
                    "initials_match", "looks_like_person", "source", "domain", "raw_line"]
CONTEXT_FIELDS = ["acronym", "context", "source", "page_title"]
SENSE_FIELDS = ["acronym", "expansion", "context", "source", "page_title", "provenance"]


def write_table(path, fields, rows, encoding="utf-8-sig"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate(acronym, expansion, hits, source="wikipedia"):
    return dict(acronym=acronym, page_title=acronym, expansion=expansion, hits=hits,
                script="hebrew", initials_match=True, looks_like_person=False,
                source=source, domain="", raw_line=expansion)


def sentence(term):
    return f"לאחר הישיבה הארוכה נמסר כי {term} ימשיך לפעול במקום גם במהלך השבוע הקרוב."


class DataWorkflowTests(unittest.TestCase):
    def setUp(self):
        # A missed fixture response must fail rather than contact a live source.
        self.network = ExitStack()
        self.addCleanup(self.network.close)
        for name in ("connect", "connect_ex"):
            self.network.enter_context(patch.object(
                socket.socket, name, side_effect=AssertionError("live network forbidden")))
        self.network.enter_context(patch(
            "requests.sessions.Session.request", side_effect=AssertionError("unmocked HTTP")))

    def exercise(self, scenario):
        cli = importlib.import_module("hebrew_acronyms.data_processing.__main__")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            client = importlib.import_module(cli.__package__ + ".wikipedia.client")
            def response(api, **params):
                calls.append((api.api_url, params))
                return self.wiki_response(api.api_url, params)
            with (patch.object(client.WikiAPI, "get", response),
                  patch("time.monotonic", return_value=100.0),
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                scenario(cli, root)
            return calls

    @staticmethod
    def invoke(cli, argv):
        args = cli.build_parser().parse_args([str(arg) for arg in argv])
        args.func(args)

    @staticmethod
    def wiki_response(url, params):
        is_wikt = "wiktionary" in url
        titles = [a.replace('״', '"') for a in TYPES[:2]]
        if params.get("list") == "categorymembers":
            return {"query": {"categorymembers": [{"title": a} for a in titles]}}
        if params.get("action") == "parse":
            return {"parse": {"wikitext": "* [[אור בהיר]]\n* [[אור בחוץ]]\n* [[אור בהיר]]"}}
        if params.get("prop") == "revisions":
            return {"query": {"pages": [
                {"title": title, "revisions": [{"slots": {"main": {"content":
                    "# [[אור בהיר]]\n# [[אור בחוץ]]\n#: " + sentence(title)}}}]}
                for title in params["titles"].split("|")]}}
        if params.get("list") == "search":
            if params.get("srinfo") == "totalhits":
                return {"query": {"searchinfo": {"totalhits": 37}}}
            return {"query": {"search": [{"title": "fixture article"},
                                           {"title": "fixture article"}]}}
        if params.get("prop") == "extracts":
            text = " ".join(sentence(term) for term in (*TYPES, *EXPANSIONS))
            return {"query": {"pages": [{"title": params["titles"], "extract": text}]}}
        raise AssertionError((is_wikt, params))

    @staticmethod
    def candidate_fixture(root):
        rows = [candidate(TYPES[0], EXPANSIONS[0], 60),
                candidate(TYPES[1], EXPANSIONS[2], 100),
                candidate(TYPES[0], EXPANSIONS[1], 30),
                candidate(TYPES[1], EXPANSIONS[3], 30),
                candidate(TYPES[2], EXPANSIONS[4], 200),
                candidate(TYPES[0], 'סיכת א״ב', 1)]
        write_table(root / "candidates.csv", CANDIDATE_FIELDS, rows)
        (root / "types.txt").write_text("א\"ב ג״ד ו״ז", encoding="utf-8-sig")
        return rows

    def test_imports_have_no_writes_or_network_in_fresh_process(self):
        # Audit hooks cover actual file/socket operations, including imports that
        # might otherwise evade a requests-only guard. -B disables bytecode writes.
        code = r'''
import importlib
import os
import sys
def audit(event, args):
    if event == "open":
        _, mode, flags = args
        if (mode and any(c in mode for c in "wax+")) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise AssertionError("write during import: " + repr(args))
    if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename",
                 "socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("side effect during import: " + event)
sys.addaudithook(audit)
for name in (
    "hebrew_acronyms.data_processing.__main__", "hebrew_acronyms.data_processing.candidate_workflows",
    "hebrew_acronyms.data_processing.sentence_workflows", "hebrew_acronyms.data_processing.knesset.workflows",
    "hebrew_acronyms.data_processing.build_annotation_table", "hebrew_acronyms.data_processing.common.reporting",
    "hebrew_acronyms.data_processing.common.candidates", "hebrew_acronyms.data_processing.wikipedia.source",
    "hebrew_acronyms.data_processing.wiktionary.source", "hebrew_acronyms.data_processing.dedupe_expansions",
    "hebrew_acronyms.data_processing.merge_sources",
):
    importlib.import_module(name)
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-c", code],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_candidate_count_resume_and_restart(self):
        for command in ("wikipedia", "wiktionary"):
            with self.subTest(command=command):
                def scenario(cli, root):
                    out = root / "out.csv"
                    args = [command, "--out", out, "--delay", "0", "--limit", "1"]
                    self.invoke(cli, args)
                    first = out.read_bytes()
                    self.invoke(cli, [command, "--out", out, "--delay", "0"])
                    resumed = out.read_bytes()
                    self.invoke(cli, [*args, "--restart", "--summary", root / "custom.json"])
                    self.assertEqual(out.read_bytes(), first)
                    self.assertGreater(len(resumed), len(first))
                    return first, resumed
                calls = self.exercise(scenario)
                hit_calls = [p for _, p in calls if p.get("srinfo") == "totalhits"]
                # The second page reuses cached counts; restart recomputes them.
                self.assertEqual(len(hit_calls), 4)

    def test_merge_flag_and_apply_review(self):
        def scenario(cli, root):
            rows = self.candidate_fixture(root)
            rows.extend([candidate(TYPES[0], "אור-בהיר", 90), candidate("A", "אות", 2)])
            write_table(root / "wiki.csv", CANDIDATE_FIELDS, rows)
            write_table(root / "wikt.csv", CANDIDATE_FIELDS,
                        [candidate(TYPES[1], EXPANSIONS[2], 10, "wiktionary")])
            sources = {name: (root / name).read_bytes() for name in ("wiki.csv", "wikt.csv")}
            self.invoke(cli, ["merge", "--wikipedia", root / "wiki.csv", "--wiktionary",
                              root / "wikt.csv", "--out", root / "merged.csv"])
            self.invoke(cli, ["flag-duplicates", "--in", root / "merged.csv", "--out",
                              root / "review.csv", "--min-ratio", "0.4"])
            with (root / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
                review = list(csv.DictReader(handle))
            self.assertTrue(review)
            review[0]["decision"] = "merge"
            write_table(root / "review.csv", list(review[0]), review)
            protected = {name: (root / name).read_bytes() for name in ("merged.csv", "review.csv")}
            self.invoke(cli, ["apply-review", "--in", root / "merged.csv", "--review",
                              root / "review.csv", "--out", root / "clean.csv"])
            with (root / "clean.csv").open(encoding="utf-8-sig", newline="") as handle:
                cleaned = list(csv.DictReader(handle))
            self.assertEqual([(row["acronym"], row["expansion"]) for row in cleaned],
                             [("א״ב", "אור-בהיר"), ("א״ב", "אור בחוץ"), ("א״ב", "סיכת א״ב"),
                              ("ג״ד", "גן דשא"), ("ו״ז", "ורד זהוב")])
            self.assertEqual(cleaned[3]["source"], "wikipedia+wiktionary")
            self.assertEqual(cleaned[3]["hits"], "100")
            for name, contents in {**sources, **protected}.items():
                self.assertEqual((root / name).read_bytes(), contents)
            return len(review)
        self.exercise(scenario)

    def test_empty_merge_and_missing_shards_do_not_create_output(self):
        def scenario(cli, root):
            self.invoke(cli, ["merge", "--wikipedia", root / "absent.csv", "--wiktionary",
                              root / "absent2.csv", "--out", root / "out.csv"])
            (root / "types.txt").write_text(TYPES[0], encoding="utf-8")
            self.invoke(cli, ["knesset-mine", "--acronyms", root / "types.txt",
                              "--shards-dir", root / "missing", "--out", root / "out.csv"])
            self.assertFalse((root / "out.csv").exists())
        self.exercise(scenario)

    def test_sentence_selection_order_and_explicit_types(self):
        def scenario(cli, root):
            self.candidate_fixture(root)
            base = ["mine-sentences", "--candidates", root / "candidates.csv", "--delay", "0",
                    "--per-acronym", "1", "--pages-per-acronym", "2"]
            self.invoke(cli, [*base, "--types", "1", "--out", root / "ranked.csv"])
            (root / "skip.txt").write_text(TYPES[1], encoding="utf-8-sig")
            self.invoke(cli, [*base, "--skip-types", root / "skip.txt", "--out", root / "skip.csv"])
            self.invoke(cli, [*base, "--acronyms", root / "types.txt", "--no-wiktionary",
                              "--out", root / "explicit.csv"])
            with (root / "ranked.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["acronym"], TYPES[1])  # reverse acronym tie break
            self.assertEqual(rows[-1]["source"], "wiktionary")
        self.exercise(scenario)

    def test_sense_resume_partial_type_preserves_existing_duplicate_behavior(self):
        for initial_types in ([], [TYPES[0]], [TYPES[0], TYPES[1]]):
            with self.subTest(initial_types=initial_types):
                def scenario(cli, root):
                    self.candidate_fixture(root)
                    out = root / "sense.csv"
                    prior = [dict(acronym=a, expansion="old", context="partial fixture",
                                  source="fixture", page_title="old", provenance="natural")
                             for a in initial_types]
                    if prior:
                        write_table(out, SENSE_FIELDS, prior)
                    self.invoke(cli, ["mine-by-sense", "--acronyms", root / "types.txt",
                                      "--candidates", root / "candidates.csv", "--out", out,
                                      "--per-expansion", "1", "--resume", "--delay", "0"])
                    with out.open(encoding="utf-8-sig") as handle:
                        rows = list(csv.DictReader(handle))
                    summary = json.loads(Path(str(out) + ".summary.json").read_text())
                    self.assertTrue(any(r["provenance"] == "substituted" for r in rows))
                    if len(prior) > 1:
                        self.assertEqual(rows[:len(prior)], prior)
                        self.assertTrue(any(r["acronym"] == initial_types[-1]
                                            for r in rows[len(prior):]))
                        self.assertEqual(summary["n_rows"], len(rows) - len(prior))
                    else:
                        self.assertFalse(any(r["context"] == "partial fixture" for r in rows))
                        self.assertEqual(summary["n_rows"], len(rows))
                    return len(rows)
                self.exercise(scenario)

    def test_sense_overwrite_and_no_substitution(self):
        def scenario(cli, root):
            self.candidate_fixture(root)
            out = root / "sense.csv"
            out.write_text("old content", encoding="utf-8")
            self.invoke(cli, ["mine-by-sense", "--acronyms", root / "types.txt",
                              "--candidates", root / "candidates.csv", "--out", out,
                              "--per-expansion", "1", "--no-substitution", "--delay", "0"])
            with out.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({r["provenance"] for r in rows}, {"natural"})
            self.assertNotIn('סיכת א״ב', {r["expansion"] for r in rows})
        self.exercise(scenario)

    def test_annotation_join_order_skips_and_summary(self):
        def scenario(cli, root):
            self.candidate_fixture(root)
            contexts = [dict(acronym=a, context=sentence(a), source="fixture", page_title=str(i))
                        for i, a in enumerate((TYPES[1], "absent", TYPES[0], TYPES[1]))]
            write_table(root / "contexts.csv", CONTEXT_FIELDS, contexts, encoding="utf-8")
            self.invoke(cli, ["build-annotation-table", "--candidates", root / "candidates.csv",
                              "--contexts", root / "contexts.csv", "--out", root / "annotation.csv"])
            with (root / "annotation.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["page_title"] for r in rows], ["0", "2", "3"])
            self.assertEqual(rows[1]["candidates"], 'אור בהיר|אור בחוץ|סיכת א״ב')
        self.exercise(scenario)

    def test_annotation_bom_input_preserves_existing_key_error(self):
        def scenario(cli, root):
            self.candidate_fixture(root)
            write_table(root / "contexts.csv", CONTEXT_FIELDS,
                        [dict(acronym=TYPES[0], context="fixture", source="fixture", page_title="1")])
            with self.assertRaises(KeyError) as error:
                self.invoke(cli, ["build-annotation-table", "--candidates", root / "candidates.csv",
                                  "--contexts", root / "contexts.csv", "--out", root / "out.csv"])
            self.assertEqual(error.exception.args, ("acronym",))
        self.exercise(scenario)

    def test_knesset_local_shard_order_limits_and_download_resume(self):
        def scenario(cli, root):
            (root / "types.txt").write_text(TYPES[0], encoding="utf-8-sig")
            shards = root / "shards"
            shards.mkdir()
            for name in ("z", "a"):
                data = dict(protocol_name=name, protocol_sentences=[
                    {"sentence_text": sentence(TYPES[0])}, {"sentence_text": "short"}])
                (shards / f"{name}.jsonl.bz2").write_bytes(
                    bz2.compress((json.dumps(data, ensure_ascii=False) + "\n").encode()))
            self.invoke(cli, ["knesset-mine", "--acronyms", root / "types.txt", "--shards-dir",
                              shards, "--out", root / "out.csv", "--per-acronym", "1"])
            with (root / "out.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["page_title"] for r in rows], ["a"])
            requests_seen = []
            class Response:
                content = b"invented compressed shard payload"
                def raise_for_status(self):
                    pass
                def json(self):
                    return [{"type": "file", "path": "fixture/one.jsonl.bz2"},
                            {"type": "file", "path": "fixture/two.jsonl.bz2"}]
            def get(url, **kwargs):
                requests_seen.append(url)
                return Response()
            with patch("requests.get", get):
                for _ in range(2):
                    self.invoke(cli, ["knesset-download", "--n", "1", "--shards-dir", root / "downloads"])
            self.assertEqual(sum("resolve/main" in url for url in requests_seen), 1)
            return requests_seen
        self.exercise(scenario)


if __name__ == "__main__":
    unittest.main()
