"""CLI for the acronym candidate tables.

    # Wikipedia disambiguation bullets -> bullet_counts.csv
    python -m data_preprocess wikipedia --out data/mined/bullet_counts.csv

    # Wiktionary senses -> wiktionary_counts.csv
    python -m data_preprocess wiktionary --out data/mined/wiktionary_counts.csv

    # union of the two -> merged_counts.csv
    python -m data_preprocess merge \
        --wikipedia data/mined/bullet_counts.csv \
        --wiktionary data/mined/wiktionary_counts.csv \
        --out data/mined/merged_counts.csv

Every command writes `<out>.summary.json` alongside its CSV. All three CSVs
share one column set, so they can be concatenated or diffed directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .wikipedia_source import (
    ACRONYM_DISAMBIG_CATEGORY,
    count_bullets,
    read_existing,
    summarise,
    write_csv,
)
from .dedupe_expansions import (
    apply_review_decisions,
    dedupe_expansions,
    drop_single_letter_acronyms,
    find_near_duplicates,
    write_review_csv,
)
from .build_annotation_table import build_rows, load_candidates, write_annotation_table
from .wiktionary_source import count_senses, fetch_entries
from .merge_sources import merge_rows, merge_summary
from . import hebrew_text
from .mine_sentences import (
    mine_by_expansion,
    mine_sentences,
    mine_substituted,
    mine_wiktionary_sentences,
)
from .wiki_client import API_URL, WikiAPI
from .wiktionary_parser import ACRONYM_CATEGORY, WIKTIONARY_API

LOG = logging.getLogger("data_preprocess")


def _write_summary(out: str, summary: dict, explicit: str | None = None) -> None:
    path = Path(explicit or f"{out}.summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("summary -> %s", path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_wikipedia(args: argparse.Namespace) -> None:
    api = WikiAPI(delay=args.delay)
    existing, done, cache = ([], set(), {}) if args.restart else read_existing(args.out)
    rows = write_csv(
        args.out,
        count_bullets(api, args.category, limit=args.limit, cache=cache, skip_titles=done),
        existing=existing,
    )
    _write_summary(args.out, summarise(rows), args.summary)


def cmd_wiktionary(args: argparse.Namespace) -> None:
    wikt = WikiAPI(api_url=WIKTIONARY_API, delay=args.delay)
    wiki = WikiAPI(api_url=API_URL, delay=args.delay)
    existing, done, cache = ([], set(), {}) if args.restart else read_existing(args.out)
    texts = fetch_entries(wikt, args.category, limit=args.limit)
    rows = write_csv(
        args.out,
        count_senses(
            wiki, texts, min_senses=args.min_senses, cache=cache, skip_titles=done
        ),
        existing=existing,
    )
    _write_summary(args.out, summarise(rows), args.summary)


def cmd_merge(args: argparse.Namespace) -> None:
    wikipedia, _, _ = read_existing(args.wikipedia)
    wiktionary, _, _ = read_existing(args.wiktionary)
    if not wikipedia and not wiktionary:
        LOG.error("both input tables are empty; nothing to merge")
        return
    LOG.info("merging %d Wikipedia + %d Wiktionary rows", len(wikipedia), len(wiktionary))
    rows = merge_rows(wikipedia, wiktionary)
    rows = dedupe_expansions(rows)
    rows = drop_single_letter_acronyms(rows)
    write_csv(args.out, iter(()), existing=rows)
    _write_summary(args.out, merge_summary(rows), args.summary)


def cmd_flag_duplicates(args: argparse.Namespace) -> None:
    rows, _, _ = read_existing(args.inp)
    flagged = find_near_duplicates(rows, min_ratio=args.min_ratio)
    write_review_csv(args.out, flagged)


def cmd_apply_review(args: argparse.Namespace) -> None:
    rows, _, _ = read_existing(args.inp)
    cleaned = apply_review_decisions(rows, args.review)
    write_csv(args.out, iter(()), existing=cleaned)


def cmd_mine_sentences(args: argparse.Namespace) -> None:
    """Mine clean prose sentences for a focused subset of acronym types.

    Types come either from `--acronyms` (an explicit list, used when a batch
    is a re-run of types already known to yield) or, failing that, from the
    candidate table: types with at least two senses whose *second* sense is
    genuinely attested (`--min-second-hits`), so every selected type is really
    ambiguous in the corpus rather than nominally polysemous.

    Hit count is a weak proxy for yield — roughly two thirds of types selected
    that way produce nothing usable, because the "hits" are prefix collisions
    (`ב"שלום` for `ב"ש`) rather than the acronym. So once a batch has measured
    a type's real yield, prefer feeding those types back in via `--acronyms`.
    """
    import csv as csv_module
    from collections import defaultdict
    from pathlib import Path

    if args.acronyms:
        acronyms = [
            hebrew_text.normalize_acronym(a.strip())
            for a in Path(args.acronyms).read_text(encoding="utf-8-sig").split()
            if a.strip()
        ]
        LOG.info("mining %d types from %s", len(acronyms), args.acronyms)
    else:
        by_acronym: dict[str, list[int]] = defaultdict(list)
        with open(args.candidates, encoding="utf-8-sig", newline="") as fh:
            for rec in csv_module.DictReader(fh):
                by_acronym[rec["acronym"]].append(int(rec["hits"]))

        eligible = []
        for acronym, hits in by_acronym.items():
            if len(hits) < 2:
                continue
            ranked = sorted(hits, reverse=True)
            if ranked[1] < args.min_second_hits:
                continue
            eligible.append((ranked[1], acronym))
        eligible.sort(reverse=True)
        if args.skip_types:
            skip = {
                hebrew_text.normalize_acronym(a.strip())
                for a in Path(args.skip_types).read_text(encoding="utf-8-sig").split()
                if a.strip()
            }
            eligible = [(h, a) for h, a in eligible if a not in skip]
            LOG.info("skipping %d already-mined types", len(skip))
        acronyms = [a for _, a in eligible[: args.types]]
        LOG.info("selected %d of %d eligible types", len(acronyms), len(eligible))

    def as_row(ctx) -> dict:
        return {
            "acronym": ctx.acronym,
            "context": ctx.context,
            "source": ctx.source,
            "page_title": ctx.page_title,
        }

    def all_contexts():
        """Wikipedia sentences, then Wiktionary usage examples, as one stream."""
        yield from mine_sentences(
            WikiAPI(delay=args.delay),
            acronyms,
            pages_per_acronym=args.pages_per_acronym,
            max_per_acronym=args.per_acronym,
        )
        if args.no_wiktionary:
            return
        # Wiktionary page titles use a plain ASCII quote, not gershayim, so
        # both surface variants must be tried or most lookups miss silently.
        wikt = WikiAPI(api_url=WIKTIONARY_API, delay=args.delay)
        titles = sorted({v for a in acronyms for v in (hebrew_text.variants(a) or [a])})
        entries: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            entries.update(wikt.wikitext_batch(titles[i : i + 50]))
        yield from mine_wiktionary_sentences(entries)

    # Flush per row: the run makes hundreds of throttled API calls, and the
    # user checks the file mid-run rather than waiting for a final write.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    types_seen: set[str] = set()
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=["acronym", "context", "source", "page_title"])
        writer.writeheader()
        fh.flush()
        for ctx in all_contexts():
            writer.writerow(as_row(ctx))
            fh.flush()
            n_rows += 1
            types_seen.add(ctx.acronym)
    LOG.info("wrote %d sentences -> %s", n_rows, out_path)
    _write_summary(
        args.out,
        {
            "n_rows": n_rows,
            "n_types_selected": len(acronyms),
            "n_types_with_sentences": len(types_seen),
        },
        args.summary,
    )


def _hms(seconds: float) -> str:
    """Duration as h:mm:ss, for progress lines."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def cmd_mine_by_sense(args: argparse.Namespace) -> None:
    """Mine sentences tied to a specific expansion, for a list of types.

    Runs both sense-aware strategies per type: `mine_by_expansion` for natural
    acronym usage, then `mine_substituted` for rewritten full forms. Together
    they answer the question flat mining cannot — whether a type has more than
    one sense actually attested — because flat mining returns whichever sense
    dominates and records no sense at all.

    Rows carry `provenance`, which must survive into any split: natural and
    substituted items are different distributions, and keeping them separable
    is what allows checking whether a model behaves differently on each.
    """
    import csv as csv_module
    import time
    from collections import defaultdict
    from pathlib import Path

    types = [
        hebrew_text.normalize_acronym(a.strip())
        for a in Path(args.acronyms).read_text(encoding="utf-8-sig").split()
        if a.strip()
    ]
    candidates: dict[str, list[str]] = defaultdict(list)
    with open(args.candidates, encoding="utf-8-sig", newline="") as fh:
        for rec in csv_module.DictReader(fh):
            candidates[rec["acronym"]].append(rec["expansion"])

    api = WikiAPI(delay=args.delay)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["acronym", "expansion", "context", "source", "page_title", "provenance"]

    # A full sweep is thousands of throttled calls over hours, so an interrupted
    # run must not start over. Types already present in the output file are
    # skipped and the file is appended to. A type is only "done" if the run
    # advanced past it, so the last (possibly partial) type is re-mined —
    # cheaper than reasoning about how far into its expansions it got.
    done: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8-sig", newline="") as fh:
            seen = [rec["acronym"] for rec in csv_module.DictReader(fh)]
        if seen:
            done = set(seen[:-1]) - {seen[-1]}
        LOG.info("resuming: %d types already complete", len(done))

    n_rows = 0
    mined = 0
    run_started = time.monotonic()
    mode = "a" if done else "w"
    # Flush per row: the run makes thousands of throttled calls, and the file
    # is read mid-run rather than waited on.
    with out_path.open(mode, encoding="utf-8-sig", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=fields)
        if not done:
            writer.writeheader()
        fh.flush()
        for i, acronym in enumerate(types, 1):
            if acronym in done:
                continue
            # An "expansion" that itself contains the acronym (סיכת מ"מ) is a
            # phrase named after it, not a reading of it — there is nothing to
            # substitute into, and every sentence would be rejected anyway.
            expansions = [
                e for e in candidates.get(acronym, []) if '"' not in e and "״" not in e
            ]
            LOG.info("[%d/%d] %s — %d expansions", i, len(types), acronym, len(expansions))
            before = n_rows
            started = time.monotonic()
            strategies = [
                mine_by_expansion(
                    api, acronym, expansions,
                    pages_per_expansion=args.pages_per_expansion,
                    max_per_expansion=args.per_expansion,
                )
            ]
            if not args.no_substitution:
                strategies.append(
                    mine_substituted(
                        api, acronym, expansions,
                        pages_per_expansion=args.pages_per_expansion,
                        max_per_expansion=args.per_expansion,
                    )
                )
            for strategy in strategies:
                for row in strategy:
                    writer.writerow({
                        "acronym": row.acronym,
                        "expansion": row.expansion,
                        "context": row.context,
                        "source": row.source,
                        "page_title": row.page_title,
                        "provenance": row.provenance,
                    })
                    fh.flush()
                    n_rows += 1
            # Progress on stderr, separate from the log: a full sweep runs for
            # hours, and the useful question mid-run is how much is left, which
            # the per-type log lines do not answer. Rate is measured over types
            # actually mined this session, so a resumed run does not divide by
            # work it skipped.
            elapsed = time.monotonic() - run_started
            mined += 1
            remaining = len(types) - i
            eta = (elapsed / mined) * remaining if mined else 0.0
            print(
                f"[{i}/{len(types)}] {acronym}  "
                f"{n_rows - before:2d} rows  "
                f"{time.monotonic() - started:4.1f}s  "
                f"| total {n_rows} rows  "
                f"elapsed {_hms(elapsed)}  eta {_hms(eta)}",
                file=sys.stderr, flush=True,
            )
            LOG.info("[%d/%d] %s — %d rows", i, len(types), acronym, n_rows - before)
    LOG.info("wrote %d rows -> %s", n_rows, out_path)
    _write_summary(args.out, {"n_rows": n_rows, "n_types": len(types)}, args.summary)


def cmd_build_annotation_table(args: argparse.Namespace) -> None:
    import csv as csv_module

    candidates = load_candidates(args.candidates)
    with open(args.contexts, encoding="utf-8", newline="") as fh:
        contexts = list(csv_module.DictReader(fh))
    rows = build_rows(candidates, contexts)
    write_annotation_table(args.out, rows)
    _write_summary(
        args.out,
        {"n_rows": len(rows), "n_acronyms": len({r["acronym"] for r in rows})},
        args.summary,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_preprocess",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("wikipedia", help="count Wikipedia disambiguation bullets")
    w.add_argument("--out", default="data/mined/bullet_counts.csv")
    w.add_argument("--summary", default=None)
    w.add_argument("--category", default=ACRONYM_DISAMBIG_CATEGORY)
    w.add_argument("--limit", type=int, default=None, help="stop after N pages")
    w.add_argument("--delay", type=float, default=0.25)
    w.add_argument("--restart", action="store_true", help="ignore an existing CSV")
    w.set_defaults(func=cmd_wikipedia)

    k = sub.add_parser("wiktionary", help="count Wiktionary acronym senses")
    k.add_argument("--out", default="data/mined/wiktionary_counts.csv")
    k.add_argument("--summary", default=None)
    k.add_argument("--category", default=ACRONYM_CATEGORY)
    k.add_argument("--limit", type=int, default=None, help="stop after N pages")
    k.add_argument("--min-senses", type=int, default=1,
                   help="skip entries with fewer than N senses (2 = polysemous only)")
    k.add_argument("--delay", type=float, default=0.25)
    k.add_argument("--restart", action="store_true", help="ignore an existing CSV")
    k.set_defaults(func=cmd_wiktionary)

    m = sub.add_parser("merge", help="union the two source tables")
    m.add_argument("--wikipedia", default="data/mined/bullet_counts.csv")
    m.add_argument("--wiktionary", default="data/mined/wiktionary_counts.csv")
    m.add_argument("--out", default="data/mined/merged_counts.csv")
    m.add_argument("--summary", default=None)
    m.set_defaults(func=cmd_merge)

    fd = sub.add_parser("flag-duplicates", help="flag near-duplicate expansions for human review")
    fd.add_argument("--in", dest="inp", default="data/mined/merged_counts.csv")
    fd.add_argument("--out", default="data/mined/duplicate_review.csv")
    fd.add_argument("--min-ratio", type=float, default=0.75)
    fd.set_defaults(func=cmd_flag_duplicates)

    ar = sub.add_parser("apply-review", help="drop rows marked 'merge' in a filled-in review CSV")
    ar.add_argument("--in", dest="inp", default="data/mined/merged_counts.csv")
    ar.add_argument("--review", default="data/mined/duplicate_review.csv")
    ar.add_argument("--out", default="data/mined/merged_counts.clean.csv")
    ar.set_defaults(func=cmd_apply_review)


    ms = sub.add_parser("mine-sentences", help="mine clean prose sentences for a focused type subset")
    ms.add_argument("--candidates", default="data/mined/candidate_table.csv")
    ms.add_argument("--out", default="data/mined/mined_sentences.csv")
    ms.add_argument("--summary", default=None)
    ms.add_argument("--types", type=int, default=40, help="how many acronym types to mine")
    ms.add_argument("--acronyms", default=None,
                    help="file of acronym types (whitespace-separated) to mine instead of "
                         "selecting from the candidate table by hit count")
    ms.add_argument("--skip-types", default=None,
                    help="file of acronym types to exclude from candidate-table selection")
    ms.add_argument("--min-second-hits", type=int, default=20,
                    help="require the 2nd sense to have at least N Wikipedia hits")
    ms.add_argument("--per-acronym", type=int, default=12, help="max sentences per acronym")
    ms.add_argument("--pages-per-acronym", type=int, default=30, help="max pages to search per acronym")
    ms.add_argument("--delay", type=float, default=0.25)
    ms.add_argument("--no-wiktionary", action="store_true",
                    help="skip Wiktionary usage examples")
    ms.set_defaults(func=cmd_mine_sentences)

    mbs = sub.add_parser("mine-by-sense",
                         help="mine sentences tied to each expansion (natural + substituted)")
    mbs.add_argument("--acronyms", required=True,
                     help="file of acronym types (whitespace-separated) to mine")
    mbs.add_argument("--candidates", default="data/mined/candidate_table.csv")
    mbs.add_argument("--out", default="data/mined/by_sense.csv")
    mbs.add_argument("--summary", default=None)
    mbs.add_argument("--per-expansion", type=int, default=3,
                     help="max sentences per expansion, per strategy")
    mbs.add_argument("--pages-per-expansion", type=int, default=12,
                     help="max pages to fetch per expansion")
    mbs.add_argument("--no-substitution", action="store_true",
                     help="natural acronym usage only; skip rewritten full forms")
    mbs.add_argument("--resume", action="store_true",
                     help="skip types already present in --out and append to it")
    mbs.add_argument("--delay", type=float, default=0.3)
    mbs.set_defaults(func=cmd_mine_by_sense)

    bat = sub.add_parser("build-annotation-table", help="join candidates with mined contexts for annotation")
    bat.add_argument("--candidates", default="data/mined/candidate_table.csv")
    bat.add_argument("--contexts", default="data/mined/mined_contexts.csv")
    bat.add_argument("--out", default="data/mined/annotation_table.csv")
    bat.add_argument("--summary", default=None)
    bat.set_defaults(func=cmd_build_annotation_table)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
