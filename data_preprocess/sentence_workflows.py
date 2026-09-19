"""Select acronym types and write mined contexts, including sense-run resumption."""
from __future__ import annotations

import csv as csv_module
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from .common import hebrew_text
from .common.reporting import write_summary
from .wikipedia.client import WikiAPI
from .wikipedia.mining import mine_by_expansion, mine_sentences, mine_substituted
from .wiktionary.mining import mine_wiktionary_sentences
from .wiktionary.parser import WIKTIONARY_API

LOG = logging.getLogger("data_preprocess")


def _hms(seconds: float) -> str:
    """Duration as h:mm:ss, for progress lines."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def select_acronyms(
    *, acronyms_path: str | None, candidates_path: str, skip_types: str | None,
    type_limit: int, min_second_hits: int,
) -> list[str]:
    """Use an explicit list or rank types by their second-highest hit count."""
    if acronyms_path:
        acronyms = [
            hebrew_text.normalize_acronym(a.strip())
            for a in Path(acronyms_path).read_text(encoding="utf-8-sig").split()
            if a.strip()
        ]
        LOG.info("mining %d types from %s", len(acronyms), acronyms_path)
    else:
        by_acronym: dict[str, list[int]] = defaultdict(list)
        with open(candidates_path, encoding="utf-8-sig", newline="") as fh:
            for rec in csv_module.DictReader(fh):
                by_acronym[rec["acronym"]].append(int(rec["hits"]))

        eligible = []
        for acronym, hits in by_acronym.items():
            if len(hits) < 2:
                continue
            ranked = sorted(hits, reverse=True)
            if ranked[1] < min_second_hits:
                continue
            eligible.append((ranked[1], acronym))
        eligible.sort(reverse=True)
        if skip_types:
            skip = {
                hebrew_text.normalize_acronym(a.strip())
                for a in Path(skip_types).read_text(encoding="utf-8-sig").split()
                if a.strip()
            }
            eligible = [(h, a) for h, a in eligible if a not in skip]
            LOG.info("skipping %d already-mined types", len(skip))
        acronyms = [a for _, a in eligible[: type_limit]]
        LOG.info("selected %d of %d eligible types", len(acronyms), len(eligible))

    return acronyms


def collect_sentences(
    *,
    acronyms_path: str | None,
    candidates_path: str,
    skip_types: str | None,
    type_limit: int,
    min_second_hits: int,
    delay: float,
    pages_per_acronym: int,
    per_acronym: int,
    no_wiktionary: bool,
    out: str,
    summary: str | None,
) -> None:
    """Mine clean prose sentences for a focused subset of acronym types.

    Types come either from `--acronyms` (an explicit list, used when a run
    re-visits types already known to yield) or, failing that, from the
    candidate table: types with at least two senses whose *second* sense is
    genuinely attested (`--min-second-hits`), so every selected type is really
    ambiguous in the corpus rather than nominally polysemous.

    Hit count is a weak proxy for yield — roughly two thirds of types selected
    that way produce nothing usable, because the "hits" are prefix collisions
    (`ב"שלום` for `ב"ש`) rather than the acronym. So once a run has measured
    a type's real yield, prefer feeding those types back in via `--acronyms`.
    """
    acronyms = select_acronyms(
        acronyms_path=acronyms_path, candidates_path=candidates_path,
        skip_types=skip_types, type_limit=type_limit, min_second_hits=min_second_hits,
    )

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
            WikiAPI(delay=delay),
            acronyms,
            pages_per_acronym=pages_per_acronym,
            max_per_acronym=per_acronym,
        )
        if no_wiktionary:
            return
        # Wiktionary page titles use a plain ASCII quote, not gershayim, so
        # both surface variants must be tried or most lookups miss silently.
        wikt = WikiAPI(api_url=WIKTIONARY_API, delay=delay)
        titles = sorted({v for a in acronyms for v in (hebrew_text.variants(a) or [a])})
        entries: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            entries.update(wikt.wikitext_batch(titles[i : i + 50]))
        yield from mine_wiktionary_sentences(entries)

    # Flush per row: the run makes hundreds of throttled API calls, and the
    # user checks the file mid-run rather than waiting for a final write.
    out_path = Path(out)
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
    write_summary(
        out,
        {
            "n_rows": n_rows,
            "n_types_selected": len(acronyms),
            "n_types_with_sentences": len(types_seen),
        },
        summary,
    )


def completed_types(out_path: Path, resume: bool) -> set[str]:
    """Recover completed types, retaining the historical partial-type behavior."""
    # A full sweep is thousands of throttled calls over hours, so an interrupted
    # run must not start over. Types already present in the output file are
    # skipped and the file is appended to. A type is only "done" if the run
    # advanced past it, so the last (possibly partial) type is re-mined —
    # cheaper than reasoning about how far into its expansions it got.
    done: set[str] = set()
    if resume and out_path.exists():
        with out_path.open(encoding="utf-8-sig", newline="") as fh:
            seen = [rec["acronym"] for rec in csv_module.DictReader(fh)]
        if seen:
            done = set(seen[:-1]) - {seen[-1]}
        LOG.info("resuming: %d types already complete", len(done))

    return done


def collect_by_sense(
    *,
    acronyms_path: str,
    candidates_path: str,
    delay: float,
    out: str,
    resume: bool,
    pages_per_expansion: int,
    per_expansion: int,
    no_substitution: bool,
    summary: str | None,
) -> None:
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
    types = [
        hebrew_text.normalize_acronym(a.strip())
        for a in Path(acronyms_path).read_text(encoding="utf-8-sig").split()
        if a.strip()
    ]
    candidates: dict[str, list[str]] = defaultdict(list)
    with open(candidates_path, encoding="utf-8-sig", newline="") as fh:
        for rec in csv_module.DictReader(fh):
            candidates[rec["acronym"]].append(rec["expansion"])

    api = WikiAPI(delay=delay)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["acronym", "expansion", "context", "source", "page_title", "provenance"]

    done = completed_types(out_path, resume)

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
                    pages_per_expansion=pages_per_expansion,
                    max_per_expansion=per_expansion,
                )
            ]
            if not no_substitution:
                strategies.append(
                    mine_substituted(
                        api, acronym, expansions,
                        pages_per_expansion=pages_per_expansion,
                        max_per_expansion=per_expansion,
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
    write_summary(out, {"n_rows": n_rows, "n_types": len(types)}, summary)
