"""CLI for the acronym candidate tables.

    # Wikipedia disambiguation bullets -> bullet_counts.csv
    python -m data_preprocess wikipedia --out data/mined/wikipedia/bullet_counts.csv

    # Wiktionary senses -> wiktionary_counts.csv
    python -m data_preprocess wiktionary --out data/mined/wiktionary/wiktionary_counts.csv

    # union of the two -> merged_counts.csv
    python -m data_preprocess merge \
        --wikipedia data/mined/wikipedia/bullet_counts.csv \
        --wiktionary data/mined/wiktionary/wiktionary_counts.csv \
        --out data/mined/merged_counts.csv

Every command writes `<out>.summary.json` alongside its CSV. All three CSVs
share one column set, so they can be concatenated or diffed directly.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import candidate_workflows, sentence_workflows, build_annotation_table
from .knesset import workflows as knesset_workflows
from .wikipedia.parser import ACRONYM_DISAMBIG_CATEGORY
from .wiktionary.parser import ACRONYM_CATEGORY


def cmd_wikipedia(args: argparse.Namespace) -> None:
    candidate_workflows.collect_wikipedia(
        out=args.out,
        category=args.category,
        limit=args.limit,
        delay=args.delay,
        restart=args.restart,
        summary=args.summary,
    )


def cmd_wiktionary(args: argparse.Namespace) -> None:
    candidate_workflows.collect_wiktionary(
        out=args.out,
        category=args.category,
        limit=args.limit,
        min_senses=args.min_senses,
        delay=args.delay,
        restart=args.restart,
        summary=args.summary,
    )


def cmd_merge(args: argparse.Namespace) -> None:
    candidate_workflows.merge_tables(
        wikipedia_path=args.wikipedia,
        wiktionary_path=args.wiktionary,
        out=args.out,
        summary=args.summary,
    )


def cmd_flag_duplicates(args: argparse.Namespace) -> None:
    candidate_workflows.flag_duplicates(
        inp=args.inp,
        out=args.out,
        min_ratio=args.min_ratio,
    )


def cmd_apply_review(args: argparse.Namespace) -> None:
    candidate_workflows.apply_review(
        inp=args.inp,
        review=args.review,
        out=args.out,
    )


def cmd_mine_sentences(args: argparse.Namespace) -> None:
    sentence_workflows.collect_sentences(
        acronyms_path=args.acronyms,
        candidates_path=args.candidates,
        skip_types=args.skip_types,
        type_limit=args.types,
        min_second_hits=args.min_second_hits,
        delay=args.delay,
        pages_per_acronym=args.pages_per_acronym,
        per_acronym=args.per_acronym,
        no_wiktionary=args.no_wiktionary,
        out=args.out,
        summary=args.summary,
    )


def cmd_mine_by_sense(args: argparse.Namespace) -> None:
    sentence_workflows.collect_by_sense(
        acronyms_path=args.acronyms,
        candidates_path=args.candidates,
        delay=args.delay,
        out=args.out,
        resume=args.resume,
        pages_per_expansion=args.pages_per_expansion,
        per_expansion=args.per_expansion,
        no_substitution=args.no_substitution,
        summary=args.summary,
    )


def cmd_build_annotation_table(args: argparse.Namespace) -> None:
    build_annotation_table.build_table(
        candidates_path=args.candidates,
        contexts_path=args.contexts,
        out=args.out,
        summary=args.summary,
    )


def cmd_knesset_download(args: argparse.Namespace) -> None:
    knesset_workflows.download_corpus(
        config=args.config,
        n=args.n,
        shards_dir=args.shards_dir,
    )


def cmd_knesset_mine(args: argparse.Namespace) -> None:
    knesset_workflows.collect_sentences(
        acronyms_path=args.acronyms,
        shards_dir=args.shards_dir,
        out=args.out,
        per_acronym=args.per_acronym,
        summary=args.summary,
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
    w.add_argument("--out", default="data/mined/wikipedia/bullet_counts.csv")
    w.add_argument("--summary", default=None)
    w.add_argument("--category", default=ACRONYM_DISAMBIG_CATEGORY)
    w.add_argument("--limit", type=int, default=None, help="stop after N pages")
    w.add_argument("--delay", type=float, default=0.25)
    w.add_argument("--restart", action="store_true", help="ignore an existing CSV")
    w.set_defaults(func=cmd_wikipedia)

    k = sub.add_parser("wiktionary", help="count Wiktionary acronym senses")
    k.add_argument("--out", default="data/mined/wiktionary/wiktionary_counts.csv")
    k.add_argument("--summary", default=None)
    k.add_argument("--category", default=ACRONYM_CATEGORY)
    k.add_argument("--limit", type=int, default=None, help="stop after N pages")
    k.add_argument("--min-senses", type=int, default=1,
                   help="skip entries with fewer than N senses (2 = polysemous only)")
    k.add_argument("--delay", type=float, default=0.25)
    k.add_argument("--restart", action="store_true", help="ignore an existing CSV")
    k.set_defaults(func=cmd_wiktionary)

    m = sub.add_parser("merge", help="union the two source tables")
    m.add_argument("--wikipedia", default="data/mined/wikipedia/bullet_counts.csv")
    m.add_argument("--wiktionary", default="data/mined/wiktionary/wiktionary_counts.csv")
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

    kd = sub.add_parser("knesset-download", help="download Knesset Corpus shards for local mining")
    kd.add_argument("--config", default="plenary_protocols",
                     choices=["plenary_protocols", "committee_protocols"])
    kd.add_argument("--n", type=int, default=30, help="how many shards to fetch")
    kd.add_argument("--shards-dir", default="data/raw/knesset_shards")
    kd.set_defaults(func=cmd_knesset_download)

    km = sub.add_parser("knesset-mine", help="mine clean sentences from downloaded Knesset shards")
    km.add_argument("--acronyms", required=True,
                     help="file of acronym types (whitespace-separated) to mine")
    km.add_argument("--shards-dir", default="data/raw/knesset_shards")
    km.add_argument("--out", default="data/mined/knesset/knesset_mined.csv")
    km.add_argument("--summary", default=None)
    km.add_argument("--per-acronym", type=int, default=15, help="max sentences per acronym")
    km.set_defaults(func=cmd_knesset_mine)
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
