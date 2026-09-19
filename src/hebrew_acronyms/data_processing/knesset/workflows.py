"""Coordinate Knesset shard downloads and local sentence-table exports."""
from __future__ import annotations

import logging

from ..common import hebrew_text
from ..common.reporting import write_summary

LOG = logging.getLogger("hebrew_acronyms.data_processing")


def download_corpus(
    *,
    config: str,
    n: int,
    shards_dir: str,
) -> None:
    """Download missing shards and report the returned local paths."""
    from .source import download_shards

    paths = download_shards(config, n=n, out_dir=shards_dir)
    LOG.info("%d shards available in %s", len(paths), shards_dir)
    print(f"{len(paths)} shards -> {shards_dir}")


def collect_sentences(
    *,
    acronyms_path: str,
    shards_dir: str,
    out: str,
    per_acronym: int,
    summary: str | None,
) -> None:
    """Scan sorted local shards and export contexts with the existing limits."""
    import csv as csv_module
    from pathlib import Path

    from .source import mine_knesset

    acronyms = [
        hebrew_text.normalize_acronym(a.strip())
        for a in Path(acronyms_path).read_text(encoding="utf-8-sig").split()
        if a.strip()
    ]
    shard_dir = Path(shards_dir)
    shards = sorted(shard_dir.glob("*.jsonl.bz2"))
    if not shards:
        LOG.error("no shards found in %s — run `knesset-download` first", shard_dir)
        return
    LOG.info("scanning %d shards for %d acronym types", len(shards), len(acronyms))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    types_seen: set[str] = set()
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=["acronym", "context", "source", "page_title"])
        writer.writeheader()
        for ctx in mine_knesset(shards, acronyms, max_per_acronym=per_acronym):
            writer.writerow({
                "acronym": ctx.acronym, "context": ctx.context,
                "source": ctx.source, "page_title": ctx.page_title,
            })
            n_rows += 1
            types_seen.add(ctx.acronym)
    LOG.info("wrote %d rows -> %s", n_rows, out_path)
    write_summary(
        out,
        {"n_rows": n_rows, "n_types_selected": len(acronyms), "n_types_with_sentences": len(types_seen)},
        summary,
    )
