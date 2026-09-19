# Data processing reference

These modules retain the historical collection and construction steps. They can write
research data, contact live sources or download large corpora; use them only for an
explicitly authorized task. For safe local checks, use the [root README](../README.md#install-and-check).
[The data inventory](../data/README.md) separates source exports, review evidence and
split inputs; [the construction record](../data/mined/DATASET_CARD.md) explains their history.

## Responsibilities

| Component | Purpose |
|---|---|
| `__main__.py` | Argument parsing, logging and dispatch for the ten commands below. |
| `candidate_workflows.py` | Collect/merge candidate tables and apply duplicate review. |
| `sentence_workflows.py` | Select types, mine contexts and handle output/progress/resume. |
| `wikipedia/`, `wiktionary/`, `knesset/` | Source access and source-specific extraction; Knesset workflow also manages shards. |
| `common/` | Shared candidate representation, CSV reading, Hebrew matching, filters and reporting. |
| `merge_sources.py`, `dedupe_expansions.py`, `review_duplicates_cli.py` | Merge inventories, flag possible duplicates and collect interactive decisions. |
| `build_annotation_table.py` | Convert candidate/context tables to annotation rows. |
| `apply_dev_review.py` | Apply the saved historical development review. |
| `build_splits.py` | Historical Knesset allocation step; does not recreate all later additions. |

`common/candidates.py` owns candidate rows and candidate-table I/O; `common/csv_io.py`
reads general CSV rows. Specialized readers retain their own validation/conversion.
The abandoned Sefaria probe is recoverable from Git, as recorded in the dataset card.

## Command map

Invoke data commands from the repository root as
`.venv/bin/python -m data_preprocess <command>`. Append `--help` to inspect arguments;
providing real inputs is an execution step and may overwrite an output.

| Command | Action |
|---|---|
| `wikipedia` | Collect/count Wikipedia disambiguation candidates. |
| `wiktionary` | Collect/count Wiktionary senses. |
| `merge` | Merge candidate inventories. |
| `flag-duplicates` | Propose near-duplicate expansions for review. |
| `apply-review` | Apply recorded duplicate decisions. |
| `mine-sentences` | Export natural, unlabelled acronym contexts. |
| `mine-by-sense` | Export contexts for selected candidate senses; supports resume. |
| `build-annotation-table` | Assemble annotation rows from candidates and contexts. |
| `knesset-download` | Download selected corpus shards. |
| `knesset-mine` | Mine acronym contexts from local shards. |

The standalone modules `review_duplicates_cli`, `apply_dev_review` and `build_splits`
also have `python -m data_preprocess.<module> --help` interfaces. They remain because
they capture distinct manual-review or reconstruction steps, not alternate pipelines.

## Candidate table fields

| Field | Meaning |
|---|---|
| `acronym`, `expansion` | Normalized acronym surface and possible long form. |
| `page_title`, `raw_line` | Source location and original parsed line. |
| `hits` | Hebrew Wikipedia articles containing the expansion, including for Wiktionary senses. |
| `script`, `initials_match` | Script and heuristic acronym/expansion compatibility. |
| `looks_like_person` | Biographical-sense heuristic; not a verified label. |
| `source`, `domain` | Source inventory and optional Wiktionary register. |

Parser precision is unmeasured. Wiktionary definition trimming is heuristic, and phrase
hits do not count occurrences or cover every proclitic. Candidate evidence requires
interpretation before use as a scientific inventory.

## Preserved execution limitations

- Sense-mining resume can retain partial rows and re-mine the last type when earlier
  complete types exist; types with no output are not recorded as complete.
- Annotation context input uses plain UTF-8, while mining writers emit a BOM. Feeding
  such output directly can raise `KeyError('acronym')`; organization does not repair it.
- Numeral/honorific filtering and repeated-target handling retain the behavior described
  in the dataset card. No research filters or label rules were changed by cleanup.

The fixture suite compares commands, row order, CSV/JSON bytes, summaries and resume
against Git baseline `2b9b84eb20b0be89d728b963cc753924e83b53ea`, with temporary inputs
and network access blocked. Shared-reader checks also compare their pre-cleanup sources.
Keep the full history; do not use live mining as a smoke test.
