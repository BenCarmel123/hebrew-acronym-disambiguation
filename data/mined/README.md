# `data/mined/` — automatically extracted candidates, awaiting human review

**Layer 3 of the data lifecycle.** See [`../README.md`](../README.md) for all seven layers.

The output of the mining pipeline. Machine-produced, unreviewed, and **not labels**.

## What belongs here

- Candidate acronym types and their candidate long forms, extracted by a parser.
- Candidate contexts: natural mentions found in text, and reverse-substituted contexts
  built by replacing a long form with its acronym.
- Weak labels and heuristic verdicts, each carrying the rule that produced it.
- Extraction quality flags and parser review flags.

## The rule that matters

**Nothing here is a gold label.** A weak label is a hypothesis with a provenance record.
It becomes a label only after a person judges it against a written annotation guide, and
that happens under human review, not here.

The project has already been burned by this exact confusion: an automatically mined pool
of 196 acronym types was described as "196 validated types". It was a pool. The correction
is retained in `research/decision_evidence/`; it is evidence history, not a current decision.

## What may be committed

| | |
|---|---|
| ✅ | `README.md` (this file) |
| ✅ | Extraction output. Regenerating it is thousands of throttled API calls over hours, so it is committed — but it is still machine output, not data anyone has checked |

If an extraction cannot be regenerated because no script exists, that is a defect in the
extraction, not a reason to commit its output.

## Current state

**Empty.** The source-availability probe under
[`../../research/source_evidence/`](../../research/source_evidence/) established that a
large candidate pool is minable, and that in-context examples are scarce. It produced no
inventory and no labels. Read its README before quoting any count from it.

## The retry sweep

`retry_types.txt` holds 43 acronym types selected from the 92 that the original mining
swept but that produced no sentences. They were reviewed by hand in two passes,
recorded in `unmined_triage.csv` and `merge_review.csv`:

1. **Triage** — is this a usable type at all? Parser precision was never measured, so the
   list contains things that are not acronyms (`ערעור`), truncated parses
   (`מרדכי אהרן גינצבורג (1795`), and etymology notes captured as expansions.
2. **Merge** — several types were rejected only because their expansions are spellings of
   one sense rather than distinct readings; `שליט״א` carries four variants of one
   blessing. Grouping those decides whether two real senses survive.

Of the 80 marked minable across both passes, 37 were then filtered out: 26 with zero
corpus hits (deeper searching cannot help — the expansions do not occur in Hebrew
Wikipedia), 7 rabbinic-name types (`ריב״א`, `רש״ש`, `חרל״פ` …, whose candidates are
different rabbis sharing an initial — a knowledge task, not a context task), 3 that fell
under two senses once merged, and `גר'`, which carries no gershayim and is not an
acronym.

Only the top dozen or so have real corpus presence. Below ~10 hits the odds of any yield
are poor; they are included because they cost nothing extra in the same sweep.

    python -m data_preprocess mine-by-sense \
        --acronyms data/mined/retry_types.txt \
        --candidates data/mined/candidate_table.csv \
        --out data/mined/retry_by_sense.csv \
        --per-expansion 3 --pages-per-expansion 40 --resume

`--pages-per-expansion 40` against the 12 of the original mining: these types already
failed once at the shallower depth, so the retry only makes sense if it searches further.
`--resume` skips types already in the output, so an interrupted run continues rather than
restarting.
