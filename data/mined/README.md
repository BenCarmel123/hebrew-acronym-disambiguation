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
