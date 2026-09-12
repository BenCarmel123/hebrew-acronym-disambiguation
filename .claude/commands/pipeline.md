---
description: Validate a train/dev split and run every eval arm against it, printing the combined results table
---

## Prerequisites

- Python venv set up (`.venv/`) with `requirements.txt` installed.
- For the qwen arm: Ollama running locally (`brew services start ollama`) with
  `qwen2.5:7b` pulled.
- For the gemini arm: `GEMINI_API_KEY` set in a local `.env` file at the repo root
  (never committed — see `.gitignore`). Get a free key at
  https://aistudio.google.com/apikey. Skip this arm with `--skip-llm` if no key is
  available.
- For the dictabertX arm: a trained checkpoint downloaded from
  `notebooks/train_dictabert.ipynb` (see below) — optional, the arm is skipped without one.

Run the full eval pipeline: `pipeline/run_pipeline.sh $ARGUMENTS`

This validates the given (or default) train/dev CSVs against `pipeline/validate_data.py`'s
checks (required columns, non-empty gold, at least 2 candidates per row, gold present in
its own candidate list, no acronym-type leakage between train and dev), then runs every
arm (random/most_frequent/most_mined/oracle baselines, untrained DictaBERT, and — if
`--checkpoint <path>` is passed — the fine-tuned DictaBERTX) against the dev file via
`pipeline/run_all.py`, printing one combined markdown table and saving it to
`results/all_arms_summary.md` (or wherever `--out` points).

Training itself (the fine-tuned DictaBERTX checkpoint) is NOT run here — it needs a Colab
GPU. See `notebooks/train_dictabert.ipynb`. Download the resulting `.pt` and pass its path
with `--checkpoint` to include that arm in the table.

Pass `--skip-llm` to skip the qwen/gemini arms (they call a local model / hosted API and
are slower); useful for a quick sanity check after swapping in a new dataset.

After running, if the validator reports any issues, stop and report them plainly rather
than proceeding to the eval step — a bad split (leakage, missing columns, empty golds)
should be fixed before spending time on evaluation, not evaluated anyway.

## After a successful run

Update `results/all_arms_summary.md` (the pipeline writes this automatically) AND
`README.md`'s results tables and prose by hand — the README's numbers and discussion do
not update themselves, so if any arm's result changed from what's currently written
there, edit it to match before considering the run "done."
