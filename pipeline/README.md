# `pipeline/` — validate a dataset, run every eval arm, get one results table

**Historical research entry point; not authorized for S3a.** Use the safe checks and
local training appendix in [the root README](../README.md). The shell pipeline
automatically evaluates test when its file exists; `--skip-llm` does not disable it.
Descriptions below are reference material, not instructions to run during organization.

Lets anyone swap in a new/improved `train_items.csv` + `dev_items.csv` and re-run the
whole comparison without touching model code — the dataset and the evaluation code are
decoupled on purpose, so dataset work and model/eval work can happen in parallel.

## Prerequisites

- Python venv set up (`.venv/`) with `requirements.txt` installed.
- **qwen arm**: Ollama running locally (`brew services start ollama`) with `qwen2.5:7b`
  pulled (`ollama pull qwen2.5:7b`).
- **gemini arm**: `GEMINI_API_KEY` set in a local `.env` file at the repo root — never
  committed, see `.gitignore`. Get a free key at https://aistudio.google.com/apikey.
  Without it, run with `--skip-llm` or expect that arm to fail.
- **dictabertX arm** (fine-tuned cross-encoder): needs a trained checkpoint. Training itself does not run here. The local
  [training appendix](../notebooks/train_dictabert.ipynb) defaults to inference-only smoke;
  its explicit training mode is reserved for later authorized use. Pass an existing
  checkpoint path with `--checkpoint` when evaluation is authorized. Without `--checkpoint`, this arm is skipped (shown as `—` in the
  table) and every other arm still runs.

## Usage

```bash
pipeline/run_pipeline.sh
pipeline/run_pipeline.sh --train data/splits/train_items.csv --dev data/splits/dev_items.csv
pipeline/run_pipeline.sh --checkpoint weights/dictabert-crossenc-<ts>.pt
pipeline/run_pipeline.sh --skip-llm   # fast: skips qwen/gemini (local model + API calls)
```

The earlier `/pipeline` tool-specific command referred to a file absent from this
repository; use the documented entry points when separately authorized.

## What it does

1. **`validate_data.py`** checks the given train/dev pair: required columns present,
   every row has a non-empty `gold_expansion`, at least 2 candidates per row (a
   single-candidate row has no real choice to make), `gold_expansion` actually appears
   in its own `candidates` list, and — importantly — **no acronym type appears in both
   train and dev**. That last one matters most: dev is supposed to measure
   generalization to acronyms never seen in training (see `data/splits/README.md`); if
   the validator finds overlap, fix the split before trusting any number it produces.
2. **`run_all.py`** runs every arm against `dev` and prints/saves one combined table:
   random / most-frequent / most-mined / oracle baselines, untrained DictaBERT,
   fine-tuned DictaBERTX (if `--checkpoint` given), and the qwen/gemini LLM arms in both
   generate and select mode (unless `--skip-llm`).

3. If the test CSV exists, the shell pipeline validates it and repeats evaluation,
   writing a separate test summary. This happens even with `--skip-llm`.

## After a successful authorized run

The script writes `results/all_arms_summary.md` automatically. It does **not** update
the project `README.md` — its results tables and surrounding discussion are written by
hand and need to be edited to match whenever a number changes, so treat updating both
`results/all_arms_summary.md` and `README.md` as part of finishing a run, not optional
cleanup.
