# `pipeline/` — validate a dataset, run every eval arm, get one results table

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
- **dictabertX arm** (fine-tuned cross-encoder): needs a trained checkpoint. Training
  itself does NOT run here — it needs a Colab GPU. Run
  `notebooks/train_dictabert.ipynb`, download the resulting `.pt`, and pass its path
  with `--checkpoint`. Without `--checkpoint`, this arm is skipped (shown as `—` in the
  table) and every other arm still runs.

## Usage

```bash
pipeline/run_pipeline.sh
pipeline/run_pipeline.sh --train data/splits/train_items.csv --dev data/splits/dev_items.csv
pipeline/run_pipeline.sh --checkpoint checkpoints/dictabert-crossenc-<ts>.pt
pipeline/run_pipeline.sh --skip-llm   # fast: skips qwen/gemini (local model + API calls)
```

Or from Claude Code: `/pipeline` (see `.claude/commands/pipeline.md`).

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

## After a successful run

The script writes `results/all_arms_summary.md` automatically. It does **not** update
the project `README.md` — its results tables and surrounding discussion are written by
hand and need to be edited to match whenever a number changes, so treat updating both
`results/all_arms_summary.md` and `README.md` as part of finishing a run, not optional
cleanup.
