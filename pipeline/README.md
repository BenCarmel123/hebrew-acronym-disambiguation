# Validation and evaluation entry points

Use the [root README](../README.md#install-and-check) for safe fixture and environment
checks. The research entry points below require a separately authorized task.
**`run_pipeline.sh` evaluates test automatically when its file exists; `--skip-llm`
does not disable test.** Neither it nor `run_all.py` is a structural smoke check.

| Module | Behavior |
|---|---|
| `check_environment.py` | Offline imports, invented pair checks and one cached encoder forward pass on CPU; no research inputs. |
| `validate_data.py` | Check required columns, nonempty gold, candidate count/coverage and split overlaps. These checks reflect existing data assumptions, not a scientific decision. |
| `run_all.py` | Evaluate the supplied items with reference baselines, encoder similarity, optional trained cross-encoder and optional Qwen/Gemini arms. |
| `run_pipeline.sh` | Validate train/dev and existing test inputs, run dev evaluation, then repeat on test if present. Does not train. |

The model evaluators retain distinct selection, scoring and skip rules. The combined
runner assembles their reported metrics; it does not make them a single evaluator.

## Research invocation reference

Run from the repository root with the environment described in README. Inspect Python
arguments with `.venv/bin/python -m pipeline.run_all --help` or
`.venv/bin/python -m pipeline.validate_data --help` before an authorized execution.
The shell script has no help-only mode; inspect its source instead.

| Argument | Existing shell default or meaning |
|---|---|
| `--train`, `--dev`, `--test` | `data/splits/train_items.csv`, `dev_items.csv`, `test_items.csv`. Missing test file skips test; present test is evaluated. |
| `--candidates` | `data/mined/candidate_table.csv`. |
| `--checkpoint` | Existing trained checkpoint; without it, cross-encoder evaluation is omitted. See [checkpoint notes](../weights/README.md). |
| `--skip-llm` | Skip Qwen/Gemini only; baselines, DictaBERT and existing test still run. |
| `--out` | `results/all_arms_summary.md`; test output uses an added `_test` suffix. Existing output can be overwritten. |

The Qwen arm requires a running Ollama service and its configured model. The Gemini
arm uses `GEMINI_API_KEY` from the environment/local ignored `.env`. Service setup or
API credentials are not needed for safe checks. The encoder path can download model
weights unless its runtime is prepared for offline use; it is distinct from the offline
smoke entry point. Training requires the notebook's explicit `train` mode and local inputs.

Before an authorized research run, choose explicit inputs and an output destination
that preserves [historical results](../results/all_arms_summary.md). Future reported
results must be associated with their actual data/configuration; do not overwrite the
historical table or manually copy inconsistent scores into the root README.
