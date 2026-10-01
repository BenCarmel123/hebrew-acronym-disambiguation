# Local checks and component entry points

There is no combined experiment runner. Import model/data functions explicitly, as
shown in the [training appendix](../notebooks/train_dictabert.ipynb), so inputs and
operations remain visible. No entry point selects test because a file exists.

| Module | Purpose |
|---|---|
| [check_environment.py](../src/hebrew_acronyms/pipelines/check_environment.py) | Installed imports, exact invented target pairs and an optional cached encoder forward pass on CPU. |
| [validate_data.py](../src/hebrew_acronyms/pipelines/validate_data.py) | Required fields, nonempty gold, candidate coverage and split overlaps for explicitly selected inputs. These historical assumptions do not qualify research data. |

Use the environment in the [README](../README.md#install-and-check). An optional
cached-model smoke check is:

```bash
.venv/bin/python -B -m hebrew_acronyms.pipelines.check_environment --snapshot /path/to/local/snapshot
```

Without `--snapshot`, the command checks the local DictaBERT cache. It blocks socket
access in its isolated process and uses invented inputs only. Missing cache returns
exit code 2 (`NOT RUN`); other failures are nonzero. It never downloads weights.
The cache-free notebook check is `.venv/bin/python -B -m tests.run_notebook`.

The [item predictor](../src/hebrew_acronyms/models/dictabert_cross_encoder/eval.py)
returns IDs, candidate choices, raw scores and failure statuses. It does not compute
benchmark aggregates. Other existing evaluators have different scoring and skip
rules; they cannot yet supply a consistent comparison without a shared protocol.

Research data preparation, training, evaluation and service calls require explicit
inputs and an appropriate execution decision. Inspect data-validation arguments with
`.venv/bin/python -m hebrew_acronyms.pipelines.validate_data --help`. Source preparation
commands are described [separately](data_processing.md). Preserve historical outputs
and associate new results with their actual data, configuration and model revisions.
Qwen uses Ollama; Gemini uses a local `GEMINI_API_KEY`. Neither service is needed for
the fixture suite. Standalone historical model CLIs are research tools, not smoke checks.
