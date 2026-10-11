# Hebrew Acronym Disambiguation

A Tel Aviv University NLP course project comparing Hebrew acronym expansion by
language models and an encoder. The study contrasts free generation with selection
from a supplied candidate inventory; it does not isolate the causal effect of model size.

Read the [paper source](paper/main.tex) and the
[experimental study notebook](notebooks/experimental_study.ipynb). The notebook
contains saved tables, figures and interpretation for the 381-item test cohort.
Its outputs can be read without running any code. The
[export index](results/submission-20261010/README.md) identifies reusable results;
the [paper guide](paper/README.md) explains PDF building.

## Install and check

From the repository root, create an isolated Python environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e . jupyterlab ipykernel
.venv/bin/python -m pip check
.venv/bin/python -m jupyter lab notebooks/experimental_study.ipynb
```

Select this environment's kernel. If necessary, register it with
`.venv/bin/python -m ipykernel install --user --name hebrew-acronym-analysis`.
Dependencies are pinned in `requirements.txt` through `pyproject.toml`; the complete
transitive environment is not locked. Use the installed package without import-path
modifications.

Default analysis reads bundled evidence and checks its identities. It requires no
model weights, credentials, training or API calls. For a fresh, guarded execution
from a commit archive, follow
[reproduce the saved analysis](docs/reproducible_evaluation.md#reproduce-the-saved-test-analysis).
That guide also gives offline checks and explains the separate collection workflow.

## Repository map

| Location | Contents |
|---|---|
| [notebooks/experimental_study.ipynb](notebooks/experimental_study.ipynb) | Main methods and saved-results analysis. |
| [src/hebrew_acronyms/](src/hebrew_acronyms/) and [tests/](tests/) | Data preparation, models, scoring, collection and engineering checks. |
| [data/](data/README.md) | Frozen exports, candidate inventory, review evidence and encoder inputs. |
| [saved-results/](saved-results/README.md) | Identified response journals, annotation provenance and source signatures. |
| [results/submission-20261010/](results/submission-20261010/README.md) | Tables derived from the saved evidence. |
| [paper/](paper/README.md) | LaTeX manuscript, bibliography and figures. |

The optional [collection notebook](notebooks/run_test_eval_colab.ipynb) and
[training example](notebooks/train_dictabert.ipynb) perform separate operations;
they are not needed for saved analysis. New collection needs explicit inputs,
provider access and budget settings. Checkpoint retrieval and the limits of
historical training evidence are documented in [checkpoints](docs/checkpoints.md).

The [dataset card](data/mined/DATASET_CARD.md) distinguishes observed, transformed
and authored text and explains incomplete annotation and reconstruction evidence.
Human review is partial and diagnostically selected. AI assistance contributed code,
checks, draft prose and the explicitly identified data work; it does not establish
independent human annotation. Development observations remain separate from test results.
