# Paper source, evidence and build

`main.tex` and `sections/*.tex` are the sole maintained manuscript source.
Ben's original manuscript is retained in Git at
`4d2dc738914fa67970f18ee897b3deb7ff19f4f2:paper/overleaf_main.tex`.
The duplicate source has been removed from the active tree. For Overleaf, import the multi-file project, select `main.tex` and XeLaTeX.
The old `manuscript.md` is a pointer, not another manuscript.

The 10 October draft uses the saved **381-item test results**, retains two qualified
conclusions, and reports DictaLM separately with four measurements. It is a review
draft. Author contributions and personal AI reflections still require author
content or confirmation; independent scientific review has not been performed
on this revision. The confirmed submission date is 14 October 2026.

## Evidence and reproduction

Open [the main analysis notebook](../notebooks/experimental_study.ipynb).
Its saved-result path reads [hash-identified evidence](../saved-results/README.md)
without model requests, credentials or weights. The complete output map is in
[the test export guide](../results/submission-20261010/README.md).

| Manuscript content | Saved source under `results/submission-20261010/` |
|---|---|
| Main comparison | `automatic_metrics.csv`, `baselines.csv`, `similarity_metric.csv` |
| Similarity provenance limits | `similarity_source.json` |
| Four DictaLM measurements | `dictalm_reported_metrics.csv` |
| Ben's annotation and agreement counts | `dictalm_ben_judgments.csv`, `dictalm_ben_cross_tab.csv` |
| Failures and format validity | `metric_coverage.csv`, `technical_failures.csv` |
| Existing examples `kn-0002` and `kn4-0048` | `dictalm_responses_381.csv`, `response_review_bindings.csv` |
| Model identities and settings | `models_and_settings.csv` and the bound run manifests |

The source breakdown table retains Knesset/Wikipedia/authored groups and the same
automatic rules. No labels, denominators or results change during manuscript editing.
The main table retains the numerical results. `figures/generate_vs_select.pdf`
shows candidate selection (two encoder variants and six LLMs) alongside generation
and selection for the same six LLMs. The panels preserve task-specific scoring;
encoders rank candidates without the strict LLM text-label decoder. The most-mined
line uses the saved 0.4803 value, displayed as 48.0%. All three bar columns span
0--100%. A shared model-label column aligns selection across all systems with
separate generation and selection columns for the six LLMs. Numbers have fixed
positions outside the bars; generation uses open hatched bars and selection uses
solid bars. Encoder rows in the LLM columns are not applicable, not zero scores.
Full metric definitions appear in the adjacent caption.

Regenerate this vector PDF from the repository root with the installed project
environment, then build the paper:

```bash
python paper/make_result_figures.py
make -C paper check
```

The generator reads only the three saved CSV exports listed above. It checks the
381-item denominators, metric identities and count/accuracy agreement; it neither
rescores responses nor regenerates exports. Timestamp metadata is omitted for
repeatable PDF bytes in the same environment (Matplotlib 3.10.8; DejaVu Sans).
The historical selection-only figure/helper remains available but is not rebuilt
by the default command and is not included in the manuscript.

The constructed illustration remains a table: a trial two-path method diagram
made scoring harder to read in one column and increased the body beyond eight
pages. The results figure takes priority, with space still needed for the authors'
contributions and personal reflections. The notebook's saved figures remain
unchanged: they report the same values and measurement layers.

The paper's external links pin the remotely verified Ben revision `4d2dc73`.
The local review snapshot includes later DictaLM annotation supplements and
presentation corrections. Those supplements and the current paper commits are
not represented as published. Use the accompanying checkout for the exact current
analysis and record its revision when distributing it.

### Data and checkpoint evidence boundaries

Counts and exact-string separation refer to these current files:

- `data/study_v1/encoder_inputs/train.csv`: 2,829 items.
- `data/study_v1/encoder_inputs/dev.csv`: 62 items, distinct from the collection pilot.
- `data/splits/test_items.csv`: 381 items.

Pairwise exact acronym-string and sentence-string intersections are empty.
Recorded nonempty document keys `(source, page_title)` overlap for three train/dev
documents and none between test and train/dev. These are file checks, not proof
of the historical training input, family-level separation or pretraining absence.
The stored inventory includes gematria and other conventional interpretations.

The 253 Knesset test labels have recorded human-review evidence (221 confirmed,
32 corrected); 87 authored items carry a human-review marker. The 41 Wikipedia
items have `label_status=verified` but no review verdict. Ben's construction commit
`afbbe016bc0d31ac1817913e2c67cc670c6618ef` documents review for inclusion and removal
of rows without labels; it does not establish contextual human review of all
Wikipedia labels. No comprehensive test-gold review is inferred from that status.
Gold provenance is distinct from human review of model responses.

[Checkpoint documentation](../docs/checkpoints.md) retains historical training
unknowns, including the seed discrepancy and missing input snapshot. The saved
`checkpoint-reconstruction.json` reports 184,347,649 loaded encoder parameters;
DictaLM's saved full-test manifest reports 7,251,169,280 parameters. These support
the rounded model sizes in the paper. They do not reconstruct training.

## Literature verification

Retained claims were checked against primary sources on 10 October 2026:

| Source | Evidence used and scope |
|---|---|
| [Jacobs et al.](https://link.springer.com/article/10.1007/s10472-018-9608-8) | Publisher abstract: Modern Hebrew dictionary construction and contextual disambiguation. |
| [HAADS](https://onlinelibrary.wiley.com/doi/10.1002/asi.21367) | Publisher abstract: Hebrew/Aramaic Jewish Law domain; no unsupported dictionary-design detail. |
| [GLADIS](https://aclanthology.org/2023.eacl-main.152.pdf) | Sections 3.2 and 4: acronym-disjoint benchmark splits, substitution, joint pair representation and triplet ranking. |
| [Agrawal et al.](https://aclanthology.org/2022.emnlp-main.130.pdf) | Section 4: generated clinical expansions resolved through contiguous character overlap with candidates. |
| [Meconi et al.](https://aclanthology.org/2025.emnlp-main.1720.pdf) | Sections 3–4.1: selection, candidate order, and a sampled generation evaluation with expert judgments. |
| [DictaBERT](https://arxiv.org/html/2308.16687v2) | Hebrew pretrained encoder and architecture; local reconstruction supplies parameter count. |
| [Qwen2.5](https://arxiv.org/html/2412.15115v2) | Model family; saved manifests supply the actual run configuration. |
| [DictaLM](https://arxiv.org/abs/2407.07080) | Hebrew adaptation and instruction tuning; author corrected to Amir DN Cohen. |
| [DictaLM publisher format](https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF#instruction-format) | Required newline after `[/INST]`; no causal inference about its omission. |

The cognates paragraph was removed because it did not directly develop either
conclusion. No cross-study score ranking is made. The literature checks are
source verification, not an independent scientific referee report.

## Build and inspect

Use XeLaTeX, BibTeX, latexmk, Polyglossia and Poppler (`pdfinfo`, `pdftoppm`).
The fonts are Times New Roman (including Hebrew), Arial and Menlo. The unmodified
course ACL style and checksums are in [acl_template/PROVENANCE.md](acl_template/PROVENANCE.md).

From the repository root:

```bash
make -C paper check
pdftoppm -r 130 -png paper/build/main.pdf paper/build/review-page
```

`make check` builds `paper/build/main.pdf` and verifies ACL hashes, references,
glyphs, overfull boxes and the eight-page body limit. Inspect every rendered page,
including Hebrew, figures, tables and bibliography. Build outputs are ignored.
Building does not load models, call services, alter research data or regenerate
results. If an optional historical generated bundle exists, supply the Python
interpreter from the project environment with `PYTHON=/path/to/venv/bin/python`.

## Historical development route

[generated/README.md](generated/README.md) describes the earlier `full_dev` exporter.
It is inactive for the current test paper: no `generated/current.tex` is present,
and the test tables are maintained in the sections against the identified exports.
Do not use dev exports as test measurements. Historical development provenance is
retained in [results/dev-2026-10-03](../results/dev-2026-10-03/README.md) and the
checkpoint documentation. The historical development cells remain in Git at `5c6980e` and do not
contribute test observations.

## Reviewing the manuscript changes

```bash
git diff 4d2dc738914fa67970f18ee897b3deb7ff19f4f2 -- paper/
git diff b9cf541acabd402344b07ef38e31da254b51477f -- paper/ README.md
```

Relative to Ben, the main changes qualify the two conclusions, separate DictaLM's
metrics, align the task definition with the inventory, bound gold-review and
training claims to evidence, and connect the discussion to verified literature.
The examples and numerical results are preserved. Notation tables and a redundant
figure were removed from the PDF to keep the argument readable. The existing
accurate motivation, architecture description and scientific wording are retained
where no substantive correction was needed.
