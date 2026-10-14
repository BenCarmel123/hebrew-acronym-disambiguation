# Hebrew Acronym Disambiguation

This Tel Aviv University NLP course project studies how models resolve Hebrew
acronyms in context. Its evaluation pipeline compares free generation and selection
from a supplied candidate inventory on the same occurrences, using prompted language
models and a task-trained Hebrew encoder. Ben Carmel and Shaked Schnarch are the
authors.

Start with the [paper source and PDF build guide](paper/README.md) and the
[main experimental notebook](notebooks/experimental_study.ipynb). **The notebook's
saved tables, figures and explanations can be read without installing or running
anything.** It presents the retained 381-item test cohort and links to the underlying
evidence. The final paper (14 October 2026) is [paper/main.tex](paper/main.tex) with its
[sections](paper/sections/); the [build guide](paper/README.md) produces the PDF.

For a fresh execution, follow the single
[saved-analysis installation and reproduction procedure](docs/reproducible_evaluation.md#reproduce-the-saved-test-analysis).
It creates an isolated environment from a committed source archive, checks the
installed package location and executes the notebook. Execution needs no model
weights, credentials, training or API calls. Dependencies are pinned directly;
the complete transitive environment is not locked.

The [data guide](data/README.md) explains corpus construction and known reference
errors; the [saved-evidence guide](saved-results/README.md) records response and
review provenance. The [export index](results/submission-20261010/README.md) describes
reusable tables. Selection exceeds generation under the task-specific automatic
metrics; selective, reference-relative human review does not establish a common
semantic ranking. The encoder is competitive with the tested local Qwen
configurations, subject to the documented comparison limits.

Optional collection and training notebooks are separate from this reading and
reproduction path; their requirements and historical limitations are linked from
the reproduction guide.
