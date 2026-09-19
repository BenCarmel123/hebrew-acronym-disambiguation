# Data locations and processing layers

These research inputs and historical exports remain unchanged since baseline
`8ca117d50c3f01d4473b944c99611c7191af05b4`. Structural organization and handoff
checks do not rebuild the data or remeasure the recorded research results.
The [dataset card](mined/DATASET_CARD.md) is the source for field definitions,
construction history and limitations; the next scientific protocol remains open.

| Processing layer | Location | What is retained |
|---|---|---|
| External source and saved source export | Wikipedia/Wiktionary APIs; Knesset corpus; ignored `data/raw/` when locally downloaded | API responses were not systematically retained according to the mining documentation. Do not assume an original snapshot exists for every row. |
| Candidate inventory and mining exports | [mined/](mined/): `candidate_table.csv`, `acronym_items.csv`, source subdirectories, `merged_counts.csv` | Candidate hypotheses, mined contexts and mechanically assigned labels; not a uniformly reviewed dataset. |
| Review records and applied review | `mined/duplicate_review.csv`, `mined/dev_review.csv`, `mined/knesset/knesset_reviewed.csv` | Existing decisions and review annotations as documented. Review status must be traced to records, not inferred from a filename. |
| Split inputs and aggregate exports | [splits/](splits/README.md): `train_items.csv`, `dev_items.csv`, `test_items.csv`, `all_items.csv`, `by_category/` | Historical input roles. The aggregate/category exports are separate artifacts; they are not guaranteed to equal the current split union. |

This is a location map, not a claim that every item followed a single linear pipeline.
Some authored/deglossed additions entered split files directly. The card records those
stages; existing build scripts do not by themselves establish full reconstruction of
every later addition.

Keep these independent questions separate:

- **Where was it processed?** Source export, mining table, review record or split input.
- **Where did the text come from, and how was it changed?** Source and construction
  method are separate from label status; `manual` includes disclosed AI-authored text.
- **What supports the label?** Mechanically derived, reviewed according to existing
  records, or unknown. `gold_expansion`, `verified` and a `reviewed` filename alone
  establish neither correctness nor independent agreement.
- **What role does the file play?** A historical train/dev/test input is not automatically
  approved for the next experiment.

[Engineering fixtures](../tests/fixtures/) are invented and kept outside this directory.
Notebook smoke reads those fixtures only and writes no weights; tiny checkpoint tests
use temporary directories. [results/](../results/) contains historical evaluation
outputs, while `weights/` contains ignored checkpoints when available. A result must be
associated with its actual run and inputs before being cited; organization is not rerunning.
