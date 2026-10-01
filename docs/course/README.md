# Course sources

These files are reference material, not approval of the current experimental protocol.
[Mor's feedback](mor_feedback.md) is preserved unchanged and asks for a precise research
question, related literature and motivation relative to prompting large language models.

The old source is [ShakedSchnarch/nlp-hw-team](https://github.com/ShakedSchnarch/nlp-hw-team),
local commit `3bae9130a6a7086e3861235afc5360e4d08bc7ea` (includes unpublished history).
The new repository baseline is `eb2e7785dab42dc8ae3ca07372d032adafee9dba`.

| Document | Source and status | SHA-256 |
|---|---|---|
| [Submitted proposal](209533108_209233857_315110841_NLP_Project_Proposal.pdf) | Confirmed by Shaked on 1 October 2026; byte-identical to his newly saved Downloads copy and the unchanged old `project/proposal/` file with this name. | `1e335e6baddcb699aecadcc290801785a2e62bea010e82c4a5ecb876f7e72c56` |
| [Earlier draft](hebrew_acronym_disambiguation_proposal.pdf) | Present at new baseline; identical to old `project/proposal/hebrew_acronym_disambiguation_proposal.pdf`. | `27c92cb4ff6378418c6d0b162008453cd3885c1a893a1c42e9e5c44606da0b5f` |
| [Guidelines](NLP_course_2025b___project_guidelines.pdf) | Present at new baseline; identical to old `project/guidelines/NLP_course_2025b___project_guidelines.pdf`. | `cc67f99fe1efb1a373d564a2332a64509772f3b166afc551769bd3a5676aaaa4` |

Shaked identified the newly saved Downloads PDF as the submitted proposal on
1 October 2026. Its SHA-256 and size (171,050 bytes) match the submitted file above.
This resolves which proposal to use; it does not establish lecturer approval of
later protocol changes. Both PDFs describe the same generation-versus-selection
question. The earlier draft differs in literature and scope wording and is retained
as history, not a competing specification. Neither PDF has been edited.

## Submitted scope and subsequent direction

The submitted proposal makes the benchmark and controlled generation/selection
comparison the core contribution, with DictaBERT as a small candidate ranker.
It targets 30–50 types, 1,000–2,000 train/development examples and 300–500 held-out
examples, initially from Wikipedia. It proposes document-disjoint splits, an
optional unseen-type stress test, frequency/classical baselines, multiple seeds
and prompt templates. Its fallback permits scope reduction when mining is noisy
and an evaluation-focused contribution when fine-tuning gives no useful signal.

The later approved direction uses the same LLM in four generation/selection ×
sentence/no-sentence conditions, a small encoder on task-unseen types, separate
natural/AI reporting, and approximately 100–120 natural evaluation occurrences.
These are later project decisions, not wording from the submitted proposal or
evidence of lecturer approval. Detailed sampling, systems, scoring and run settings
remain open in the [central plan](../../../PROJECT_PLAN.md).

The old repository's `project/benchmark/hebrew-acronym-benchmark-plan.md` describes
a separate 600-type / 3,000-item HeAcro concept, explicitly marked unratified.
It is not the submitted course scope or an active requirement. Older literature
claims and methodological recommendations in that concept are not adopted here.
The submitted PDF also contains an incorrect DictaBERT citation; the corrected
attribution is already used in the [working manuscript](../../paper/manuscript.md)
and [bibliography](../../paper/references.bib). Preserve the submitted artifact.
