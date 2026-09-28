# Hebrew Acronym Disambiguation: Candidate Access and Sentence Context

**Working manuscript — protocol pending (P1, 28 September 2026).** This draft
develops the approved research direction. The methods below describe proposals,
not a frozen benchmark or completed experiments. P1 authorizes structural data
review and writing only. Scientific decisions remain with Shaked under the
[central project plan](../../PROJECT_PLAN.md); this manuscript is not an
alternative execution plan. No abstract, results, or empirical conclusions are
asserted at this stage. Citation keys resolve in [references.bib](references.bib).

## 1 Introduction

An acronym can abbreviate several expressions, leaving a reader to infer the
intended expansion from its context. This problem has been studied in Modern
Hebrew: Jacobs, Itai, and Wintner describe dictionary construction and contextual
disambiguation in a language whose morphology and orthography complicate acronym
processing [@jacobs2020acronyms]. Thus, Hebrew acronym disambiguation has an
established research history. The question here concerns how to evaluate different
ways of presenting this task to contemporary language models.

A large language model (LLM) can be asked to generate an expansion or to select
one from an inventory. These formulations provide different information and impose
different output constraints. An inventory may help retrieve an unfamiliar
expansion, but it also limits what can be selected and may contain cues that make
the sentence unnecessary. Conversely, a generated expansion may express the
intended meaning using a spelling or name variant absent from the inventory.
Reliable comparison therefore requires both independent inventory validation and
an explicit rule for interpreting generated answers.

We ask: **How does access to a fixed expansion inventory change the same LLM's
accuracy on Hebrew acronym occurrences, and how does removing the sentence affect
each response formulation?** The approved direction pairs generation and selection,
each with and without sentence context. A Hebrew encoder trained on other acronym
types provides a complementary comparison with a smaller task-trained system.
Here, an unseen type means absent from task-specific training; it does not imply
absence from either model's pretraining.

The intended contribution is an interpretable, paired evaluation on documented
Hebrew material. The study distinguishes inventory size from the diversity of
meanings actually observed in the sample, and naturally occurring text from
substituted, edited, and authored examples. Its value does not depend on selection
outperforming generation or on the encoder outperforming the LLM. A null
difference, successful context-free answering, or an advantage for generation
would each constrain the interpretation of this task. No claim of priority or
representativeness of Hebrew as a whole is made.

## 2 Related Work

**Hebrew abbreviation processing.** Jacobs et al. study acronym identification,
expansion, and disambiguation in Modern Hebrew [@jacobs2020acronyms]. Their
publisher abstract describes building a dictionary from unannotated text and
adding contextual information, including cases where an expansion is absent from
the acronym's document. The article appeared online in 2018 and in volume 88 in
2020. Only its abstract and bibliographic record were inspected for this draft;
its experimental splits and detailed evaluation are not characterized here.
HAADS addresses Hebrew and Aramaic abbreviations in Jewish Law documents using
contextual and statistical features with machine learning
[@hacohenkerner2010haads]. This is relevant linguistic precedent, although its
domain differs from the modern sources in this project. Its publisher abstract
and metadata, rather than full experimental text, support this description.

**Candidate-based acronym disambiguation.** GLADIS constructs an acronym
dictionary and datasets covering general, scientific, and biomedical domains.
Its task splits separate acronym types, and AcroBERT scores context–expansion
pairs [@chen2023gladis]. This provides precedent for evaluating generalization to
types absent from task training. It does not make evaluation on known types
invalid: that design answers a different question. GLADIS also uses expansion
replacement to construct examples, reinforcing the need to distinguish text
construction from naturally observed acronym use.

**Generation and answer interpretation.** Agrawal et al. prompt an LLM to expand
clinical abbreviations without supplying answer choices, then resolve its output
to a candidate using contiguous character overlap [@agrawal2022large]. This
separates the model's input from the inventory used by the evaluator. It is a
precedent for mapping generated answers, not evidence that any particular
substring rule is valid for Hebrew. Meconi et al. compare dictionary-definition
selection with generative tasks for English word senses, including human
assessment of generated definitions and explanations [@meconi2025large]. Their
context and candidate-order analyses motivate documenting both prompt content and
answer processing. Word-sense definitions and acronym expansions are different
targets, so their task-specific rubric cannot be transferred unchanged.

**Partial-input controls.** Balepur et al. evaluate multiple-choice questions
using the choices alone [@balepur2024artifacts]. Their analyses do not support
memorization as a sufficient explanation of this behavior and examine relations
among choices and inferred questions. Our no-sentence condition retains the
acronym itself, making it a related partial-input control rather than the same
experiment. Its success could reflect knowledge of acronym–expansion associations,
inventory cues, or other regularities; it would not isolate memorization.

**Hebrew representations.** DictaBERT supplies a pretrained Hebrew encoder
[@shmidman2023dictabert]. Its tokenizer explicitly handles quotation marks used
in abbreviations. The correct authors are **Shaltiel Shmidman, Avi Shmidman, and
Moshe Koppel**, and the paper is **arXiv:2308.16687**. This corrects the bibliographic
details in the preserved project proposal. The model paper does not establish
acronym-disambiguation performance or absence of exposure to our sources.

The comparison below records only details supported by the inspected material.
“Not verified” describes a reading limit, not a claim that a paper omitted a
procedure. No cross-paper score comparison is intended.

| Work | Language/domain | Candidate access | Context | Training/evaluation distinction | Scoring |
|---|---|---|---|---|---|
| Jacobs et al. (2020) | Modern Hebrew | Automatically built dictionary | Contextual information | Split details not verified | Detailed metric not verified |
| HAADS (2010) | Hebrew/Aramaic, Jewish Law | Abbreviation disambiguation; inventory construction not verified | Contextual/statistical features | Split details not verified | Accuracy reported in abstract; procedure not verified |
| GLADIS (2023) | English, multiple domains | Dictionary candidates for ranker | Sentence–expansion pairs | Acronym types separated in task splits | Accuracy and macro F1 |
| Agrawal et al. (2022), clinical sense task | English clinical notes | No choices in generation prompt; inventory in resolver | Clinical snippet | CASI evaluation and transfer of distilled model to substituted MIMIC examples; no type-held-out claim here | Per-acronym accuracy and F1, averaged across acronyms |
| Balepur et al. (2024) | English multiple-choice benchmarks | Answer choices supplied | Full versus choices-only prompts | Prompted evaluation; not an unseen-acronym design | Choice accuracy |
| Meconi et al. (2025) | English word senses | Definitions supplied in selection, absent in generation | Target in context; additional-context and ordering checks | Multiple WSD benchmarks; not a type-held-out acronym design | Selection F1; separate human generation rubric |
| Present proposal | Hebrew, natural text as proposed main evaluation | Same LLM with/without type inventory | Sentence present/absent in both formulations | Encoder trained on other acronym types; extension conditional | Proposed macro/micro Top-1 on a common occurrence set |

## 3 Proposed Methods

### 3.1 Task and paired comparisons

Let an item be an acronym type \(a_i\), a sentence \(x_i\), an explicitly identified
target span \(s_i\), and a reviewed expansion \(y_i\). A type inventory
\(C(a_i)\) contains possible expansions with documented aliases. It is shared
across that type's occurrences, rather than constructed around each sentence's
answer. A system produces one expansion or one selected candidate for the
identified occurrence.

| Condition | Input | Requested output |
|---|---|---|
| Generation with context | Acronym and sentence with target marked | One expansion |
| Selection with context | Same acronym and marked sentence, plus fixed inventory | One candidate identifier |
| Generation without context | Acronym | One expansion |
| Selection without context | Acronym and same inventory/order | One candidate identifier |
| Trained Hebrew encoder | Marked sentence paired with each candidate | Highest-scoring candidate |

All four LLM conditions use the same model version. The proposed primary contrast
is selection minus generation accuracy with context. The two within-formulation
context contrasts are secondary; their difference, if reported, is a secondary
interaction analysis. Candidate access also changes answer format, so this design
does not isolate a pure information effect. Likewise, comparison with the encoder
changes both model family and training regime.

Zero-shot prompts, a fixed candidate order per type shared by both selection
conditions, and recorded decoding settings are proposed controls. Exact model,
prompt, ordering, decoding, retry, and sample-size choices remain pending approval.
No-sentence inputs may repeat across occurrences; any reuse of a response must be
recorded so repeated items are not mistaken for independent model calls.

### 3.2 Data provenance and the proposed evaluation population

The retained corpus combines observed Wikipedia acronym usage and Knesset text,
Wikipedia sentences in which an expansion was replaced by its acronym, edited
examples with an adjacent explanation removed, and authored examples. Wiktionary
and Wikipedia records also provide candidate evidence. These roles are distinct:
the origin of a sentence does not establish the correctness of its expansion, and
a candidate's presence in an inventory does not show that it occurs in the sample.
The [data documentation](../data/README.md) describes the retained exports and
construction history; saved exports are not guaranteed to reconstruct every
upstream retrieval.

The proposal keeps the historical allocation of acronym types to task splits,
while requiring document and duplicate checks before training. This differs from
the submitted proposal's document-first primary split and optional type-held-out
stress test: the intended encoder question now concerns unseen task-training
types. Historical files remain reference inputs until human review establishes
eligibility. Natural development candidates are being audited separately from the
largely substituted historical development set. Their scores, if later obtained,
would not be pooled without an explicit scientific decision.

P1 prepares a [source manifest](../data/study_v1/review/source_manifest.json),
[item audit](../data/study_v1/review/item_audit.csv),
[proposed inventory](../data/study_v1/review/inventory_proposed.csv), and
[review queue](../data/study_v1/review/review_queue.csv). These are review artifacts,
not approved model inputs. The audit links stable identifiers to original rows,
retains source values, distinguishes text origin from label origin, and proposes
target spans and risk flags. Historical item identifiers alone are insufficient
for joining exports because collisions exist. A nonempty source/title pair is a
structural document key; incomplete provenance and possible document aliases
still require review.

The proposed main population uses naturally occurring text. Substituted,
deglossed, and authored examples retain separate labels for construction method;
AI-authored examples would form a separately disclosed challenge set. A difference
between these groups would not identify a causal effect of text source, since
their acronym and sense composition also differ. Final counts and an exclusion
flow belong to the approved data release, not to this preliminary manuscript.

### 3.3 Inventory and human data review

The proposed scope is literal expansions. Numerical letter interpretations and
anonymized identities are proposed exclusions requiring recorded reasons. The
review prioritizes conflicting inventories, labels without independent inventory
support, quotation and prefix variants, repeated target occurrences, definitions
in the sentence, missing provenance, and previously uncertain records. Flags
identify cases for inspection; they are not semantic verdicts.

Inventory changes require evidence independent of the evaluated sentence and
apply consistently across a type. Neither a row's reference answer nor the desire
for two alternatives justifies adding a candidate. Aliases must denote the same
expansion under an approved rule; similar names are not automatically equivalent.
Coverage before exclusions must be reported alongside coverage of the final
evaluation population. An unresolved label or missing candidate remains a review
issue rather than being repaired by copying the row's answer into its inventory.

Inventory size \(|C(a)|\) and the number of distinct reviewed expansions observed
for \(a\) measure different things. A valid singleton inventory remains visible
as a proposed evaluation stratum; selecting its sole member is structurally
trivial. Overall reporting and a separate stratum with at least two candidates
are proposed, without requiring two observed meanings per type. Singleton
exclusion from pairwise training, if adopted, is a separate training decision.

Shaked is the available human reviewer. Earlier review records attributed to Ben
are preserved with their recorded uncertainty and notes. P1 prepares a
[16-item development/training pilot](../data/study_v1/review/pilot_items.csv) and
[short rubric](../data/study_v1/review/pilot_rubric.md) with separate checks of
label, target, and inventory. Human decisions and timing remain unfilled until
that review occurs.
The proposed workload estimate uses observed review time and issue types from
that pilot. No human pilot or inter-annotator agreement is claimed in this draft.

### 3.4 Encoder and conditional exposure extension

The proposed encoder ranks context–candidate pairs using a Hebrew pretrained
backbone, with DictaBERT the documented candidate. Task training uses other
acronym types, and competing expansions from the independently reviewed inventory
provide negative pairs. Target-preserving input construction and a fixed training
configuration must be specified and tested before execution. The existing training
implementation is not evidence that its settings have been scientifically
approved; checkpoint selection, seed handling, and budgets remain protocol choices.

An optional extension would replace some substituted training examples from
training types with substituted examples from evaluation types. It would compare
the same encoder architecture on the same eligible evaluation occurrences while
matching source, example count, candidate-pair count, and optimizer-update budget.
Adding examples without replacement would confound exposure with training volume.
Matching item counts alone is insufficient when inventory sizes differ.

P1 assesses only structural feasibility in
[extension_feasibility.csv](../data/study_v1/review/extension_feasibility.csv).
Eligibility must be based on provenance, separation, and matching constraints,
without system scores or choosing examples to match a test answer. Label coverage
is a descriptive diagnostic after structural selection, not an eligibility filter.
Even a feasible replacement can leave unequal sense coverage, input lengths, or
examples per type. Inventories must therefore be reconciled before matching is
accepted. The extension remains conditional, and P1 constructs no active replacement
training set.

### 3.5 Proposed evaluation and uncertainty

Let \(I_a\) be the approved evaluation occurrences of type \(a\), let \(A\) be
their type set, and let \(z_i\) indicate a correct answer. The proposed primary
metric gives each acronym type equal weight:

\[
\mathrm{MacroAcc}=\frac{1}{|A|}\sum_{a\in A}
\frac{1}{|I_a|}\sum_{i\in I_a}z_i.
\]

Micro accuracy averages \(z_i\) over all occurrences and is complementary.
Every condition must use the same item universe. The proposed denominator keeps
missing, invalid, or unmapped answers as unsuccessful alongside separately
reported service-completion and parsing rates. A systemic execution failure must
be resolved rather than interpreted as a model-quality result. Uniform selection
has expected per-item accuracy \(1/|C(a_i)|\) when exactly one inventory member
is correct; its macro aggregation must follow the same type weighting. No
training-derived most-frequent-sense baseline is available for types absent from
task training. Mining or search-hit counts are not established sense frequencies.

Generated outputs would first undergo narrowly specified normalization and
mapping to approved aliases. A generated answer outside the inventory is not
automatically a hallucination. Unresolved outputs require an approved human
rubric, with recorded decisions and a sample check of automatic mappings. Multiple
answers, negated answers, and ambiguous mappings must be handled explicitly.
Data review precedes output judging; output judgments cannot silently change the
benchmark. System and condition labels should be hidden where feasible, although
answer format may reveal the condition. Shaked's judgments cannot support a
claim of agreement between independent human annotators.

Comparisons are paired by occurrence, but observations can share both a type and
a document. Owen and Eckles study reweighting by crossed factors
[@owen2012bootstrapping]. This motivates the central plan's proposed type/document
weighting, rather than an independent-sentence uncertainty model. The project's
macro-weighted statistic and finite-sample behavior still require methodological
review and hand-checkable fixtures before adoption. This citation does not prove
confidence-interval coverage for a small, selected corpus. Training-seed variation
would be reported separately and would not increase the number of independent
test occurrences.

## 4 Limitations

This study uses a curated, non-random collection from limited sources. Review can
improve traceability but cannot make the sample representative of everyday Hebrew.
An inventory may omit legitimate expansions; reporting performance only after
coverage filtering can conceal that limitation. Target ambiguity, inconsistent
spellings, and incomplete document metadata can survive mechanical checks.
Natural and constructed data differ in more than their construction procedure.

Separating task-training types and documents does not establish pretraining
separation, particularly for public text. The LLM and encoder differ in scale,
training data, and adaptation, preventing attribution of their difference solely
to architecture. A single LLM and fixed prompts also limit conclusions about LLMs
in general. Human review is constrained to one currently available reviewer;
neither model-assisted checking nor preserved earlier review establishes
independent double annotation. Historical predictions are documentary material
and are not findings on the proposed evaluation population.

## AI Disclosure and Reflection — current draft

An AI coding assistant drafted this manuscript and bibliography, inspected the
specified primary-source sections, and helped prepare protocol and data-review
materials. Shaked selected the research direction and explicitly authorized the
P1 preparation package. Existing data documentation attributes earlier human
review to Ben and records AI authorship for some examples; these attributions are
preserved, not independently re-established here. The assistant is not an
additional human annotator.

This draft has not yet been represented as read, rewritten, or approved by Shaked.
The final disclosure must reflect the work actually retained and verified, and
the final reflection must be supplied from the student's own experience. A current
limitation of assistance is that structural flags cannot settle semantic labels,
and a plausible citation can overstate what was read. Explicit verification scopes
and unresolved human decisions are retained to make those limits inspectable.

## Reference verification notes — drafting appendix

Primary sources were checked on 28 September 2026. “Full-text sections” means
the specified sections were inspected in the linked PDF; it does **not** mean
every page, appendix, or released implementation was reviewed. These notes can be
condensed when the manuscript is transferred to the required ACL format.

| Reference key | Primary record and inspected scope | Boundary of supported use |
|---|---|---|
| `jacobs2020acronyms` | [Publisher page](https://link.springer.com/article/10.1007/s10472-018-9608-8): abstract and metadata only | Modern Hebrew precedent; dictionary/context description. Full text was not inspected; no detailed split or score claim. |
| `hacohenkerner2010haads` | [Publisher page](https://onlinelibrary.wiley.com/doi/10.1002/asi.21367): abstract and metadata only | Hebrew/Aramaic Jewish Law domain and contextual/statistical ML approach; no detailed split claim. |
| `chen2023gladis` | [ACL record](https://aclanthology.org/2023.eacl-main.152/) and [PDF](https://aclanthology.org/2023.eacl-main.152.pdf): §§3.2, 4, and Table 5 | Type separation, substituted construction, pair scoring, named metrics. Does not validate our data or loss. |
| `agrawal2022large` | [ACL record](https://aclanthology.org/2022.emnlp-main.130/) and [PDF](https://aclanthology.org/2022.emnlp-main.130.pdf): §4 | Clinical task, candidate-free prompt, character-overlap resolver, evaluation aggregation. No claim that its mapping is valid here. |
| `balepur2024artifacts` | [ACL record](https://aclanthology.org/2024.acl-long.555/) and [PDF](https://aclanthology.org/2024.acl-long.555.pdf): §§3–4 and §5 motivation/prompt definitions | Choices-only control and limits of a memorization-only explanation; no claim of eliminating contamination. |
| `meconi2025large` | [ACL record](https://aclanthology.org/2025.emnlp-main.1720/) and [PDF](https://aclanthology.org/2025.emnlp-main.1720.pdf): §3.1, context/order discussion, §4.1, §6 | English WSD versus generation, answer interpretation, human rubric, limitations. No direct transfer of scores or rubric. |
| `shmidman2023dictabert` | [arXiv record](https://arxiv.org/abs/2308.16687) and [v2 PDF](https://arxiv.org/pdf/2308.16687v2): title/authors and §2 | Correct authors/identifier, Hebrew encoder and tokenizer. No acronym-task or non-exposure claim. |
| `owen2012bootstrapping` | [arXiv/journal record](https://arxiv.org/abs/1106.2125) and [PDF](https://arxiv.org/pdf/1106.2125): abstract, introduction and §4 definition | Crossed-factor reweighting as motivation; proofs and our proposed macro estimator have not been fully validated. |
