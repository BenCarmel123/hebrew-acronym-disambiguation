# Hebrew Acronym Disambiguation: Candidate Access and Sentence Context

**Working manuscript — protocol pending (status clarified 1 October 2026).** This draft develops
the approved research direction and task rules. Remaining method settings are
proposals, not a frozen benchmark or completed experiments. P1 was an earlier
structural preparation package; it does not authorize further execution. Scientific
decisions remain with Shaked under the [central project plan](../../PROJECT_PLAN.md).
No abstract, results, or empirical conclusions are asserted at this stage.
Citation keys resolve in [references.bib](references.bib).

## 1 Introduction

An acronym can stand for several expressions. Interpreting a particular occurrence
requires identifying the expansion appropriate to its use in the sentence. Hebrew
acronym processing has an established research history: Jacobs, Itai, and Wintner
study dictionary construction and contextual disambiguation in Modern Hebrew,
where morphology and orthography complicate the task [@jacobs2020acronyms]. Our
question concerns how to present this task to language models and interpret their
answers, without assuming that Hebrew lacks prior resources.

A large language model (LLM) can generate an expansion freely or select from an
inventory: a list of possible expansions for the acronym type. These formulations
change both the available information and the required answer. The list can make
an unfamiliar expansion available, but an incomplete list also restricts the
answers. Free generation allows alternative wording, which must be judged against
a stated rule rather than rejected merely because it is absent from the list.
Candidate validation and answer interpretation are therefore part of the
comparison, not incidental processing steps.

Candidate access and sentence context are separate dimensions. The inventory
states which expansions are possible; the sentence supplies evidence about the
particular occurrence. Comparing generation and selection with the sentence
measures a difference between response formulations. Removing the sentence from
each formulation separately tests how its accuracy changes when that evidence is
unavailable. A strong no-sentence result could reflect familiar acronym–expansion
associations or cues in the inventory; it would not by itself establish memorization
or prove that context is unnecessary for other occurrences.

We ask: **How does the same LLM's accuracy on Hebrew acronym occurrences differ
between free generation and selection from a fixed expansion inventory, and how
does removing the sentence change accuracy within each formulation?** The approved
direction uses these four conditions, with a smaller Hebrew encoder as a
complementary comparison. An encoder maps text to learned representations; here
it would score each sentence–candidate pair. Its evaluation concerns acronym types
absent from task-specific training, not necessarily absent from pretraining.

The intended contribution is a paired evaluation on documented Hebrew material,
with natural and AI-authored examples reported separately and other construction
methods identified. An advantage for selection would favor that configuration
under the reviewed inventory and scoring rules; an advantage for generation would
favor the less constrained formulation under those same rules. Little observed
difference would limit the case for a practical advantage in this sample, with
uncertainty determining what differences remain plausible. None of these outcomes
would isolate the effect of inventory information from answer format or establish
that one formulation is universally better. Useful evidence does not require a
particular system to win.

## 2 Related Work

**Hebrew abbreviation processing.** Jacobs et al. describe building an acronym
dictionary from unannotated Modern Hebrew text and augmenting it with contextual
information, including expansions absent from the acronym's document
[@jacobs2020acronyms]. HAADS studies Hebrew and Aramaic abbreviations in Jewish Law
documents using contextual and statistical features with machine learning
[@hacohenkerner2010haads]. These provide linguistic precedent in different domains.
The descriptions here rely on the publisher abstracts; detailed split and scoring
procedures were not verified.

**Candidate-based acronym disambiguation.** GLADIS covers English general,
scientific, and biomedical domains. Its dataset splits keep training acronym types
out of validation and test, while AcroBERT scores context–expansion pairs
[@chen2023gladis]. This supports our choice of a task-unseen-type question for the
encoder, without ruling out evaluation on known types for a different question.
GLADIS constructs examples by replacing entity long forms in two source datasets;
its scientific dataset is repartitioned from an existing acronym dataset. That
distinction motivates documenting construction methods rather than treating all
sentences containing acronyms as naturally observed usage.

**Generation and answer interpretation.** Agrawal et al. prompt an LLM to expand
clinical abbreviations without giving it answer choices. A separate resolver maps
the generated string to the candidate with greatest contiguous character overlap
[@agrawal2022large]. An evaluator can therefore use an inventory that the generator
never sees; this is a precedent for separating input from scoring, not a validated
Hebrew matching rule. Meconi et al. study English word sense disambiguation (WSD):
selecting a dictionary definition for a word in context. They also evaluate
candidate-free generation of definitions and explanations with human judgments
[@meconi2025large]. Their comparison highlights the need to specify how free
answers are assessed. Their additional-context and shuffled-definition experiments
also motivate recording prompt content and candidate order. WSD definitions and
acronym expansions are different targets, and selection F1 and human generation
judgments are different scoring procedures; we do not import their scores or rubric.

**Partial-input controls.** Balepur et al. test multiple-choice answering with the
choices visible and the question omitted [@balepur2024artifacts]. Their probes of exact
memorization do not explain the observed choices-only performance, but the
authors explicitly do not rule out all forms of memorization. They also examine
information in individual choices and relations among choices. Our no-sentence
condition retains the acronym, so it is a related control rather than a replication.
We use it to measure performance without sentence evidence, not to identify a
particular internal reasoning or memory mechanism.

**Hebrew representations.** DictaBERT is a pretrained Hebrew encoder whose
pre-tokenizer handles quotation marks used in abbreviations
[@shmidman2023dictabert]. It is the documented backbone candidate for the smaller
ranker. Its model description supports that choice of representation; it does not
validate this project's acronym task or establish pretraining separation.

The comparison below records only details supported by the inspected material.
“Not verified” describes a reading limit, not a claim that a paper omitted a
procedure. No cross-paper score comparison is intended.

| Work | Language/domain | Candidate access | Context | Training/evaluation distinction | Scoring |
|---|---|---|---|---|---|
| Jacobs et al. (2020) | Modern Hebrew | Automatically built dictionary | Contextual information | Split details not verified | Detailed metric not verified |
| HAADS (2010) | Hebrew/Aramaic, Jewish Law | Abbreviation disambiguation; inventory construction not verified | Contextual/statistical features | Split details not verified | Accuracy reported in abstract; procedure not verified |
| GLADIS (2023) | English, multiple domains | Dictionary candidates for ranker | Sentence–expansion pairs | Acronym types separated in task splits | Accuracy and macro F1 |
| Agrawal et al. (2022), clinical sense task | English clinical notes | No choices in generation prompt; inventory in resolver | Clinical snippet | CASI evaluation and transfer of distilled model to substituted MIMIC examples; no type-held-out claim here | Per-acronym accuracy and F1, averaged across acronyms |
| Balepur et al. (2024) | English multiple-choice benchmarks | Answer choices supplied | Full versus choices-only prompts | Prompted evaluation; not an unseen-acronym design | Choice accuracy; invalid outputs receive 0.25 in the main protocol |
| Meconi et al. (2025) | English word senses | Definitions supplied in selection, absent in generation | Target in context; additional-context and ordering checks | Multiple WSD benchmarks; not a type-held-out acronym design | Selection F1; separate human generation rubric |
| Present proposal | Hebrew, natural text as proposed main evaluation | Same LLM with/without type inventory | Sentence present/absent in both formulations | Encoder trained on other acronym types; extension conditional | Proposed macro/micro Top-1 on a common occurrence set |

## 3 Proposed Methods

### 3.1 Task and paired comparisons

A *type* groups occurrences of an acronym; an *occurrence* is one marked use in a
sentence. Its *label* is the expected expansion, while its *inventory* lists the
possible expansions for that type. Accepting one occurrence's label does not
approve every entry in its inventory.

Let an item contain a type \(a_i\), sentence \(x_i\), target span \(s_i\), raw target
surface \(r_i=x_i[s_i]\), and reviewed expansion \(y_i\). Under the approved task
rule, the answer must be a linguistic expansion appropriate to the use in the
sentence. Identifying a referent or giving the historical origin of its name alone
is insufficient. The approved input rule preserves the full target surface,
including an attached prefix, in both sentence and no-sentence conditions. The
sentence must not be used to strip a prefix before constructing a no-sentence
input. Canonical type grouping and its consequences for split membership remain
pending; preserving the input surface does not settle them.

A proposed type inventory \(C(a_i)\) contains independently reviewed expansions
and any explicitly approved aliases. It is shared across the type's occurrences,
rather than constructed around each sentence's answer. A system produces one
expansion or selects one candidate for the identified occurrence.

| Condition | Input | Requested output |
|---|---|---|
| Generation with context | Raw target surface and sentence with target marked | One expansion |
| Selection with context | Same surface and marked sentence, plus fixed inventory | One candidate identifier |
| Generation without context | Same raw target surface, including any attached prefix | One expansion |
| Selection without context | Same raw target surface and inventory/order | One candidate identifier |
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

P1 produced a [source manifest](../data/study_v1/review/source_manifest.json),
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

The approved scope is context-appropriate linguistic expansion. Numerical letter
interpretations and anonymized identities remain proposed exclusion categories
requiring recorded decisions; no global exclusion policy is inferred from the
pilot. Referent names and etymological expansions require case-specific review. The
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
are preserved with their uncertainty and notes; a historical clean label does not
by itself establish eligibility under the current task definition. The existing
[16-item development/training pilot](../data/study_v1/review/pilot_items.csv) uses a
[short rubric](../data/study_v1/review/pilot_rubric.md) with separate label, target,
and inventory checks.

At the agreed 29 September 2026 snapshot (identified in the appendix), all 16
initial answers and occurrence decisions are recorded. The label decisions are
nine acceptances, one explicitly approved correction that remains unapplied,
three proposed exclusions, and three uncertain cases. Recorded target decisions
are separate from these label counts.
These are review records, not applied changes to research data: the inventory is
unapproved, the uncertain cases remain open, and no review times were measured.
The 16-item definition and data-quality review is complete within that scope;
inventory qualification and unresolved semantic decisions remain open. This was
not a model-performance experiment. This deliberately selected pilot does not
estimate the population error rate. There is no empirical
workload estimate and no independent second annotation or agreement measurement.

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

P1 assessed only structural feasibility in
[extension_feasibility.csv](../data/study_v1/review/extension_feasibility.csv).
Eligibility must be based on provenance, separation, and matching constraints,
without system scores or choosing examples to match a test answer. Label coverage
is a descriptive diagnostic after structural selection, not an eligibility filter.
Even a feasible replacement can leave unequal sense coverage, input lengths, or
examples per type. Inventories must therefore be reconciled before matching is
accepted. The extension remains conditional and requires an explicit decision before test
evaluation, independent of test scores. P1 constructed no active replacement
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

An AI coding assistant drafted and revised this manuscript and bibliography,
inspected the specified primary-source passages, and helped prepare protocol and
data-review materials. Shaked selected the research direction, approved the two
task rules described in Section 3.1, and authorized the bounded preparation and
writing packages. These approvals do not approve the remaining protocol settings
or the manuscript text. Existing data documentation attributes earlier human
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

Primary-source claims retained here were checked on 29 September 2026. The scope
below names the inspected passages; it does **not** imply that every page,
appendix, experiment, or released implementation was reviewed. Abstract-only
sources support only the narrow descriptions marked below. These drafting notes
can be condensed during ACL typesetting; the course limit is eight body pages,
excluding references and appendices.

| Reference key | Primary record and inspected scope | Boundary of supported use |
|---|---|---|
| `jacobs2020acronyms` | [Publisher](https://link.springer.com/article/10.1007/s10472-018-9608-8): abstract and metadata | Modern Hebrew; dictionary/context description and non-local expansion. No full-text, detailed split, or score claim. Online publication 2018; journal volume 88, 2020. |
| `hacohenkerner2010haads` | [Publisher](https://onlinelibrary.wiley.com/doi/10.1002/asi.21367): abstract and metadata | Hebrew/Aramaic Jewish Law; contextual/statistical features and ML. Full experimental methods not inspected. |
| `chen2023gladis` | [ACL record](https://aclanthology.org/2023.eacl-main.152/) and [PDF](https://aclanthology.org/2023.eacl-main.152.pdf): §3.1 “Input Corpora” (p. 2075), §3.2 (pp. 2076–2077), §4 input/scoring description and Equations 1–2 (p. 2078), Table 5 caption | English source/domain context; type separation; replacement in two entity datasets versus repartitioning SciAD; pair scoring; named metrics. No validation of our data or loss. |
| `agrawal2022large` | [ACL record](https://aclanthology.org/2022.emnlp-main.130/) and [PDF](https://aclanthology.org/2022.emnlp-main.130.pdf): §4 task, datasets, prompting/resolver, distillation and evaluation paragraphs (pp. 2001–2002) | Candidate-free clinical prompting, character-overlap resolution, CASI-to-MIMIC distinction, per-acronym aggregation. No transfer of the resolver's validity to Hebrew. |
| `balepur2024artifacts` | [ACL record](https://aclanthology.org/2024.acl-long.555/) and [PDF](https://aclanthology.org/2024.acl-long.555.pdf): §2.4 scoring and §3.1/Prompt 3.1 (pp. 10310–10311), §4 prompts/caveat and §4.2, §5 opening definitions (p. 10312) | Choices-only control, invalid-output credit, limited exact-memorization probes, individual/group choice information. No claim that contamination or all memorization is excluded. |
| `meconi2025large` | [ACL record](https://aclanthology.org/2025.emnlp-main.1720/) and [PDF](https://aclanthology.org/2025.emnlp-main.1720.pdf): §3.1 answer extraction, Table 1 (pp. 33898–33899); “Impact of context and candidate order”/Table 5 (pp. 33901–33902); §4.1 task/rubric (p. 33903), Table 20 (p. 33913); §6 (p. 33906) | Selection versus candidate-free generation, separate human rubric, English-only boundary, context/order interventions. No common-score or causal transfer claim. |
| `shmidman2023dictabert` | [arXiv record](https://arxiv.org/abs/2308.16687), [v2 text](https://arxiv.org/html/2308.16687v2): title/authors, abstract and §2.1 | Hebrew encoder and abbreviation-aware pre-tokenizer. Correct metadata replaces the submitted proposal's erroneous author names/identifier; no acronym-performance or non-exposure claim. |
| `owen2012bootstrapping` | [arXiv/journal record](https://arxiv.org/abs/1106.2125), [text](https://arxiv.org/html/1106.2125v3): abstract and §4 opening definition, Equation 12 | Crossed-factor product weights motivate a proposal. Proofs and the project's macro estimator have not been validated here. |

The pilot status uses the unchanged files at repository commit
`63ffa68224e6350446e9913bec2f8de5ae39d8c0`, with agreed cutoff
`2026-09-29T10:04:07.134100+00:00`. Counts come from the pilot IDs joined to
`review_decisions.csv` and the registration fields in `diagnostic_summary.json`.
SHA-256 values identify the inspected snapshot:

| File under `data/study_v1/review/` | SHA-256 |
|---|---|
| `review_decisions.csv` | `c5c6314aacdb5cdf1d41692fabe1c5c259586079f456732200824a0b445cf0bb` |
| `pilot_items.csv` | `7bf4ac1bbe83a0d0c8a85b3e1e129c0c4291c2ba9a6ceade3ea729c97d9a7f90` |
| `diagnostic_summary.json` | `04cdc7b95133faced90386fa4233642227ef55ee834b873076e84983f64ffc92` |
