# Historical construction and review record

This card records the rationale, annotations and limitations of earlier construction
stages. It does not approve a new benchmark. [data/README.md](../README.md) is the current
inventory and field guide; counts here refer to their stated historical stage.
Human-review work is attributed to Ben by Shaked. Earlier documentation often said
“the user”; the recorded verdicts do not establish independent annotator agreement.

The full earlier descriptions remain in local Git at
`63b90acfae36e6b7fee8114760506868e29c681f` (`data/mined/DATASET_CARD.md`,
`data/mined/README.md`, `data/splits/README.md`). For example:

```bash
git show 63b90acfae36e6b7fee8114760506868e29c681f:data/mined/DATASET_CARD.md
```

## Sources and candidate inventory

The original task formulation contrasted free expansion generation with selection
from a candidate list. Literal expansions and the treatment of other candidate kinds
must be settled explicitly for the next experiment.

- **Wikipedia:** the MediaWiki API supplies namespace-0 article plaintext through
  `prop=extracts&explaintext`. An abandoned snippet miner had about 32% usable text
  after repeated filtering: search snippets included references, captions and tables.
  The existing code therefore mines article text, not search-result snippets.
- **Wiktionary:** supplies candidate expansions and some usage examples. Historical
  source counts were 3,678 types/701 polysemous types, versus Wikipedia's 117/~80;
  50 types overlapped. These are construction-stage counts, not current coverage.
- **Knesset:** [HaifaCLGroup/KnessetCorpus](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus)
  supplies pre-segmented parliamentary sentences. The recorded pass used 1,000 plenary
  shards with a cap of 15 rows per acronym. Committee material was not pursued after
  a partial download because of review capacity.

Wikipedia bullets and Wiktionary senses were counted, merged and reviewed for duplicate
expansions. [duplicate_review.csv](duplicate_review.csv) preserves the decisions.
Latin-script types and the single-letter `א'` were removed at that stage. Earlier
commits `caffc54`, `6f89c3b`, `5f27b78` cited in historical prose are absent after a rebase;
the review records, rather than those hashes, are the surviving evidence.

The inventory changed during construction (earlier descriptions give 2,264 rows/680
and 2,002 rows/638 types). Candidate rows describe possible meanings, not sentence labels.
`hits` counts Wikipedia articles containing the expansion, not occurrences, and misses
proclitic variants. Prefix collisions make it a poor proxy for acronym usage.
`mined_items` and `mined_natural` record yield. At the earlier 2,002-row stage, 762
candidate senses produced no contexts; dictionary presence did not establish attestation.

## Wikipedia mining and filtering

Three retained strategies have distinct purposes:

| Function | Text and label evidence |
|---|---|
| `mine_sentences` | Natural contexts found by acronym search, without a sense label; retained for exploration, not the original benchmark construction. |
| `mine_by_expansion` | Natural contexts from pages also mentioning an expansion; that co-occurrence provides a provisional sense, not proof. |
| `mine_substituted` | Replaces a spelled-out expansion with its acronym, preserving an attached proclitic; the target is mechanically derived. |

An early `מ״מ` run found only the millimetre sense in flat mining; expansion-specific
natural mining found 2 of 17 expansions, while substitution found 14. The original
sweep selected 638 types, found contexts for 546 and none for 92. After removing 24
exact acronym/sentence duplicates it held 3,386 items: 128 natural and 3,258 substituted.
At that stage, 295 types/2,664 items met the ≥2 senses with ≥2 rows mining flag. Earlier
recommendations to exclude other types were not a final scientific approval.

The code checks length (40–200 characters), target boundaries with permitted Hebrew
proclitics, other acronym types, glosses, explicit acronym-definition phrases, non-prose
residue and numeral references. A spaced dash can mark a gloss; a tight compound hyphen
is retained. Substitution also rejects an already-present acronym or a defining sentence.
Candidates containing another acronym were skipped during mining; 230 such candidate
rows remained in the earlier inventory. Their future treatment is unresolved.

Known limitations preserved in the code and data:

- Substitution can replace a compositional phrase (`מכל מקום בו הרכב נמצא`) or only
  part of a longer term (`מרחב מכפלה פנימית`), damaging meaning or grammar.
- Honorific handling differs: general numeral detection exempts `ר ד ע`, but the
  substitution path uses the bare numeral regex and can discard `ר׳` contexts.
- The different-acronym check compares a set and therefore permits repeated targets.
  The original 3,386 rows had 3,260 single occurrences and 126 multiple occurrences
  under quote folding (up to seven), without an annotated target span.
- Exact quotation-mark matching missed 77 target occurrences found after folding
  ASCII quotes and gershayim. Keep source bytes while matching leniently.
- Only 1,662 original `sense_id` values matched a later candidate rank; IDs were
  assigned before re-ranking and must not be reconstructed from current order.
- Per-expansion quotas reveal rare meanings but do not estimate natural frequency.

## Development review (2026-09-12)

[dev_review.csv](dev_review.csv) records a review of the historical 285-row development
snapshot, described at the time as substituted text:

| Verdict | Rows |
|---|---:|
| clean | 227 |
| wrong_sense | 11 |
| broken | 7 |
| unsure | 38 |
| unreviewed | 2 |

The earlier report gave a damage rate of 18/245 decided rows (7.3%, reported 95% CI
4.1–10.6%); this is a historical calculation, not a current quality estimate. Of the
38 unsure cases, 29 concerned rabbinic-name types whose sentences often lacked enough
biographical context. They were retained with flags rather than silently removed.

Review exposed a named-entity usage (`דו״ד`, a programme named after `דין ודברים`), a
candidate-description error (`יעב״ץ` → `מבעלי התוספות` rather than `יעקב בן צבי`), and
49 dev candidate slots with neither corpus hits nor mined items. Seventeen rows had
only one corpus-supported choice under that historical diagnostic.

Fourteen rows were relabelled and four dropped, leaving 281. Corrected expansions
were added across all rows of the affected type, not only corrected examples, so the
candidate list did not signal which item was edited. `label_origin`, `review_verdict`
and `review_note` preserve the changes. Automated flagging had caught only five of
311 rows in an earlier check and was not a substitute for the review.

## Deglossed and authored additions

`mine_deglossed` removes a parenthetical explanation from an observed acronym usage.
The recorded sweep of 368 expansions with hits but no mined items produced 18 usable
rows over 12 types: six added to dev under `להב״ה`, twelve to train. A named-after-person
`מס״ב` example was discarded; `NAMED_AFTER_RE` was added for that phrasing. Later
allocation changed the file composition, so these counts are not current split totals.

Four examples for `בא״ח` = `בסיס אימונים חטיבתי` and `אז״ר` = `אויב זרק רימון` were
historically described as hand-written (two per sense). Their surviving aggregate rows
declare `human` / `human_authored`; two `בא״ח` rows remain in dev. The writer supplied
the label, which is not evidence of independent review.

Further additions explicitly disclose AI authorship (`claude-sonnet-5`,
`claude_authored`, `provenance=authored`):

- **87 test examples:** three per each of 29 types whose reviewed Knesset examples
  shared one gold sense. A different existing candidate sense was chosen, and the
  recorded construction checked its membership in `candidate_table.csv`.
- **142 train examples:** added for 58 thin types to bring each sense to at least
  three examples. Deeper Wikipedia searches had been attempted first; the authored
  route, not those exploratory mining outputs, supplied the recorded addition.

Authored rows are evidence of generated examples, not naturally observed ambiguity.
The category name `manual` does not identify their author; current counts and the four
human-declared aggregate rows are distinguished in [the inventory](../README.md).

## Knesset review and split construction (2026-09-13)

The recorded mining pass produced 1,045 rows/193 types. Human review against candidate
lists retained 1,041 rows: 941 clean, 100 corrected and four dropped. Dropped cases
included two clitic/tokenization artifacts, a unit transcription error and a garbled
token. [knesset_reviewed.csv](knesset/knesset_reviewed.csv) preserves applied review;
[knesset_mined.csv](knesset/knesset_mined.csv) preserves the mining output.

Review introduced gematria candidates across all 638 inventory types and the generic
privacy-redaction candidate `ראשי תיבות שם פרטי ומשפחה (זהות חסויה)` for confirmed
initial placeholders. Those are historical annotation decisions, not approval to include
such meanings in a future literal-expansion benchmark. Distinguishing a genuine acronym
beginning with a clitic letter from a prefixed shorter acronym remains a mining limitation.

Three accidental candidate duplicates (`ח״י`, `ל״ב` gematria and `י״ל` name spelling)
were merged; 17 reviewed rows were redirected to the canonical forms. About 39 similar
pairs were retained as distinct meanings, such as `רבי אלעזר` versus `רבי אליעזר`.
Spelling similarity alone was not a reason to merge them.

The historical natural-text pool added 112 Wikipedia rows after excluding dev-owned
types and rows without gold. All 191 reviewed Knesset types already belonged to the
old train/dev inventory, so test construction required removing selected types' old
Wikipedia rows from train. The retained `build_splits.py` performs the earlier Knesset
allocation step; it does not alone reproduce all later natural/authored additions.
The historical final allocation selected 60 test types (stratification plus forced
manual-example types), removed 514 old train rows and capped other pool additions at
8 rows/type. Dev was left unchanged by that construction step.

Earlier documentation called these splits “frozen” and accepted document overlap as a
capacity tradeoff. That is a historical rationale, not Shaked's approval of the next
protocol. Current overlap and ID collisions are listed in [the inventory](../README.md).
`multi_sense_type` was left blank for later Knesset/natural/authored additions rather
than guessed; a simpler distinct-gold-count rule disagreed on 40 of 289 dev rows.

## Retry evidence and abandoned sources

[wikipedia/unmined_triage.csv](wikipedia/unmined_triage.csv) and [merge_review.csv](merge_review.csv)
record two passes over unsuccessful types: whether the parser found an acronym at all,
and whether proposed expansions were genuinely different senses. Of 80 marked minable,
37 were subsequently excluded in that historical pass: 26 with zero corpus hits,
seven rabbinic-name types, three left with fewer than two senses after merging, and `גר'`.
The remaining 43 are retained in [retry_types.txt](wikipedia/retry_types.txt).

The recorded retry used three contexts per expansion, depth 40 instead of 12 pages,
and resume. [retry_by_sense.csv](wikipedia/retry_by_sense.csv) and its summary retain
15 sentences from seven types, each with one attested sense; none entered training.
`עמ״נ` yielded only `על-מנת` despite 46,031 hits, illustrating why deeper search did not
establish a rare alternative. The earlier claim that all unmined types were exhausted
was an interpretation of that trial, not proof of corpus-wide absence.

Sefaria was tried for rabbinic material and abandoned after reported yield of roughly
3%: fetched Hebrew segments were not reliably sentence-granular. Its unused exploratory
code was removed from the active tree; recover it from Git at
`63b90acfae36e6b7fee8114760506868e29c681f:data_preprocess/sefaria/`.
Hebrew Wikisource was also noted as a possible fallback but was not integrated.
No live source is needed for the safe structural checks.
