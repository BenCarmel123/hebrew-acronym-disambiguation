# Dataset construction and limitations

The current [inventory](../README.md) distinguishes source exports from encoder
inputs. The test cohort contains 253 Knesset, 41 naturally occurring Wikipedia and
87 Claude-authored sentences. Natural text is not automatically a verified label;
authored text is not observed usage.

Wikipedia article plaintext and Wiktionary senses supplied candidate expansions.
Knesset contexts came from
[HaifaCLGroup/KnessetCorpus](https://huggingface.co/datasets/HaifaCLGroup/KnessetCorpus).
Source fields and document titles retain attribution. Candidate `hits` count articles
containing an expansion, not acronym occurrences or natural frequencies, and miss
proclitic variants. Parser precision was not measured. Duplicate decisions survive in
[duplicate_review.csv](duplicate_review.csv); possible meanings need not be attested
in the corpus. Historical filtering rejected Latin-script/single-letter types and
some numeral, honorific, zero-hit and unresolved-sense cases. A historically rejected
pair can still occur in the retained candidate table: it is not an inventory wholly
approved by humans.

Wikipedia construction combined observed acronym contexts, substitution of spelled-out
expansions and removal of adjacent explanations. Co-occurrence with an expansion gave
provisional labels, not semantic verification. Filters checked text length, boundaries,
other acronyms, glosses and non-prose residue. Substitution can alter meaning or grammar;
repeated targets, attached prefixes and quote variants remain material. Honorific and
numeral filters differ by path. `sense_id` predates reranking and must not be inferred
from candidate order. Mining quotas do not estimate real-world frequency.

Knesset review retained 1,041 of 1,045 mined rows: 941 clean, 100 corrected and four
removed. [Applied review](knesset/knesset_reviewed.csv) remains; the earlier mining
snapshot is archived. Review introduced gematria and privacy-redaction candidates.
Human work is attributed to Ben by Shaked, without evidence of independent double
annotation. [Development review](dev_review.csv) contains 283 records from an older
snapshot; it is not full review of the current 289 development rows. Uncertainty
and adverse decisions remain, and some historical records cannot be linked
unambiguously to current items.

The authored additions declare `claude-sonnet-5` / `claude_authored`: 142 training
examples for thinly represented senses and 87 test examples using alternative
existing candidates. Those model declarations are preserved, not independently
verified. Four aggregate examples declare human authorship; two remain in historical
dev. The writer-supplied label does not establish independent review.

The early 16-item preparation pilot was deterministically stratified: four natural-dev,
four natural Wikipedia train, four routine Knesset train and four central-risk cases,
with diversity/risk ranking and fallback for shortages; test was excluded. The intended
sequence recorded an initial expansion before showing stored labels, then label,
target and inventory decisions. CSV presentation could not enforce blindness.
Shaked's recorded reasons disclose AI assistance and exposure; blank or missing
records are not approvals. No inter-annotator agreement or measured time-saving claim
is established. [Decisions](../study_v1/review/review_decisions.csv) retain all 16
populated records; [local approvals](../study_v1/encoder_inputs/local_decisions.json)
retain the 20 exact item decisions and the separate type-separation rule.

The immutable [item audit](../study_v1/review/item_audit.csv) embeds relevant mining
and review evidence in full. Its source is the historical `p1-proposal-v2` audit
at source commit `64e68af00e5c9dee7942954d51c655f633a51502`; source manifests and
intermediate proposals remain archived. References count CSV records from one,
excluding the header, not physical lines. Stable item identities bind the exact
acronym, sentence, label and candidates.

Later additions are not fully reconstructed by retained scripts: current encoder
train has 2,829 items, while retained preparation yields 2,665. Exact-string/document
checks do not exclude related acronym families, document aliases or pretraining
exposure. Original online mining and historical training have not been reproduced.
