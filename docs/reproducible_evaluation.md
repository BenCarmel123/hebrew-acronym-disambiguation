# Reproducible evaluation in Colab

Use [run_test_eval_colab.ipynb](../notebooks/run_test_eval_colab.ipynb) as the
execution entry point. It installs one reviewed commit, saves requests and outcomes
to Drive, runs the fixed development pilot separately for each selected system,
and gates each 395-item test on successful operation and a combined 100 ILS budget.
This guide does not claim a live Colab run has passed. Offline tests establish software contracts, not model performance.

## Accounts and keys

Create keys only for the approved providers. Keep the values in Colab Secrets;
never send them in chat or put them in a notebook cell.

| Provider | Account action | Colab Secret | Exact model |
|---|---|---|---|
| Google | In [AI Studio](https://aistudio.google.com/api-keys), create/select the project and create its API key. Check its billing/quota before running. | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| OpenAI | Open the API dashboard's [API keys](https://platform.openai.com/api-keys) page, create a project key and check API billing/credits. | `OPENAI_API_KEY` | `gpt-4.1-mini-2025-04-14` |
| Anthropic | Open [Claude Console](https://platform.claude.com/), then **Settings → API keys → Create key**. Check API credit and spending limits. | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| xAI | In [xAI Console](https://console.x.ai), select the key’s team and add prepaid credit. Keep automatic top-up disabled. | `XAI_API_KEY` | `grok-4.7`, low reasoning, no tools |
| Qwen14 | No API account/key. Download only inside the Colab runtime. | None | `qwen2.5:14b`, Q4_K_M, plus resolved digest |
| Qwen | No API account/key. The notebook installs Ollama and explicitly pulls `qwen2.5:7b` into the Colab runtime. | None | `qwen2.5:7b`, plus the resolved digest |
| DictaLM | No API account/key. Runs in a local Ollama server; see [Local DictaLM run](#local-dictalm-run). | None | Researcher-chosen tag, plus the resolved digest |

These steps follow the official [Gemini key instructions](https://ai.google.dev/gemini-api/docs/api-key),
[OpenAI quickstart](https://developers.openai.com/api/docs/quickstart), and
[Claude authentication guide](https://platform.claude.com/docs/en/manage-claude/authentication).
A key does not establish model availability or available credit. The notebook checks
the selected exact model IDs before generating text and each pilot checks actual
requests. A missing key or failed availability check is reported for that system;
other selected systems may proceed.
Do not choose a replacement model if access fails. Shaked handles any payment or
credit purchase; no subscription, paid GPU or automatic replenishment is enabled
by this workflow. All new API charges, including pilots and retries, share 100 ILS.

## Upload a reviewed commit without publishing it

After the reviewed changes have been committed locally, run these commands from
that checkout. An archive of `HEAD` excludes uncommitted changes; check the tree and
use the actual reviewed commit. No push is needed.

```bash
git status --short
git rev-parse HEAD
mkdir -p ../artifacts/colab-packages
RELEASE_DIR=$(mktemp -d ../artifacts/colab-packages/evaluation-XXXXXXXX)
git archive --format=tar --output="$RELEASE_DIR/evaluation-release.tar" HEAD
git show HEAD:notebooks/run_test_eval_colab.ipynb > "$RELEASE_DIR/run_test_eval_colab.ipynb"
git rev-parse HEAD > "$RELEASE_DIR/COMMIT.txt"
shasum -a 256 "$RELEASE_DIR/evaluation-release.tar"
```

The fresh release directory preserves earlier bundles. Record both the 40-character
commit and the archive SHA-256 alongside the archive. Open
[Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**,
and upload `run_test_eval_colab.ipynb` from the fresh release directory. In the
Files panel upload `evaluation-release.tar`. Set `CODE_REVISION`, `ARCHIVE_SHA256`
and `ARCHIVE_PATH` in the first cell. The cell checks both hashes and installs the
complete package with a normal installation, immediately importable in the current
kernel. It checks every project dependency pin and reports unrelated Colab package
conflicts separately. If Colab asks to restart after replacing an already-loaded
dependency, restart and rerun setup. It never downloads individual source files
from a mutable branch.

The alternative `SOURCE_MODE="git"` fetches the exact `CODE_REVISION` from the
repository. It works only if that commit is remotely accessible. Publishing a new
branch still requires the user's separate authorization.

## Run selected systems in stages

The six system identities remain fixed within each new collection session. `SELECTED_SYSTEMS` chooses which to attempt
in the current session; it does not change the shared protocol or erase previously
attempted systems. A blocked provider must not prevent an available system from
completing its own pilot and, if qualified, test run.

1. Set `SELECTED_SYSTEMS` to the available systems. Choose **Runtime → Change
   runtime type → T4 GPU** if Qwen is selected and a free GPU is available. If no
   free GPU is available, assess CPU time or leave Qwen explicitly unrun; do not
   buy compute or replace it. Ollama installation and model download run only
   when either Qwen system is selected.
2. In the key-shaped **Secrets** panel add secrets for selected API providers and
   enable notebook access. Run setup, then approve the Google Drive mount. The
   [official Colab input/output notebook](https://colab.research.google.com/notebooks/io.ipynb)
   describes Drive mounting. Choose a stable `RUN_NAME` for this experiment.
3. Set `PRIOR_SPEND_ILS` to all earlier API costs outside this new session directory,
   including failed attempts and the earlier Gemini pilot. The default `None`
   requires an explicit amount, not an assumption of zero. Confirm uncertain
   amounts against the providers' usage records before live execution. Keep the
   entire experiment, including retries, within 100 ILS.
4. Run the selected availability checks. For Qwen, inspect the digest, template,
   quantization and stops. Explicit Qwen options are temperature 0, seed 42 and
   512 output tokens. The separate candidate-order seed is 42. Returned service
   versions and hardware are evidence; identical bits across services or devices
   are not guaranteed. Inspect the report of unavailable and unselected systems.
5. Run each available system's pilot on the same first ten development rows in
   both tasks: **20 logical responses per system**. All systems retain the same
   item identities, prompts and candidate order. Inspect raw model outputs,
   finish reasons, usage and the technical coverage report. Semantic mistakes
   are results, not a reason to retry. An incomplete pilot does not unlock that
   system's full test. Truncation or prompt defects require a corrected, separately
   identified pilot; a different system's successful pilot is insufficient.
6. Review the displayed usage and time projection. It scales to 395 items with a
   25% cost margin; 4 ILS/USD is a conversion/fee/tax allowance, not a quoted rate.
   Enter each inspected successful pilot's printed identity hash in
   `INSPECTED_PILOTS`. Before full test, its stored identity, source and settings,
   all 20 responses, remaining requests and shared spend are checked again.
7. Run or resume qualified systems' full test: **395 items × 2 tasks = 790 logical
   responses per system**. Retain failed or unrun items in the reported cohort.
   Keep the Drive output and use the download cell for a second copy. Historical
   `results/` directories and earlier release bundles remain unchanged.

Selecting another system later uses the same `RUN_NAME` and session settings.
`session.json` freezes code, inputs, prices, allowances and prior spend. Every cost
check sums journals for **all systems in the session**, including systems omitted
from the current selection, before another call. Pilot and full-test costs share
that total. The summary explicitly records systems not run and their reasons.
Changing selection never resets cost or substitutes for a system's own pilot.

Use a new `RUN_NAME` for changed code/settings or the older combined-directory
layout. Carry all earlier expenditure into `PRIOR_SPEND_ILS` exactly once, preserve
the old directories, and run new matching pilots. Renaming a session or deleting
journals is not a budget reset or a permitted way around an identity check.

Token-based costs are estimates from actual reported usage at the documented
rates, not a settled provider invoice. The runner retains usage and allowances
for uncertain calls and stops before the next request exceeds the configured
budget. Missing usage is not zero cost: the saved per-call allowance is shown
separately from measured token costs and included in the projection. Incomplete
logical responses still block that system's full test even when the allowance
fits the budget. The official rates used are
[Gemini](https://ai.google.dev/gemini-api/docs/pricing),
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), and
[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/overview),
checked on 9 October 2026. The current prompts are short; a changed provider tariff
or long-context tier needs a new recorded price assumption.

## Read provider diagnostics

The error record stores a small diagnostic category, retryability, an optional
HTTP status and normalized waiting time. Provider error bodies are parsed only
in memory; their raw messages and headers are not retained. Model answer text is
saved for evaluation, separately from these filtered error diagnostics.

| Diagnostic | Interpretation |
|---|---|
| `rate_limit` | Explicit evidence of a temporary rate limit, such as a per-minute quota. |
| `quota_exhausted` | Explicit exhausted/zero allocation or daily quota; current collection stops for this provider. |
| `billing_blocked` | Explicit payment/credit problem or HTTP 402; account action is required. |
| `rate_or_quota_unknown` | A 429 without enough evidence to distinguish rate from quota. This is uncertain, not proof of invalid credentials or missing billing. |
| `authentication`, `permission`, `missing_credentials` | The provider cannot be used with the current access configuration. |
| `transient_service`, `connection` | Service or connection failure that may allow bounded retry. |

`provider_blocked` stops collection for that provider; it does not mean the
account is permanently banned. A daily quota and a billing failure are distinct
conditions. The earlier Gemini attempt returned 14 of 20 logical responses and
included 429, 503 and connection failures. It did not pass the technical coverage
gate, and those observations alone do not identify the quota subtype.

Transient retries are bounded by the configured attempt cap. `Retry-After`
(seconds or HTTP date) and Gemini retry hints become a saved provider deadline.
Waits of at most 60 seconds are handled in the runner; a longer instruction pauses
that provider instead of retrying early or sleeping indefinitely. Other providers
can proceed. Exhausting a request's cap pauses its provider for at least 60 seconds
or the longer server delay. Resuming before that deadline sends no request to that
provider. After it expires, later unattempted items may proceed, but an exhausted
item remains failed and still prevents a successful pilot.

An explicit provider block is retained when resuming the same identified run.
After the account is repaired, create a new identified session, preserve the old
journals and include their costs in its prior spend. That new session requires
matching pilots again. Do not remove a failure record or lower prior spend to
force another attempt. The runner's optional cost ceiling can tighten the saved
ceiling; it cannot increase it.

## Resume and interpret

After a disconnect, reopen the same notebook, reinstall the same commit, remount
Drive, reload Secrets and use the same `RUN_NAME` and settings. Preparation compares
input bytes, complete prompts, candidate orders, systems, settings and source
hashes. Qwen identity binds its digest, server version, template, quantization and
settings; download timestamps are retained in separate per-session inspection files,
so a fresh VM with identical model content can resume. Before full test, the notebook
reloads the saved pilot, compares its configuration with current values, and
recomputes that system's coverage and the shared remaining budget. The newly
prepared test manifest must also match the pilot's installed-source hashes before
any test API call. Changing settings after inspecting a pilot cannot reuse its
inspection hash. Completed requests are not resent. A started call with no saved
completion is ambiguous: it may have been billed, and is retained for inspection instead of
being automatically retried. Its conservative allowance remains part of the
shared spend even if another system is selected. Do not delete its log to force
another request.
Never run two writers against the same directory.

Drive contains `session.json` and one directory per system. Each system has
`dev-pilot/` and `full-test/`, with `manifest.json`, `attempts.jsonl` and a
regenerated `summary.json` when prepared. Each request is journalled
before execution and each outcome after return. Drive outages can prevent saving;
this workflow cannot guarantee exactly-once billing or durability during a storage
failure. Keep the mounted storage connected and inspect the journal after an error.

The four deterministic baselines remain runnable through the installed package.
The untrained DictaBERT similarity and trained DictaBERT selection remain distinct
systems. The [checkpoint guide](checkpoints.md) and package
[Colab loader](../src/hebrew_acronyms/models/dictabert_cross_encoder/colab.py)
preserve inference from the existing complete weights; retraining is not required.
The authorized checkpoint SHA-256 is
`23bbff0324a7ce4d9d4a24a9ebfb87a4f6ac296bf1f1ce1bd25126c09260267f`.
A fresh strict encoder run also requires qualified target spans: the historical test
CSV lacks the explicit span columns, and multiple target occurrences must not be
resolved by silently taking the first match. The identified test inference
`dictabert-test-20261009-f49c3b5` uses a researcher-authorized, quote-normalized
first-occurrence policy, recorded with the derived 395-item input and checkpoint
identity. These spans are automatic targets, not human annotations. Its selected
candidates reproduce the historical CSV, without proving that file's execution
origin. Historical encoder results remain separate. The original training seed,
selected epoch, library versions and exact training
rows are not proven by the state dictionary. Record supplied snapshot/tokenizer
identities without inventing that missing provenance.

Do not judge generation correctness from foreign letters alone. Keep automatic
scores, output-quality flags and human semantic judgments separate. Retain all
395 identities and explicit failures. Test contains 267 Knesset, 41 Wikipedia and
87 AI-authored items; it is not natural-only. Candidate selection now supports
labels beyond Z, retaining the two 30-candidate items. This protocol change must
remain visible when comparing historical and new results.

[experimental_study.ipynb](../notebooks/experimental_study.ipynb) remains the methods
and analysis companion. The existing paper exporter accepts selected full-dev
artifacts; do not present it as a full-test exporter. Any final table must identify
its selected run, cohort, scoring rule and judgment provenance.

## Offline checks

```bash
.venv/bin/python -B -m unittest tests.test_test_eval_notebook tests.test_test_evaluation tests.test_staged_evaluation tests.test_provider_adapters -v
```

Notebook checks compile the Python cells and exercise setup/gating contracts with
invented state. Runner tests cover identity checks and recovery without paid API,
training, live Colab or research performance claims.

## Local development study and training

This development workflow uses its own settings and saved-run format. Its
`enable_*` switches and resume command below do not replace the Colab test
runner's per-system pilot checks or shared session budget.

From this checkout, create an isolated environment and open the main notebook:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . jupyterlab ipykernel
.venv/bin/python -m pip check
.venv/bin/python -m jupyter lab notebooks/experimental_study.ipynb
```

Select the `.venv` kernel. Project dependencies are pinned in `requirements.txt`
through `pyproject.toml`; the complete transitive environment is not locked. The
checkpoint also records the required model, tokenizer and library versions. For
installation troubleshooting, use a fresh environment rather than import-path overrides.

**Qwen:** provision `qwen2.5:7b` in local Ollama, then check `ollama list` and
`curl http://localhost:11434/api/tags`. The notebook connects to the existing service;
it does not start it or download models.

**Gemini:** the selected model is `gemini-3.8-flash` with
`{"thinkingConfig": {"thinkingLevel": "low"}}`. Low thinking is not disabled thinking.
Copy `.env.example` to `.env` in the checkout root and set `GEMINI_API_KEY` there.
The study loads this file when Gemini is enabled, using the notebook's `root`
setting. Existing environment variables take precedence. Restart the kernel after
changing the key. Keep `.env` local (Git ignores it); never put keys in notebook
cells or settings. See the [official model settings](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).

The systems can run independently. The basic notebook request has a 120-second timeout
and no automatic retries. The explicit `resume_gemini_study` runner below bounds
transient retries and saves each attempt. Qwen records its server-reported digest;
Gemini records the returned
`modelVersion`. Partial answers are retained and marked incomplete. Checkpoints and
live services still need the manual validation below.

### Configure and run the development study

The [main notebook](../notebooks/experimental_study.ipynb) follows inputs → encoder →
LLMs → results. Its centralized settings include:

| Setting | Value or required input |
|---|---|
| `train_path` | Qualified `data/study_v1/encoder_inputs/train.csv`, used to check the checkpoint's training inputs. |
| `input_path` | Qualified `data/study_v1/encoder_inputs/dev.csv`, containing all 62 dev items. |
| `checkpoint` | Exact checkpoint path. Default `checkpoint_format="manifest"` requires the original adjacent JSON; the explicit Colab route below accepts an authorized complete state dictionary. |
| `snapshot_path` | A content-identical local base-model/tokenizer snapshot if relocated; otherwise `None`. |
| `device` | `"cpu"` by default; set a supported `"mps"` or `"cuda"` explicitly if needed. |
| `enable_encoder`, `enable_qwen`, `enable_gemini` | Independent switches, initially `False`. |
| `output_root` | `<project>/artifacts/study-runs`; each run creates a fresh directory outside Git. |
| `saved_run`, `saved_run_id` | Exact result JSON path and run ID for reload. |

For manifest-bound encoder prediction or saved-prediction reuse, the study compares the complete
qualified train/dev rows with the checkpoint's recorded inputs. Validation checks
against the full dev file, even though it predicts only three items. The compact
checkpoint summary shows training settings, selected epoch, train/dev counts and
match status; full metadata stays in the saved result. See [checkpoint details](checkpoints.md).

- **Preview:** leave `mode="preview"` for invented examples with no model calls or research reads.
- **Validation:** set `mode="run"`, `run_kind="validation"` and enable the desired systems. DictaBERT predicts three items; each LLM sends one generation and one selection request, four requests with both enabled.
- **Full dev:** after manual validation succeeds, rerun from fresh settings with `run_kind="full_dev"`. All 62 items are requested per enabled arm: **248 LLM requests** with both providers.
- **Reload:** set `mode="reload"`, `saved_run` and `saved_run_id`. Inspection needs no models, services, credentials or original research files. Older three-arm results remain readable as three-arm results.

The notebook defines the scoring rules and interpretation limits beside the results.
Expand an item to inspect its context, candidate mapping, answers and failure details.
To reuse encoder predictions in a new run, set `enable_encoder=False`, `saved_encoder`
to the source `study.json`, and `saved_encoder_run_id` to its run ID. Keep the current
train/dev paths: reuse must pass the same input comparison as fresh prediction.

### Offline study checks

These checks use invented inputs and mocked responses, without training or live services:

```bash
.venv/bin/python -B -m unittest tests.test_experimental_study tests.test_five_arm_study tests.test_study_evaluation tests.test_qwen_study tests.test_gemini_study tests.test_checkpoint_relocation tests.test_encoder_study_inputs -v
```

Network and research-file guards apply only inside the test processes. The wider
repository suite separately includes tiny-model learning tests.

### Training appendix

Open [train_dictabert.ipynb](../notebooks/train_dictabert.ipynb) in a notebook editor using
the installed environment. Imports and settings appear first, then inspectable steps:
input rows → candidate pairs → model → optional training → predictions.
Jupyter and a notebook kernel are not installed by the project dependencies.

The default (`TRAIN=False`, `LOAD_CHECKPOINT=None`) uses two invented rows and an
existing local DictaBERT snapshot, selected with `SNAPSHOT` or discovered in local
cache. No training, research-file reads or services occur. A missing snapshot raises
an explicit error; no model is downloaded. Initial rankings come from a random head.
For a cache-free engineering execution of all cells with a tiny injected model:

```bash
.venv/bin/python -B -m tests.run_notebook
```

The test runner blocks sockets and research-file access in its own process and
executes unchanged cells. It performs inference only. The single tiny learning check
runs separately in the contract suite. Ordinary notebook setup does not install a
permanent socket blocker.

For separately authorized training, choose explicit `TRAIN_PATH`, `DEV_PATH` and a
fresh `CHECKPOINT_PATH`, then set `TRAIN=True`. The notebook validates inputs, builds
pairs, initializes the model, trains and reloads the best development-loss checkpoint.
For checkpoint inference, set `LOAD_CHECKPOINT` with `TRAIN=False`; saved settings
are restored. Change the explicit prediction input when using qualified data. No test
file or other research input is selected automatically.

Items require unique `item_id`, original `sentence`, exact `target_raw` including any
prefix, half-open `span_start`/`span_end`, and pipe-separated `candidates`. Training
requires `gold_expansion` to match exactly one candidate after surrounding whitespace
is trimmed, with at least two distinct candidates. Prediction preserves singleton
items and identified failure records. Context may be cropped; targets, markers and
candidates are retained or the input fails explicitly. No aliases or metrics are inferred.

### Development and training limits

The encoder uses the existing pooling, BCE loss, AdamW and strict development-pair-loss
checkpoint selection. Defaults in [TrainingConfig](../src/hebrew_acronyms/models/dictabert_cross_encoder/training.py)
are implementation settings, not a finalized research protocol. The default loader requires a
JSON manifest and matching model/tokenizer, inputs when supplied, and library identities;
see [checkpoint details](checkpoints.md). Unspecified weight-only files are rejected.
The explicitly selected Colab adapter below is a separate, provenance-labelled exception.
An explicit relocated snapshot is accepted only after content identity verification;
the original manifest is not rewritten. GPU training and research performance have
not been verified by the tiny CPU checks.

Qualified dev inputs and preliminary selection scoring are available. Final benchmark
runs and generation judgment rules remain separate work. Natural, substituted and
AI-authored material must remain identifiable; see the data documentation. AI assistance
contributed code, checks and draft prose, not human annotation or scientific validation.
Weights, caches, environments, secrets and raw run outputs stay outside version control.
The explicitly selected paper run may publish small derived PDF figures, TeX tables
and provenance manifests under `paper/generated/`; preview and fixtures cannot do so.

### Separate development runs and Colab checkpoints

The main notebook accepts `checkpoint_format="colab_state_dict"`, an explicit
`checkpoint_sha256`, local `snapshot_path`, and the supplied `checkpoint_attestation`.
The [Colab adapter](../src/hebrew_acronyms/models/dictabert_cross_encoder/colab.py)
reconstructs CLS pooling, length 256, `[ACR]`/`[/ACR]`, and the original pair encoding
from the inspected Colab notebook. It verifies the exact checkpoint hash and strictly
loads every encoder and head tensor, validates three items, then predicts the remaining
cohort. Original training seed, library versions and exact training-row identity remain
unknown unless separately evidenced. Reconstruction evidence is not an original manifest.

To run or resume Gemini with the exact prompts and candidate orders from a saved Qwen
full-dev source (explicit service authorization is required):

```bash
python -m hebrew_acronyms.resume_gemini_study --root . \
  --source /path/to/qwen/study.json --source-run-id EXACT_QWEN_RUN_ID \
  --output-root /path/outside/repository --run-id NEW_GEMINI_RUN_ID
```

The same command resumes unattempted items, never resends completed responses, and
retains ambiguous interrupted requests for inspection. Each HTTP request is saved
separately; transient failures permit at most three attempts per item with Retry-After
and backoff. Provider cooldown deadlines are saved before waiting, including after
an exhausted retry sequence; no individual wait exceeds 60 seconds. A longer delay
returns control and remains binding across resume and task changes. Five consecutive
failed items stop collection with remaining items marked unrun. An explicit quota,
billing or access block stops immediately and persists across resume; after account
repair, use a new identified run. Other transient failures can resume after service
recovery; existing terminal failure records are preserved.

The notebook's final comparison cell accepts `(study.json path, original run ID)`
entries for independently saved runs. It checks complete input identity and prompt
agreement, retains original arm run IDs, and displays all 62 items without model or
network calls. Selection includes service/format failures in its denominator; generation
remains semantically unscored. Review gold labels and candidate inventories with Ben,
including institutional uses, before interpreting these diagnostic development scores.

## Research artifacts and reporting

The [paper build/export guide](../paper/README.md),
[bibliography](../paper/references.bib) and [course sources](course/README.md)
preserve the research context. The ID-named PDF in the repository is the submitted
proposal. A manuscript PDF must be built from the current sources and visually
checked; the previously tracked `paper/draft.pdf` was removed. The manuscript is a
working draft: pending values and figures must be reconciled with the selected
results before submission.

[Historical summaries](../results/all_arms_summary.md) and
[training records](checkpoints.md) refer to earlier inputs and include unresolved
score discrepancies. Keep them distinct from the saved test CSVs and new runs.
Earlier work also exists in `ShakedSchnarch/nlp-hw-team`; its checkpoints and
implementation are not interchangeable with this repository. Historical
[Colab training](../notebooks/train_dictabert_colab.ipynb) sets its seed after model
initialization; it does not establish the original checkpoint's training seed.
The [Qwen scale notebook](../notebooks/run_qwen_scale_colab.ipynb) is an optional
32B/72B experiment with substantial GPU requirements, separate from the default
test entry point; its existence is not evidence that those runs completed.

## Additional collection systems

The extended package adds `xai` and `qwen14` as separate systems. It defaults to
those two selections, preserving the previous four systems' settings. Do not
rerun completed systems when opening a new package. Existing `a319f9d` sessions,
including the successful Gemini pilot, must resume using their original package.
A new session carries previous total expenditure once, including uncertain costs.

Grok 4.7 uses the Responses API, low reasoning effort, no tools, and a 1,024-token
output cap. The [documented rates](https://docs.x.ai/developers/models/grok-4.7)
are USD 2 input and USD 6 output per million tokens. Token-based accounting charges
cached input conservatively at the full rate; returned provider cost ticks remain
in the usage evidence. Reasoning tokens are included once, and inconsistent usage
receives the existing uncertain-cost reserve. Model lookup alone does not establish
billing or successful inference.

Qwen14 uses temperature 0, seed 42 and 512 output tokens, as for Qwen7B.
The [14B distribution](https://ollama.com/library/qwen2.5:14b) uses Q4_K_M;
the notebook checks quantization and records its complete resolved digest,
template and runtime settings before collection. Model weights are downloaded
only into Colab. A missing free GPU or insufficient runtime memory is a reported
resource limitation, not authorization to purchase compute or substitute a model.

Comparison accepts the two explicitly identified collector implementations, while
checking identical test inputs, prompts, candidate orders and scoring sources.
Model configurations and collection revisions remain separate in the output.

## Local DictaLM run

`dictalm` is a further Ollama system, run on the same prompts, candidate orders and
scoring as `qwen` and `qwen14`. It uses temperature 0, seed 42 and 512 output tokens.
The model tag and quantization are chosen by the researcher and recorded with the
server-reported digest and chat template; check that the template matches the
model's published instruction format before reading any result. Local runs need no
account and cost nothing, so all prices are zero.

```
python -m hebrew_acronyms.run_local_ollama_study pilot --output-dir RUN --code-revision SHA --model TAG
python -m hebrew_acronyms.run_local_ollama_study full  --output-dir RUN --code-revision SHA --model TAG --pilot-identity ID
```

The staged workflow is fixed to the 395-item test set it was audited on.
