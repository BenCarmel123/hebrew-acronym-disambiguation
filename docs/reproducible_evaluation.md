# Reproducible evaluation in Colab

Use [run_test_eval_colab.ipynb](../notebooks/run_test_eval_colab.ipynb) as the
execution entry point. It installs one reviewed commit, saves requests and outcomes
to Drive, runs the fixed development pilot, and gates the 395-item test on successful
operation and a combined 100 ILS budget. This guide does not claim a live Colab run
has passed. Offline tests establish software contracts, not model performance.

## Accounts and keys

Create keys only for the approved providers. Keep the values in Colab Secrets;
never send them in chat or put them in a notebook cell.

| Provider | Account action | Colab Secret | Exact model |
|---|---|---|---|
| Google | In [AI Studio](https://aistudio.google.com/api-keys), create/select the project and create its API key. Check its billing/quota before running. | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| OpenAI | Open the API dashboard's [API keys](https://platform.openai.com/api-keys) page, create a project key and check API billing/credits. | `OPENAI_API_KEY` | `gpt-4.1-mini-2025-04-14` |
| Anthropic | Open [Claude Console](https://platform.claude.com/), then **Settings → API keys → Create key**. Check API credit and spending limits. | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| Qwen | No API account/key. The notebook installs Ollama and explicitly pulls `qwen2.5:7b` into the Colab runtime. | None | `qwen2.5:7b`, plus the resolved digest |

These steps follow the official [Gemini key instructions](https://ai.google.dev/gemini-api/docs/api-key),
[OpenAI quickstart](https://developers.openai.com/api/docs/quickstart), and
[Claude authentication guide](https://platform.claude.com/docs/en/manage-claude/authentication).
A key does not establish model availability or available credit. The notebook checks
the exact model IDs before generating text and the pilot checks actual requests.
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
git archive --format=tar --output=../evaluation-release.tar HEAD
shasum -a 256 ../evaluation-release.tar
```

Record both the 40-character commit and the archive SHA-256. Open
[Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**,
and upload `notebooks/run_test_eval_colab.ipynb` from this same checkout. In the
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

## Run the notebook

1. Choose **Runtime → Change runtime type → T4 GPU** if a free GPU is available.
   If it is unavailable, assess Qwen's CPU time before continuing; do not buy
   compute or silently remove/replace Qwen.
2. In the key-shaped **Secrets** panel add the three named secrets and enable
   notebook access. Run setup, then approve the Google Drive mount. The
   [official Colab input/output notebook](https://colab.research.google.com/notebooks/io.ipynb)
   describes Drive mounting. Choose one stable `RUN_NAME` for this experiment.
3. Run the Qwen and model-availability cells. Inspect the Qwen digest, template,
   quantization and stops. Explicit Qwen options are temperature 0, seed 42 and
   512 output tokens. The candidate-order seed is separately 42. Service model
   versions and hardware remain part of the evidence; identical bits are not
   guaranteed across services or devices.
4. Run the first ten development rows in both tasks for all four models: 80
   logical responses. Inspect identities, raw outputs, finish reasons and usage.
   Semantic mistakes are results, not a reason to retry. Truncation or prompt
   defects must be resolved in a new identified pilot before full test.
5. Review the displayed usage-based cost and time projection once. It scales to
   395 items with a 25% cost margin. The conversion allowance of 4 ILS/USD includes
   room for conversion/fees/tax; it is not a quoted exchange rate. Set
   `PRIOR_SPEND_ILS` to all earlier experiments outside this pilot/full pair. Only
   mark `PILOT_INSPECTED=True` after checking the output. A successful inspected
   pilot and an estimate within 100 ILS unlock full test, under existing approval.
6. Run or resume full test. Keep the Drive folder and use the download cell for
   a second copy. Do not write into historical `results/` directories.

Token-based costs are estimates from actual reported usage at the documented
rates, not a settled provider invoice. The runner retains usage and allowances
for uncertain calls and stops before the next request exceeds the configured
budget. Missing usage is not zero cost: the saved per-call allowance is shown separately
from measured token costs and included in the projection. Incomplete logical
responses still block full test even when the allowance fits the budget. The official rates used are
[Gemini](https://ai.google.dev/gemini-api/docs/pricing),
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), and
[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/overview),
checked on 9 October 2026. The current prompts are short; a changed provider tariff
or long-context tier needs a new recorded price assumption.

## Resume and interpret

After a disconnect, reopen the same notebook, reinstall the same commit, remount
Drive, reload Secrets and use the same `RUN_NAME` and settings. Preparation compares
input bytes, complete prompts, candidate orders, systems, settings and source
hashes. Qwen identity binds its digest, server version, template, quantization and
settings; download timestamps are retained in separate per-session inspection files,
so a fresh VM with identical model content can resume. Before full test, the notebook
reloads the saved pilot, compares its configuration with current values, and
recomputes coverage and remaining budget. The newly prepared test manifest must
also match the pilot's installed-source hashes before any test API call. Changing settings after inspecting a
pilot cannot reuse its success flag. Completed requests are not resent. A started call with no saved completion
is ambiguous: it may have been billed, and is retained for inspection instead of
being automatically retried. Do not delete its log to force another request.
Never run two writers against the same directory.

Drive contains `dev-pilot/` and `full-test/`, each with `manifest.json`,
`attempts.jsonl` and a regenerated `summary.json`. Each request is journalled
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
resolved by silently taking the first match. That integration remains pending;
historical encoder results are preserved separately. Its original training seed,
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
.venv/bin/python -B -m unittest tests.test_test_eval_notebook tests.test_test_evaluation -v
```

Notebook checks compile the Python cells and exercise setup/gating contracts with
invented state. Runner tests cover identity checks and recovery without paid API,
training, live Colab or research performance claims.
