# Reproduce the analysis or collect new results

## Reproduce the saved test analysis

Read [experimental_study.ipynb](../notebooks/experimental_study.ipynb) directly, or
replay its saved-response analysis without accounts, weights, training or model calls.
From a reviewed, committed checkout, prepare a new external directory:

```bash
ANALYSIS_REVISION=$(git rev-parse HEAD)
ORIGINAL_CHECKOUT=$(pwd)
ANALYSIS_DIR=$(mktemp -d /tmp/hebrew-acronym-analysis-XXXXXXXX)
git archive "$ANALYSIS_REVISION" data docs notebooks paper results saved-results src tests pyproject.toml requirements.txt README.md | tar -x -C "$ANALYSIS_DIR"
python3.13 -m venv "$ANALYSIS_DIR/.venv"
"$ANALYSIS_DIR/.venv/bin/python" -m pip install -e "$ANALYSIS_DIR" nbformat nbclient nbconvert ipykernel
"$ANALYSIS_DIR/.venv/bin/python" -m pip check
"$ANALYSIS_DIR/.venv/bin/python" "$ANALYSIS_DIR/tests/run_saved_analysis.py" \
  --root "$ANALYSIS_DIR" --blocked-root "$ORIGINAL_CHECKOUT" \
  --revision "$ANALYSIS_REVISION" --receipt "$ANALYSIS_DIR/new-execution.json"
"$ANALYSIS_DIR/.venv/bin/python" -m jupyter nbconvert --to html --no-input \
  "$ANALYSIS_DIR/notebooks/experimental_study.ipynb"
```

An archive excludes uncommitted changes. Installation may use the network; execution
uses a fresh kernel and empty model cache. Kernel Python audit hooks reject
`socket.connect`, `socket.getaddrinfo`, `socket.sendto`, `subprocess.Popen`, and
`open` events under enumerated blocked roots or weight-like paths (`weights` path
components; `.safetensors`, `.pt`, `.ckpt`; `.bin`/`.pth` outside `site-packages`).
Add `--blocked-root` for every other research copy. This is **not a security
sandbox**: other process/network routes and directory access are not covered.
Observed reads from a particular audit are empirical evidence, not a guarantee
that every possible access is blocked. The runner probes representative hooks
and records imported package location, versions, hashes and real execution counts. Keep its new receipt and outputs; do not overwrite historical receipts.
The runner executes and records provenance; it does not automatically compare
exports. Separately compare all 23 generated CSV/JSON files byte-for-byte against
the accepted external outputs, including the 13 untracked files (see the
[export index](../results/submission-20261010/README.md)), and inspect Hebrew,
plots, tables and links in the rendered notebook. HTML generation
alone is not visual verification. Dependencies are directly pinned; the complete
transitive environment is not locked.

## New collection

[run_test_eval_colab.ipynb](../notebooks/run_test_eval_colab.ipynb) is the separate
collection entry point. New calls need explicit provider, input and spending settings.
Create a commit archive and extract its notebook without publishing:

```bash
RELEASE_DIR=$(mktemp -d /tmp/hebrew-evaluation-release-XXXXXXXX)
git archive --format=tar --output="$RELEASE_DIR/evaluation-release.tar" HEAD
git show HEAD:notebooks/run_test_eval_colab.ipynb > "$RELEASE_DIR/run_test_eval_colab.ipynb"
git rev-parse HEAD
shasum -a 256 "$RELEASE_DIR/evaluation-release.tar"
```

Upload the notebook and archive to Colab. Set the full `CODE_REVISION`,
`ARCHIVE_SHA256` and `ARCHIVE_PATH`; setup checks source identity and installs the
package. Restart if a changed dependency was already loaded, then rerun setup.
`SOURCE_MODE="git"` requires the exact commit to be remotely accessible.

Select providers with `SELECTED_SYSTEMS`. Put only their keys in Colab Secrets,
using `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`.
The recorded API configurations are Gemini `gemini-3.8-flash`, OpenAI
`gpt-4.1-mini-2025-04-14`, Anthropic `claude-haiku-5-5` and xAI `grok-4.7`.
Saved GPT settings use temperature 0 and 512 output tokens; Claude uses effort
`low` and 1,024 output tokens; Grok uses low reasoning, no tools and 1,024 output
tokens. These are unequal inference budgets. Truncated responses remain automatic
failures in the denominator. Exact identifiers are checked before
requests; a key or successful model lookup does not guarantee credit or inference.
A blocked provider is not silently replaced.

Qwen uses Ollama `qwen2.5:7b` or `qwen2.5:14b` (14B Q4_K_M). Select an available
GPU or explicitly leave the system unrun. Download only for a selected system;
record digest, server version, quantization, template and stops. Both use temperature
0, seed 42 and 512 output tokens. The independent candidate-order seed is 42;
identical bits across hardware/services are not guaranteed.

Mount Drive and choose one stable `RUN_NAME`. Set `PRIOR_SPEND_ILS` explicitly to
all earlier expenditure outside this session, including pilots, failed and uncertain
calls; `None` is not zero. All systems, pilots and retries share the 100 ILS ceiling.
Inspect returned usage, recorded rates and cost allowances. Missing usage is not zero
cost. Projections use a 25% margin and 4 ILS/USD allowance, not a quoted exchange
rate or settled invoice. Changed tariffs require new recorded assumptions.

Each system must complete its own pilot: the same first ten historical development
rows, both tasks, 20 logical responses. Inspect raw answers, finish reasons, technical
coverage and usage. Semantic errors are results, not retry criteria. An incomplete
pilot cannot unlock test; truncation or prompt defects require a corrected, separately
identified pilot. Copy each successful pilot identity to `INSPECTED_PILOTS`.
Full test rechecks source/settings, all pilot outcomes and remaining budget, then
collects 381 × 2 = 762 responses. Preserve failures in denominators.

## Resume and recovery

Resume with the same commit, session, settings, Secrets and Drive directory. Code,
inputs, prompts, candidate order and system identities must match. Completed requests
are not resent. A started call without a completion may have been billed and remains
ambiguous with a conservative cost allowance; deleting it is not recovery. Never use
two writers for one output directory. Storage failures preclude guaranteed exactly-once
billing or durability.

Session totals include journals of systems omitted from the current selection.
A new session carries previous spend exactly once and needs new matching pilots.
A saved ceiling can be tightened, not increased. Technical retries are bounded;
provider deadlines up to 60 seconds may be awaited, longer deadlines pause the
provider. An exhausted attempt cap leaves that response failed and pauses its provider
for at least 60 seconds. Resuming early sends no call. Billing/quota/access blocks
persist in the identified run; after repair use a new session and retain the old logs.
An ambiguous HTTP 429 does not prove billing failure. Other available systems may
proceed. Raw error bodies/headers are not retained; filtered diagnostic categories,
usage and model responses are recorded separately.

## Local DictaLM run

The separate local Ollama route uses the same prompts, candidate order and scoring:

```bash
python -m hebrew_acronyms.run_local_ollama_study pilot --output-dir RUN --code-revision SHA --model TAG
python -m hebrew_acronyms.run_local_ollama_study full --output-dir RUN --code-revision SHA --model TAG --pilot-identity ID
```

Replace `RUN` with a new output directory, `SHA` with the full 40-character code
commit, `TAG` with the exact installed Ollama tag, and `ID` with the inspected
pilot identity. The local Ollama server must already be running; this command
starts no server and downloads no model. Record the digest/template. Settings are
temperature 0, seed 42 and 512 output tokens; no API charge is assigned. The saved
DictaLM template omits a publisher-specified newline; its causal effect is unknown.
Automatic generation, Ben's generation labels, strict selection and diagnostic
leading-label parsing remain separate measurements; no combined score is defined.

## Interpretation and offline checks

Historical 395-item journals remain immutable. Scoring excludes the documented
14 overlapping Knesset items, while accounting includes all collected calls.
Gemini was collected directly on the retained 381; other saved runs can contain
395 items and are scored on the retained cohort.
The test contains observed and AI-authored text; candidate labels beyond Z retain
the two 30-candidate items. Automatic scores, format validity, technical failures and
human judgments remain distinct. The [human-review evidence](../saved-results/README.md)
contains 741 main-review response bindings and 381 separately bound Ben labels;
`fits` means matching the displayed reference, not independent contextual
correctness; selective coverage is not benchmark-wide human accuracy. Known
reference errors and the optimistic most-mined reference are disclosed in the
[data guide](../data/README.md); frozen primary scores retain the historical labels.

[Encoder inference and retrieval](checkpoints.md) use identified weights and a recorded
automatic target policy; they do not reproduce historical training. The
[training example](../notebooks/train_dictabert.ipynb) is optional and separate.

```bash
.venv/bin/python -B -m unittest tests.test_test_eval_notebook tests.test_test_evaluation tests.test_staged_evaluation tests.test_provider_adapters -v
.venv/bin/python -B -m tests.run_notebook
```

These checks use invented state or mocks, not live service performance.
`python -m hebrew_acronyms.pipelines.validate_data --help` describes explicit input
validation. The optional
`python -m hebrew_acronyms.pipelines.check_environment --snapshot /path/to/local/snapshot`
uses invented inputs, blocks sockets and
never downloads; missing cache returns code 2 (not run). Data preparation commands
and reconstruction limits are in the [data inventory](../data/README.md).
