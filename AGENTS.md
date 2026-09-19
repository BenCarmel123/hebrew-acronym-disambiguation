# Agent operating rules

- Read the current user task and [README.md](README.md) before work. The central
  `PROJECT_PLAN.md` is maintained by the coordinator outside this repository; do not
  copy, move or update it, or create competing PLAN/STATUS/HANDOFF/TODO documents.
- Obtain the current authorized package from the user task and central plan; do not
  turn an earlier package boundary into a permanent ban. Scientific choices remain
  open unless explicitly approved: do not decide splits,
  systems, metrics, labels, candidates, budgets or protocols. Historical claims,
  code comments and “ratified” labels are not evidence of explicit human approval.
- Distinguish inspected facts, documentation claims, recommendations and Shaked's
  explicit decisions. Passing a fixture or environment check is not model performance.
- Before writing, inspect applicable instructions, paths, remotes, branches, base SHA,
  staged/unstaged changes and untracked files. Preserve others' work. No automatic
  reset, clean, stash, overwrite or deletion to obtain a clean tree.
- Work on a named branch, with small coherent commits and simple English titles.
  Do not push, merge, rewrite history or change authorship unless explicitly requested.
  Keep structural changes separate from behavioral fixes; inspect every final diff.
- The old repository, temporary audit copy and synced `sources/` are read-only.
  Selective copies from the old repository must come from `project/`, with repository,
  commit, source path and SHA-256 provenance. Do not merge the old history.
- Use simple Python, focused functions, explicit inputs and paths, and clear errors.
  Reuse existing shared logic. Imports must not download, train, call APIs or write.
  Avoid frameworks, personal paths, secrets and hidden notebook state.
- Install only in a new isolated local environment from `requirements.txt`.
  Keep weights, caches, environments, secrets and generated outputs out of Git.
  Use the safe checks documented in README for structural work. Real-data training,
  evaluation, mining, Ollama and paid API use require explicit task authorization.
  `pipeline/run_pipeline.sh` automatically evaluates test if present; `--skip-llm`
  does not disable test. Do not use it or `pipeline.run_all` as a structural smoke test.
- Structural extraction preserves data, labels, candidates, splits, prompts, scoring,
  results and model behavior. Compare to the identified baseline, not only the new
  implementation. Report existing defects instead of fixing them incidentally.
  In the historical training path, keep seeding after model initialization and strict
  development-loss improvement for checkpoint selection unless separately authorized.
- For an authorized notebook restructuring, preserve behavior; put imports
  and environment setup in the first code cell, then short explanations and function
  calls. Shared logic belongs in source modules. Experimental fixes require separate scope.
- Write concise formal English technical documentation. README serves readers; this
  file serves agents. Explain NLP terms for readers with basic ML knowledge. Record AI
  assistance truthfully when preparing submission material; do not claim human work.
- Establish file ownership before delegation. No simultaneous edits to the same file;
  review isolated changes before combining them. Only the coordinator updates the plan.
- Before handoff, check the diff against the base, whitespace, documentation links,
  provenance and absence of secrets/large outputs. Recheck untouched repository states.
  Report changes, tests, skips/failures and remaining scope in the conversation; stop
  at the package boundary for independent review.
