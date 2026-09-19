#!/usr/bin/env bash
# Validate a train/dev(/test) set, then run every eval arm against dev — and, if
# --test is given (or data/splits/test_items.csv exists), against test too — and
# print/save the combined results table(s). Training (dictabertX) is NOT run here —
# see README.md for the prepared local training environment. Pass an existing
# checkpoint with --checkpoint to include that arm; otherwise it's skipped and the
# table shows "—".
#
# Usage:
#   src/hebrew_acronyms/pipelines/run_pipeline.sh
#   src/hebrew_acronyms/pipelines/run_pipeline.sh --train data/splits/train_items.csv --dev data/splits/dev_items.csv
#   src/hebrew_acronyms/pipelines/run_pipeline.sh --test data/splits/test_items.csv
#   src/hebrew_acronyms/pipelines/run_pipeline.sh --checkpoint weights/dictabert-crossenc-<ts>.pt
#   src/hebrew_acronyms/pipelines/run_pipeline.sh --skip-llm   # fast: skips qwen/gemini (local model + API calls)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

TRAIN="data/splits/train_items.csv"
DEV="data/splits/dev_items.csv"
TEST="data/splits/test_items.csv"
CANDIDATES="data/mined/candidate_table.csv"
CHECKPOINT=""
OUT="results/all_arms_summary.md"
SKIP_LLM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN="$2"; shift 2 ;;
    --dev) DEV="$2"; shift 2 ;;
    --test) TEST="$2"; shift 2 ;;
    --candidates) CANDIDATES="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --skip-llm) SKIP_LLM="--skip-llm"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Skip test validation/evaluation only when the selected test file is absent.
if [[ ! -f "$TEST" ]]; then
  TEST=""
fi

echo "== 1. Validating $TRAIN / $DEV${TEST:+ / $TEST} =="
.venv/bin/python -m hebrew_acronyms.pipelines.validate_data --train "$TRAIN" --dev "$DEV" ${TEST:+--test "$TEST"}
echo

if [[ -z "$CHECKPOINT" ]]; then
  echo "No --checkpoint given: dictabertX (fine-tuned) will be skipped in the table."
  echo "To include it, supply a checkpoint from an explicitly authorized training run;"
  echo "see README.md, then use --checkpoint <path-to-.pt> when evaluation is authorized."
  echo
fi

echo "== 2. Running all eval arms against $DEV =="
.venv/bin/python -m hebrew_acronyms.pipelines.run_all \
  --items "$DEV" \
  --candidates "$CANDIDATES" \
  --out "$OUT" \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  $SKIP_LLM

if [[ -n "$TEST" ]]; then
  TEST_OUT="${OUT%.md}_test.md"
  echo
  echo "== 3. Running all eval arms against $TEST =="
  .venv/bin/python -m hebrew_acronyms.pipelines.run_all \
    --items "$TEST" \
    --candidates "$CANDIDATES" \
    --out "$TEST_OUT" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    $SKIP_LLM
fi
