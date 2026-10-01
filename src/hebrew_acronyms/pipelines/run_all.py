"""Run every method arm against the same dev set and print one combined results table.

Each arm keeps its own eval.py (with its own CLI, flags, and printed detail) — this
script just calls each arm's evaluate() directly and collects the numbers, so the
report has one table instead of seven separate command outputs to copy by hand.

    python -m hebrew_acronyms.pipelines.run_all

Checkpoint runs are rejected before input reads or model/service calls while
benchmark metric aggregation for E1 item records remains pending.
"""
from __future__ import annotations

import argparse

import torch

from hebrew_acronyms.models.baselines import evaluate as evaluate_baselines
from hebrew_acronyms.models.baselines import load_signals
from hebrew_acronyms.models.common.eval import evaluate as evaluate_llm
from hebrew_acronyms.models.common.pairs import load_rows
from hebrew_acronyms.models.dictabert_similarity.model import MODEL_ID as DICTABERT_MODEL_ID, build_model
from hebrew_acronyms.models.dictabert_similarity.eval import evaluate as evaluate_dictabert_zeroshot


def run(items_path: str, candidates_path: str, checkpoint: str | None,
        skip_llm: bool) -> list[dict]:
    if checkpoint is not None:
        raise ValueError(
            "run_all with a checkpoint is unavailable: benchmark metric aggregation "
            "for E1 item records is pending. Use the item-record path in "
            "hebrew_acronyms.models.dictabert_cross_encoder.eval with explicit "
            "authorized inputs; no input files, models or services have been accessed."
        )
    rows = load_rows(items_path)
    results = []

    ranks, mined = load_signals(candidates_path)
    b = evaluate_baselines(rows, ranks, mined)
    for arm, key in [("random", "random"), ("most_frequent", "most_frequent"),
                     ("most_mined", "most_mined"), ("oracle", "oracle")]:
        results.append({"arm": arm, "accuracy": b[key], "invalid_rate": None,
                        "n_items": b["n_items"]})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = build_model(DICTABERT_MODEL_ID)
    model = model.to(device)
    model.eval()
    d = evaluate_dictabert_zeroshot(rows, tok, model, device)
    results.append({"arm": "dictabert (untrained)", "accuracy": d["accuracy"],
                    "invalid_rate": None, "n_items": d["n_items"]})

    if checkpoint:
        from hebrew_acronyms.models.dictabert_cross_encoder.eval import evaluate as evaluate_dictabertx
        from hebrew_acronyms.models.dictabert_cross_encoder.model import load_finetuned
        dtok, dmodel, acr_open, acr_close = load_finetuned(checkpoint, device=device)
        x = evaluate_dictabertx(rows, dtok, dmodel, acr_open, acr_close, device)
        results.append({"arm": "dictabertX (fine-tuned)", "accuracy": x["accuracy"],
                        "invalid_rate": None, "n_items": x["n_items"]})
    else:
        results.append({"arm": "dictabertX (fine-tuned)", "accuracy": None,
                        "invalid_rate": None, "n_items": None})

    if not skip_llm:
        from hebrew_acronyms.models.qwen.eval import ollama_generate
        for mode in ("generate", "select"):
            q = evaluate_llm(rows, generate_fn=ollama_generate, mode=mode)
            results.append({"arm": f"qwen ({mode})", "accuracy": q["accuracy"],
                            "invalid_rate": q["invalid_rate"], "n_items": q["n_items"]})

        from hebrew_acronyms.models.gemini.eval import gemini_generate
        for mode in ("generate", "select"):
            g = evaluate_llm(rows, generate_fn=gemini_generate, mode=mode)
            results.append({"arm": f"gemini ({mode})", "accuracy": g["accuracy"],
                            "invalid_rate": g["invalid_rate"], "n_items": g["n_items"]})

    return results


def to_markdown(results: list[dict]) -> str:
    lines = ["| Arm | Accuracy | Invalid rate | Items |",
             "|---|---|---|---|"]
    for r in results:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "—"
        inv = f"{r['invalid_rate']:.3f}" if r["invalid_rate"] is not None else "—"
        n = r["n_items"] if r["n_items"] is not None else "—"
        lines.append(f"| {r['arm']} | {acc} | {inv} | {n} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--candidates", default="data/mined/candidate_table.csv")
    ap.add_argument("--checkpoint", default=None,
                    help="currently rejected before input reads; use the E1 item-record evaluator")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip qwen/gemini arms (slow — local-model/API calls)")
    ap.add_argument("--out", default="results/all_arms_summary.md")
    a = ap.parse_args()

    results = run(a.items, a.candidates, a.checkpoint, a.skip_llm)
    table = to_markdown(results)
    print(table)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
