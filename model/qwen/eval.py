"""Open-generation / candidate-select evaluation against a local Qwen model via Ollama.

Requires Ollama running locally (`ollama serve` or `brew services start ollama`) with
the model pulled (`ollama pull qwen2.5:7b`).

    python -m model.qwen.eval --mode generate
    python -m model.qwen.eval --mode select
"""
from __future__ import annotations

import argparse

import requests

from model.common.eval import evaluate, write_details_csv
from model.common.pairs import load_rows

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--mode", default="generate", choices=["generate", "select"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/qwen/{a.mode}_details.csv"

    res = evaluate(load_rows(a.items),
                   generate_fn=lambda p: ollama_generate(p, model=a.model),
                   mode=a.mode)
    print(f"{a.items}  (model: {a.model}, mode: {a.mode})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")
    print(f"  invalid rate      {res['invalid_rate']:.3f}   response matched no candidate")

    write_details_csv(out, res["details"])
    print(f"\n  per-item details written to {out}")


if __name__ == "__main__":
    main()
