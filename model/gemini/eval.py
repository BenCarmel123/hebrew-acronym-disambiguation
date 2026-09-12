"""Open-generation / candidate-select evaluation against Gemini — the SOTA/strong LLM
arm, same prompts and scoring as model/qwen/eval.py, over the hosted API instead of
a local model.

Requires GEMINI_API_KEY in a local .env file (never committed — see .gitignore). Get a
free key at https://aistudio.google.com/apikey.

    python -m model.gemini.eval --mode generate
    python -m model.gemini.eval --mode select
"""
from __future__ import annotations

import argparse
import os
import time

import requests
from dotenv import load_dotenv

from model.common.eval import evaluate, write_details_csv
from model.common.pairs import load_rows

load_dotenv()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: Billing is enabled, so the free-tier RPM caps that forced a delay/lite-model
#: workaround no longer apply. Kept at 0 (not removed) so a retry-on-429 still exists
#: for transient issues, without slowing down a normal run.
REQUEST_DELAY_SECONDS = 0.0
MAX_RETRIES = 5


def gemini_generate(prompt: str, model: str = "gemini-3.6-flash", thinking: bool = False) -> str:
    """`thinking=False` (the default) sends the minimum thinkingBudget this model
    accepts. thinkingBudget=0 is rejected with a 400 (INVALID_ARGUMENT) on
    gemini-3.6-flash — unlike some other Gemini models it cannot fully disable
    thinking — so 1 is the practical floor. This is a short classification task, not
    one that benefits from extended reasoning, and thinking tokens cost latency and
    money for no expected accuracy gain. Set thinking=True to compare against it.
    """
    api_key = os.environ["GEMINI_API_KEY"]
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if not thinking:
        body["generationConfig"] = {"thinkingConfig": {"thinkingBudget": 1}}

    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY_SECONDS)
        response = requests.post(
            GEMINI_URL.format(model=model), params={"key": api_key}, json=body,
        )
        if response.status_code == 429 and attempt < MAX_RETRIES - 1:
            # Exponential backoff: 15s, 30s, 60s, 120s before giving up.
            time.sleep(15 * (2 ** attempt))
            continue
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--mode", default="generate", choices=["generate", "select"])
    ap.add_argument("--thinking", action="store_true",
                    help="enable extended thinking (off by default — see gemini_generate)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"data/mined/llm_gemini_{a.mode}_details.csv"

    res = evaluate(load_rows(a.items),
                   generate_fn=lambda p: gemini_generate(p, model=a.model, thinking=a.thinking),
                   mode=a.mode)
    print(f"{a.items}  (model: {a.model}, mode: {a.mode}, thinking: {a.thinking})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")
    print(f"  invalid rate      {res['invalid_rate']:.3f}   response matched no candidate")

    write_details_csv(out, res["details"])
    print(f"\n  per-item details written to {out}")


if __name__ == "__main__":
    main()
