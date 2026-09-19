"""Evaluate open generation or candidate selection through the Gemini API.

Uses the shared LLM prompts and scoring. Requires GEMINI_API_KEY in the environment
or a local ignored .env file. Account access and quotas must be checked before an
authorized service run.

    python -m hebrew_acronyms.models.gemini.eval --mode generate
    python -m hebrew_acronyms.models.gemini.eval --mode select
"""
from __future__ import annotations

import argparse
import os
import time

import requests
from dotenv import load_dotenv

from hebrew_acronyms.models.common.eval import evaluate, write_details_csv
from hebrew_acronyms.models.common.pairs import load_rows

load_dotenv()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: Delay before each request, separate from the retry backoff for HTTP 429.
REQUEST_DELAY_SECONDS = 0.0
MAX_RETRIES = 5


def gemini_generate(prompt: str, model: str = "gemini-3.6-flash", thinking: bool = False) -> str:
    """Send a prompt using the retained request configuration.

    thinking=False sends thinkingBudget=1; thinking=True omits that configuration,
    leaving the budget to the model's default. These flags do not establish that
    thinking is disabled or enabled. Model availability and support for the request
    settings must be verified before an authorized run.
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
                    help="omit the explicit thinking budget and use the model's default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model_tag = a.model.replace("gemini-", "").replace("-latest", "").replace(".", "")
    thinking_tag = "thinking" if a.thinking else "nothinking"
    out = a.out or f"results/gemini/{model_tag}_{thinking_tag}_{a.mode}_details.csv"

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
