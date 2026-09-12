"""Baselines for the acronym ranking task — the bar a trained model has to clear.

An accuracy number means nothing on its own. These say what you get for free:

  random          pick uniformly among the item's candidates. The floor.
  most_frequent   pick the candidate Wikipedia mentions most (candidate_table's
                  `rank`, which is 1 for the most-hit expansion of an acronym).
                  Sense distributions are skewed, so always guessing the dominant
                  sense is a strong cheap strategy, and a model that cannot beat
                  it has learned nothing about context. This is the classic
                  most-frequent-sense baseline from the WSD literature.
  most_mined      the same idea on a better signal. `rank` derives from `hits`,
                  which the dataset card calls a poor proxy: `ב״ש` reports
                  thousands of hits that are almost all `ב״שלום`, a prefix
                  collision rather than the acronym. `mined_items` counts the
                  sentences an expansion actually yielded, so it measures
                  attestation instead of string matches. Expect it to be the
                  harder of the two, and the fairer bar.
  oracle          pick correctly whenever gold is among the candidates. The
                  ceiling: anything below 1.0 here is a data defect, not a model
                  failure, since no ranker can select a candidate it was never
                  offered.

Random is averaged in closed form (1/n_candidates per item) rather than sampled,
so the number is exact and needs no seed.

    python -m model.baselines --items data/splits/dev_items.csv
"""
from __future__ import annotations

import argparse
import csv


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_signals(path: str) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """-> (ranks, mined). Both keyed (acronym, expansion).

    `ranks` is candidate_table's own rank, 1 = most Wikipedia hits.
    `mined` is how many sentences that expansion actually produced.
    """
    ranks, mined = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["acronym"], r["expansion"])
            try:
                ranks[key] = int(r["rank"])
            except (KeyError, ValueError):
                pass
            try:
                mined[key] = int(r["mined_items"])
            except (KeyError, ValueError):
                pass
    return ranks, mined


def evaluate(rows: list[dict], ranks: dict[tuple[str, str], int],
             mined: dict[tuple[str, str], int]) -> dict:
    n = 0
    random_sum = 0.0          # expected accuracy, summed per item
    most_frequent = 0
    most_mined = 0
    oracle = 0

    for r in rows:
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        if len(cands) < 2 or not gold:
            continue
        n += 1

        random_sum += 1.0 / len(cands)
        oracle += int(gold in cands)

        # Lowest rank number = most Wikipedia hits. An unranked candidate sorts
        # last; ties break on the item's own candidate order, which is stable.
        best = min(range(len(cands)),
                   key=lambda i: ranks.get((r["acronym"], cands[i]), 1 << 30))
        most_frequent += int(cands[best] == gold)

        # Most sentences actually mined. Negated so the largest count sorts
        # first under the same min(); an expansion absent from the table counts
        # as zero, which is what "never yielded a sentence" means.
        best_mined = min(range(len(cands)),
                         key=lambda i: -mined.get((r["acronym"], cands[i]), 0))
        most_mined += int(cands[best_mined] == gold)

    return {
        "n_items": n,
        "mean_candidates": round(sum(
            len([c for c in r["candidates"].split("|") if c.strip()])
            for r in rows) / max(len(rows), 1), 2),
        "random": round(random_sum / n, 4) if n else 0.0,
        "most_frequent": round(most_frequent / n, 4) if n else 0.0,
        "most_mined": round(most_mined / n, 4) if n else 0.0,
        "oracle": round(oracle / n, 4) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--candidates", default="data/mined/candidate_table.csv")
    a = ap.parse_args()

    ranks, mined = load_signals(a.candidates)
    res = evaluate(load_rows(a.items), ranks, mined)
    print(f"{a.items}\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  mean candidates   {res['mean_candidates']}")
    print()
    print(f"  random            {res['random']:.3f}   floor")
    print(f"  most frequent     {res['most_frequent']:.3f}   by Wikipedia hits")
    print(f"  most mined        {res['most_mined']:.3f}   by sentences yielded — the fairer bar")
    print(f"  oracle            {res['oracle']:.3f}   ceiling (gold in candidates)")


if __name__ == "__main__":
    main()
