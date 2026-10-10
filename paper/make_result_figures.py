"""Draw the two paper result figures from the saved 381-item test results.

Run from the repository root: python paper/make_result_figures.py
Reads only saved CSV files; it calls no model and changes no result.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "results" / "submission-20261010"
DICTALM = ROOT / "saved-results" / "local-runs" / "dictalm-20261010"
UNTRAINED_DICTABERT = ROOT / "results" / "dictabert" / "test_details.csv"
FIGURES = ROOT / "paper" / "figures"
N_ITEMS = 381

LLMS = [("qwen", "Qwen 2.5 7B"), ("qwen14", "Qwen 2.5 14B"), ("dictalm", "DictaLM 2.0"),
        ("openai", "GPT-4.1 mini"), ("anthropic", "Claude Haiku 5.5"), ("xai", "Grok 4.7"),
        ("gemini", "Gemini 3.8 Flash")]
OPEN = {"qwen", "qwen14", "dictalm"}
GENERATE, SELECT_OPEN, SELECT_PROPRIETARY = "#8c8c8c", "#5a9bd5", "#2b6cb0"
TRAINED, UNTRAINED = "#2e8b57", "#a0aec0"


def read_csv(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def share(rows, column):
    rows = [row for row in rows if row.get("scored_in_381", "True") == "True"]
    if len(rows) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} scored rows, found {len(rows)}")
    return sum(row[column] == "True" for row in rows) / N_ITEMS


def load_scores():
    scores = {}
    for row in read_csv(SUBMISSION / "automatic_metrics.csv"):
        if int(row["n_items"]) != N_ITEMS:
            raise ValueError(f"{row['system']} {row['task']} is not on {N_ITEMS} items")
        scores[(row["system"], row["task"])] = float(row["accuracy"])
    # DictaLM generation is judged by hand (yes = same as the reference); other generation scores are automatic.
    tags = read_csv(DICTALM / "dictalm_generate_human_tags.csv")
    if len(tags) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} hand tags, found {len(tags)}")
    scores[("dictalm", "generate")] = sum(row["same_as_gold"] == "yes" for row in tags) / N_ITEMS
    # DictaLM answers "B. <candidate>"; the leading letter is read (lenient parse, see the paper).
    scores[("dictalm", "select")] = share(read_csv(DICTALM / "dictalm_test_select_details.csv"), "lenient_correct")
    scores[("dictabert_untrained", "select")] = share(read_csv(UNTRAINED_DICTABERT), "correct")
    baselines = {row["baseline"]: float(row["accuracy"]) for row in read_csv(SUBMISSION / "baselines.csv")}
    return scores, baselines


def arm_comparison(scores, baselines, path):
    bars = [("DictaBERT (fine-tuned)", scores[("dictabert", "select")], TRAINED),
            ("DictaBERT (untrained)", scores[("dictabert_untrained", "select")], UNTRAINED)]
    for key, name in LLMS:
        bars.append((f"{name} (select)", scores[(key, "select")],
                     SELECT_OPEN if key in OPEN else SELECT_PROPRIETARY))
        bars.append((f"{name} (generate{', hand-judged' if key == 'dictalm' else ''})",
                     scores[(key, "generate")], GENERATE))
    bars.sort(key=lambda bar: bar[1])

    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    positions = range(len(bars))
    ax.barh(positions, [bar[1] for bar in bars], color=[bar[2] for bar in bars], height=0.65)
    ax.set_yticks(list(positions), [bar[0] for bar in bars], fontsize=8.5)
    for position, (_, value, _) in zip(positions, bars):
        ax.text(value + 0.01, position, f"{value:.1%}", va="center", fontsize=7.5)
    for name, label, style in [("random", "Random", "--"), ("most_frequent", "Most frequent", "--"),
                               ("most_mined", "Most mined", "--")]:
        ax.axvline(baselines[name], color="#4a5568", linestyle=style, linewidth=0.9)
        ax.text(baselines[name] + 0.008, -1.55, label, rotation=90, fontsize=7,
                color="#4a5568", va="bottom")
    ax.axvline(1.0, color="#c53030", linestyle="--", linewidth=1.2)
    ax.text(1.008, -1.55, "Oracle", rotation=90, fontsize=7, color="#c53030", va="bottom")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-1.6, len(bars) - 0.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Accuracy")
    ax.set_title(f"All systems on the held-out test set (n={N_ITEMS})", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_vs_select(scores, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.38
    for index, (key, _) in enumerate(LLMS):
        generate, select = scores[(key, "generate")], scores[(key, "select")]
        ax.bar(index - width / 2, generate, width, color=GENERATE,
               label="Generate (no candidates)" if index == 0 else None)
        ax.bar(index + width / 2, select, width, color=SELECT_PROPRIETARY,
               label="Select (candidates shown)" if index == 0 else None)
        ax.text(index - width / 2, generate + 0.015, f"{generate:.1%}", ha="center", fontsize=6.5)
        ax.text(index + width / 2, select + 0.015, f"{select:.1%}", ha="center", fontsize=6.5)
    ax.axvline(2.5, color="#a0aec0", linewidth=0.8, linestyle=":")
    ax.text(1.0, 1.06, "Open", ha="center", fontsize=8, color="#4a5568")
    ax.text(4.5, 1.06, "Proprietary", ha="center", fontsize=8, color="#4a5568")
    ax.set_xticks(range(len(LLMS)), [name for _, name in LLMS], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Free generation vs. candidate selection (held-out test, n={N_ITEMS})", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.97))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    scores, baselines = load_scores()
    arm_comparison(scores, baselines, FIGURES / "arm_comparison.pdf")
    generate_vs_select(scores, FIGURES / "generate_vs_select.pdf")


if __name__ == "__main__":
    main()
