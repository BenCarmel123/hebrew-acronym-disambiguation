"""Draw the paper figures from the identified 381-item submission exports.

Run from the repository root: python paper/make_result_figures.py
The default regenerates the two-panel figure used by the manuscript.
arm_comparison remains available as the historical selection-only helper.
Reads only saved CSV files; it calls no model and changes no result. DictaLM's
four measurements belong to the separate supplementary manuscript table.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "results" / "submission-20261010"
FIGURES = ROOT / "paper" / "figures"
N_ITEMS = 381

LLMS = [("qwen", "Qwen 2.5 7B"), ("qwen14", "Qwen 2.5 14B"),
        ("openai", "GPT-4.1 mini"), ("anthropic", "Claude Haiku 5.5"),
        ("xai", "Grok 4.7"), ("gemini", "Gemini 3.8 Flash")]
LOCAL = {"qwen", "qwen14"}
GENERATE, SELECT_LOCAL, SELECT_API = "#8c8c8c", "#5a9bd5", "#2b6cb0"
TRAINED, SIMILARITY, BASELINE = "#2e8b57", "#a0aec0", "#cccccc"
EXPECTED_RULES = {"generate": "historical_quote_normalized_gold_substring",
                  "select": "strict_candidate_label_v1"}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def saved_accuracy(row, rule):
    """Check a saved metric's identity and denominator without rescoring outputs."""
    if int(row["n_items"]) != N_ITEMS or row["score"] != rule:
        raise ValueError(f"unexpected denominator or rule: {row}")
    accuracy = float(row["accuracy"])
    if abs(accuracy - int(row["n_correct"]) / N_ITEMS) > 1e-12:
        raise ValueError(f"saved count and accuracy disagree: {row}")
    return accuracy


def load_scores():
    scores = {}
    llm_keys = {key for key, _ in LLMS}
    for row in read_csv(SUBMISSION / "automatic_metrics.csv"):
        system, task = row["system"], row["task"]
        if system not in llm_keys | {"dictabert"}:
            continue
        rule = "exact_candidate_match" if system == "dictabert" else EXPECTED_RULES[task]
        key = (system, task)
        if key in scores:
            raise ValueError(f"duplicate saved metric: {key}")
        scores[key] = saved_accuracy(row, rule)
    expected = {(key, task) for key in llm_keys for task in EXPECTED_RULES}
    if set(scores) != expected | {("dictabert", "select")}:
        raise ValueError("missing or unexpected automatic metric")
    similarity_rows = read_csv(SUBMISSION / "similarity_metric.csv")
    if len(similarity_rows) != 1 or similarity_rows[0]["system"] != "dictabert_similarity":
        raise ValueError("expected the identified pretrained similarity metric")
    scores[("dictabert_similarity", "select")] = saved_accuracy(
        similarity_rows[0], "exact_candidate_match")
    baselines = {}
    for row in read_csv(SUBMISSION / "baselines.csv"):
        if int(row["n_items"]) != N_ITEMS:
            raise ValueError("baseline denominator differs from the test cohort")
        baselines[row["baseline"]] = float(row["accuracy"])
    return scores, baselines


def style_axis(ax):
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def save_figure(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def arm_comparison(scores, baselines, path):
    # Only candidate-selection outcomes share this panel; generation has a
    # different scoring rule and is shown separately in generate_vs_select.
    bars = [("Random", baselines["random"], BASELINE),
            ("Most frequent", baselines["most_frequent"], BASELINE),
            ("Most mined", baselines["most_mined"], BASELINE),
            ("Pretrained similarity", scores[("dictabert_similarity", "select")], SIMILARITY),
            ("Task-trained encoder", scores[("dictabert", "select")], TRAINED)]
    bars.extend((name, scores[(key, "select")], SELECT_LOCAL if key in LOCAL else SELECT_API)
                for key, name in LLMS)
    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    for position, (name, value, color) in enumerate(bars):
        ax.barh(position, value, color=color, height=0.65)
        ax.text(value + 0.015, position, f"{value:.1%}", va="center", fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1}, zorder=3)
    ax.set_yticks(range(len(bars)), [bar[0] for bar in bars])
    ax.invert_yaxis()
    for boundary in (2.5, 4.5, 6.5):
        ax.axhline(boundary, color="#dddddd", linewidth=0.6)
    ax.axvline(baselines["oracle"], color="#666666", linestyle=":", linewidth=0.8)
    ax.text(baselines["oracle"], -0.95, "Oracle", ha="center", fontsize=6.5)
    style_axis(ax)
    ax.set_xlabel("Candidate-selection accuracy", fontsize=7.5)
    ax.set_title(f"Candidate selection (n = {N_ITEMS})", fontsize=8, pad=13)
    fig.tight_layout(pad=0.5)
    save_figure(fig, path)


def generate_vs_select(scores, baselines, path):
    """Stack the two comparisons, showing only applicable systems in each."""
    systems = ["qwen", "qwen14", "gemini", "openai", "anthropic", "xai"]
    llm_labels = ["Qwen 2.5\n7B", "Qwen 2.5\n14B", "Gemini\n3.8 Flash",
                  "GPT-4.1\nmini", "Claude\nHaiku 5.5", "Grok\n4.7"]
    selection = ["dictabert_similarity", "dictabert", *systems]
    selection_labels = ["DictaBERT\nsimilarity", "DictaBERT\nranker", *llm_labels]
    blue, charcoal = "#28608b", "#333333"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.0, 2.95))
        top = fig.add_axes((0.075, 0.65, 0.91, 0.235))
        bottom = fig.add_axes((0.075, 0.145, 0.91, 0.235))
        fig.text(0.075, 0.96, "(a) Candidate selection", weight="bold", fontsize=10)
        fig.text(0.985, 0.96, f"Dashed: most-mined ({baselines['most_mined']:.1%})",
                 ha="right", fontsize=10)
        fig.text(0.075, 0.465, "(b) LLM generation and selection", weight="bold", fontsize=10)

        def bar(ax, position, value, width, generation=False, encoder=False, label=None):
            ax.bar(position, value, width=width,
                   color="white" if generation else charcoal if encoder else blue,
                   edgecolor=charcoal if generation else "none", linewidth=0.8,
                   hatch="///" if generation else None, label=label, zorder=3)
            ax.annotate(f"{100 * value:.1f}", (position, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10)

        for index, key in enumerate(selection):
            bar(top, index, scores[(key, "select")], 0.55, encoder=key.startswith("dictabert"))
        top.axhline(baselines["most_mined"], color=charcoal, linestyle="--",
                    linewidth=0.8, zorder=4)
        top.set_xticks(range(len(selection)), selection_labels)
        top.set_xlim(-0.55, len(selection) - 0.45)
        for index, key in enumerate(systems):
            bar(bottom, index - 0.21, scores[(key, "generate")], 0.34,
                generation=True, label="Generation" if index == 0 else None)
            bar(bottom, index + 0.21, scores[(key, "select")], 0.34,
                label="Selection" if index == 0 else None)
        bottom.set_xticks(range(len(systems)), llm_labels)
        bottom.set_xlim(-0.55, len(systems) - 0.45)
        handles, labels = bottom.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.99, 0.475),
                   frameon=False, ncol=2, fontsize=10, handlelength=1.4,
                   handletextpad=0.5, columnspacing=1.0)
        for ax in (top, bottom):
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(axis="x", length=0, pad=5, labelsize=10)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
        save_figure(fig, path)


def main():
    scores, baselines = load_scores()
    generate_vs_select(scores, baselines, FIGURES / "generate_vs_select.pdf")


if __name__ == "__main__":
    main()
