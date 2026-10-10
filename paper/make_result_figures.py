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
from matplotlib.lines import Line2D

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
    """Two descriptive panels; task-specific scores are not a common metric."""
    # Keep the LLM order identical in both panels and the main manuscript table.
    names = dict(LLMS)
    systems = ["qwen", "qwen14", "gemini", "openai", "anthropic", "xai"]
    selection = [("dictabert_similarity", "DictaBERT similarity"),
                 ("dictabert", "DictaBERT ranker")]
    selection += [(key, names[key]) for key in systems]
    blue, charcoal = "#28608b", "#333333"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.0, 2.65))
        # One model-label column and identical row positions free horizontal
        # space for both comparisons. Encoder rows have no points in panel (b).
        left = fig.add_axes((0.21, 0.16, 0.34, 0.65))
        right = fig.add_axes((0.63, 0.16, 0.34, 0.65), sharey=left)
        fig.text(0.21, 0.95, "(a) Candidate selection", weight="bold", fontsize=10)
        fig.text(0.63, 0.95, "(b) Generation vs selection", weight="bold", fontsize=10)
        fig.legend([
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=charcoal, markersize=5),
            Line2D([], [], marker="s", linestyle="none", color=blue, markersize=5)],
            ["Generation", "Selection"], ncol=2,
            loc="upper left", bbox_to_anchor=(0.615, 0.92), frameon=False,
            fontsize=9, handletextpad=0.4, columnspacing=0.9)

        def point(ax, value, y, marker, color, filled=True, label_left=False):
            ax.plot(value, y, marker=marker, linestyle="none", markersize=5,
                    markeredgecolor=color, markerfacecolor=color if filled else "white",
                    zorder=3)
            # Keep direct labels within the 0--100% plotting region.
            high = value >= 0.80 or label_left
            ax.text(value + (-0.035 if high else 0.035), y, f"{value:.1%}",
                    ha="right" if high else "left", va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
                    zorder=4)

        for index, (key, _) in enumerate(selection):
            encoder = key.startswith("dictabert")
            point(left, scores[(key, "select")], index,
                  "D" if encoder else "s", charcoal if encoder else blue)
        left.axvline(baselines["most_mined"], color=charcoal, linestyle="--",
                     linewidth=0.8, zorder=1)
        left.text(baselines["most_mined"], -0.78,
                  f"Most-mined {baselines['most_mined']:.1%}",
                  ha="center", va="bottom", fontsize=8)
        left.set_yticks(range(len(selection)), [label for _, label in selection])
        left.set_ylim(len(selection) - 0.5, -1.0)
        left.axhline(1.5, color="#dddddd", linewidth=0.6)
        left.axhline(3.5, color="#dddddd", linewidth=0.6)
        for index, key in enumerate(systems, start=2):
            generate = scores[(key, "generate")]
            # Put mid-range generation labels before their open circles, away
            # from selection labels; near zero, retain room inside the axis.
            point(right, generate, index, "o", charcoal, False, label_left=generate >= 0.20)
            point(right, scores[(key, "select")], index, "s", blue)
        right.tick_params(axis="y", labelleft=False)
        right.text(0.5, 0.5, "LLMs only", ha="center", va="center",
                   fontsize=9, color="#555555", style="italic")
        right.axhline(1.5, color="#dddddd", linewidth=0.6)
        right.axhline(3.5, color="#dddddd", linewidth=0.6)
        for ax in (left, right):
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(axis="y", length=0, pad=5)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", color="#eeeeee", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel("Automatic score (%)", fontsize=8.5)
        save_figure(fig, path)


def main():
    scores, baselines = load_scores()
    generate_vs_select(scores, baselines, FIGURES / "generate_vs_select.pdf")


if __name__ == "__main__":
    main()
