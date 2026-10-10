"""Draw the paper figures from the identified 381-item submission exports.

Run from the repository root: python paper/make_result_figures.py
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


def generate_vs_select(scores, path):
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    width = 0.32
    for index, (key, _) in enumerate(LLMS):
        generate, select = scores[(key, "generate")], scores[(key, "select")]
        ax.barh(index - width / 2, generate, width, color=GENERATE,
                label="Generation: reference substring" if index == 0 else None)
        ax.barh(index + width / 2, select, width, color=SELECT_API,
                label="Selection: strict candidate label" if index == 0 else None)
        ax.text(generate + 0.015, index - width / 2, f"{generate:.1%}", va="center", fontsize=7.5)
        ax.text(select + 0.015, index + width / 2, f"{select:.1%}", va="center", fontsize=7.5)
    ax.axhline(1.5, color="#dddddd", linewidth=0.6)
    ax.set_yticks(range(len(LLMS)), [name for _, name in LLMS])
    ax.invert_yaxis()
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=8)
    fig.text(0.5, 0.035, "Automatic score (task-specific rule)", ha="center", fontsize=7.5)
    fig.suptitle(f"Generation and selection (n = {N_ITEMS})", fontsize=9, y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper left",
               bbox_to_anchor=(0.02, 0.935))
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.17, top=0.78)
    save_figure(fig, path)


def main():
    scores, baselines = load_scores()
    arm_comparison(scores, baselines, FIGURES / "arm_comparison.pdf")
    generate_vs_select(scores, FIGURES / "generate_vs_select.pdf")


if __name__ == "__main__":
    main()
