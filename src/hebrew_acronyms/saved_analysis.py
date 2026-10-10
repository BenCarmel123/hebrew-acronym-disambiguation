"""Saved test notebook helpers: evidence loading, exports and reader-facing tables.

The source identities and calculations come from the notebook at 5c6980e.
Presentation formatting never changes the rows supplied to numerical exports.
"""
from collections import Counter
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
from hebrew_acronyms.saved_human_review import write_csv

MODEL_LABELS = {"qwen": "Qwen2.5 7B", "qwen14": "Qwen2.5 14B",
                "dictalm": "DictaLM", "openai": "GPT-4.1 mini",
                "anthropic": "Claude Haiku 5.5", "xai": "Grok 4.7",
                "gemini": "Gemini 3.8 Flash", "dictabert": "Trained DictaBERT"}

def load_saved_analysis(root):
    ROOT = Path(root)
    SAVED_ARTIFACTS = ROOT / "saved-results"
    from hebrew_acronyms.saved_test_comparison import compare_saved_tests
    import json

    import hashlib
    inventory = json.loads((SAVED_ARTIFACTS / "files.json").read_text())
    for relative, evidence in inventory["files"].items():
        saved_file = SAVED_ARTIFACTS / relative
        if hashlib.sha256(saved_file.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError(f"Bundled evidence differs: {relative}")

    TEST_ARTIFACTS = SAVED_ARTIFACTS / "colab-runs"
    TEST_SESSIONS = [
        (TEST_ARTIFACTS / "evaluation-20261009-a319f9d/extracted",
         "8aa951a4ac96c126335f40ddc869cc7b6b6010cc94d7be1cc30925f09be9aac0"),
        (TEST_ARTIFACTS / "evaluation-gemini-retry-20261009-a319f9d/extracted",
         "837973fb8065fa7c6467d9370fb170ad58df757d3dadd33ae105910ecd245deb"),
        (TEST_ARTIFACTS / "evaluation-xai-qwen14-20261009-64f47b7/extracted",
         "f373731d81617852b35578fb78ab9d222598723f68e4907e9c5127392ebb5b64"),
        (TEST_ARTIFACTS / "evaluation-xai-qwen14-20261009-e9c8ef7/extracted",
         "2b185ec5d6baf1975629d82ec4608da353c376aaadebfbe7a483e80f5e6ee2f3"),
    ]
    TEST_SESSIONS.append((TEST_ARTIFACTS / "evaluation-gemini-381-20261010-79b767a/extracted",
                          "32f3aa9e0155ad993fa87df89bcce80d2f40276685968834c897db37c4e442b6"))
    TEST_RUN_IDS = {
        "gemini": "6658d323657f4df3aced950bfe6abb07",
        "xai": "885c4382ca904f0baca692ae9cc37ac9",
        "qwen": "cd1331b460364d4ebc341194c07dc1fe",
        "qwen14": "f23e084aace743e8a61a1a71dfbb33d4",
        "openai": "358e1fe65fa048c191eda08c078fd4cd",
        "anthropic": "3248737a597b435181227711caea89d0",
    }
    test_comparison = compare_saved_tests(TEST_SESSIONS, TEST_RUN_IDS, diagnostic_sources=[
        (TEST_ARTIFACTS / "evaluation-xai-qwen14-20261009-64f47b7/diagnostic/extracted",
         "e1436f143f73d846412d48f537cb8592afe1332de2879feba7f1572aae931d28",
         "6b3e9430e2b9f9b5f0fb298b87447b48f4eefa8fb7c659a1b7981602297deb77"),
    ], local_gemini_diagnostics=[
        (SAVED_ARTIFACTS / "gemini-diagnostic-20261010-after-billing",
         "208e223cf01dbbfab86d7bd94f85cf4079144c1d2747f521863822064f061d2a",
         "e9de33e09adc215fa5443c19a0b28a7746ce2098f6002e60e1e281257616f18c"),
    ])

    from hebrew_acronyms.encoder_test_results import load_encoder_test
    encoder_test = load_encoder_test(
        SAVED_ARTIFACTS / "study-runs/dictabert-test-20261009-f49c3b5",
        ROOT / "data/splits/test_items.csv",
        expected_run_id="dictabert-test-20261009-f49c3b5",
        expected_identity="2d3a20743715252b1ed743458f8efe728743ec8502329cca517f5e746cd3a126",
        expected_predictions_sha256="51050d90e9067c50077f14cea660c3a4ceee761044f514665ad1066cd80187b8",
    )
    combined_test_metrics = test_comparison["metrics"] + [encoder_test["metric"]]
    # Verify post-collection exclusion against the original immutable row snapshot.
    import csv
    from hebrew_acronyms.test_cohort import DOCUMENT_OVERLAP_IDS
    original_manifest = TEST_SESSIONS[0][0] / "qwen/full-test/manifest.json"
    original_rows = json.loads(original_manifest.read_text())["identity"]["rows"]
    with (ROOT / "data/splits/test_items.csv").open(encoding="utf-8-sig") as handle:
        scored_rows = list(csv.DictReader(handle))
    assert len(original_rows) == 395 and len(scored_rows) == 381
    assert [r for r in original_rows if r["item_id"] not in DOCUMENT_OVERLAP_IDS] == scored_rows
    removed_rows = [r for r in original_rows if r["item_id"] in DOCUMENT_OVERLAP_IDS]
    assert {r["item_id"] for r in removed_rows} == DOCUMENT_OVERLAP_IDS
    title_sets = {}
    for split in ("train", "dev"):
        with (ROOT / f"data/study_v1/encoder_inputs/{split}.csv").open() as handle:
            title_sets[split] = {r["page_title"] for r in csv.DictReader(handle)} - {""}
    assert {r["page_title"] for r in removed_rows} <= title_sets["train"]
    assert all(not ({r["page_title"] for r in scored_rows} & titles) for titles in title_sets.values())
    return dict(comparison=test_comparison, encoder=encoder_test, metrics=combined_test_metrics,
                original_manifest=original_manifest, run_ids=TEST_RUN_IDS, evidence_count=len(inventory["files"]))

def automatic_score_figure(test_comparison, combined_test_metrics, TEST_RUN_IDS, SUBMISSION_EXPORTS):
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    system_order = [name for name in ("qwen", "qwen14", "dictalm", "openai", "anthropic", "xai", "gemini") if name in TEST_RUN_IDS]
    system_order += ["dictabert"]
    # Exact model identities are linked in the notebook's technical evidence section.
    model_labels = {"qwen": "Qwen2.5 7B", "qwen14": "Qwen2.5 14B", "dictalm": "DictaLM",
                    "openai": "GPT-4.1 mini", "anthropic": "Claude Haiku 5.5", "xai": "Grok 4.7",
                    "gemini": "Gemini 3.8 Flash", "dictabert": "Trained DictaBERT"}
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "text.color": "#222222", "axes.labelcolor": "#222222",
                         "xtick.color": "#444444", "ytick.color": "#222222",
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.23, right=0.96, top=0.87, bottom=0.23, wspace=0.16)
        for ax, task, title in zip(axes, ("generate", "select"),
                                   ("(a) Generation: substring match", "(b) Selection: exact candidate")):
            groups = {m["system"]: m for m in combined_test_metrics if m["task"] == task}
            positions = [i for i, name in enumerate(system_order) if name in groups]
            values = [groups[system_order[i]]["accuracy"] for i in positions]
            bars = ax.barh(positions, values, color="#4D687C", edgecolor="#263746",
                           linewidth=0.4, height=0.58)
            for bar, value in zip(bars, values):
                inside = value >= 0.80
                ax.annotate(f"{value:.1%}", (value, bar.get_y() + bar.get_height() / 2),
                            xytext=(-4 if inside else 3, 0), textcoords="offset points",
                            ha="right" if inside else "left", va="center", fontsize=8,
                            color="white" if inside else "#222222")
            for i, name in enumerate(system_order):
                if name not in groups:
                    ax.text(0.03, i, "Not applicable" if name == "dictabert" else "Not measured",
                            va="center", fontsize=8, fontstyle="italic", color="#666666")
            ax.set_xlim(0, 1)
            ax.set_ylim(len(system_order) - 0.5, -0.5)
            ax.set_yticks(range(len(system_order)), [model_labels[name] for name in system_order])
            ax.set_title(title, loc="left", pad=10)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            ax.tick_params(axis="x", labelsize=8, length=3, width=0.6)
            ax.tick_params(axis="y", length=0, pad=7)
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
            ax.set_axisbelow(True)
        fig.supxlabel("Automatic match rate (all 381 items)", fontsize=9, y=0.10)
        fig.text(0.23, 0.025,
                 f"All {test_comparison['n_scored_items']} items per arm; technical failures included. Strict labels for LLM selection.",
                 fontsize=7.5, color="#444444")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(SUBMISSION_EXPORTS / f"automatic_scores.{extension}", dpi=200, bbox_inches="tight")
        plt.show()

def export_dictalm(dictalm_audit, test_comparison, SUBMISSION_EXPORTS):
    ben = dictalm_audit["ben_annotation"]
    dictalm_accounting = dict(
        local_api_increment_ils=dictalm_audit["local_increment_ils"],
        historical_local_prior_ils=dictalm_audit["local_prior_ils"],
        cumulative_accounted_ils=test_comparison["cost"]["total_accounted_ils"])
    # Each export replaces its derived table; no append or transfer of judgments.
    write_csv(SUBMISSION_EXPORTS / "dictalm_diagnostic_metrics.csv", dictalm_audit["metrics"])
    write_csv(SUBMISSION_EXPORTS / "dictalm_reported_metrics.csv", dictalm_audit["reported_metrics"])
    write_csv(SUBMISSION_EXPORTS / "dictalm_responses_381.csv", dictalm_audit["records"])
    write_csv(SUBMISSION_EXPORTS / "dictalm_ben_judgments.csv", ben["judgments"])
    write_csv(SUBMISSION_EXPORTS / "dictalm_ben_cross_tab.csv", [ben["cross_tab"]])
    write_csv(SUBMISSION_EXPORTS / "dictalm_accounting.csv", [dictalm_accounting])
    audit_metadata = {k: v for k, v in dictalm_audit.items() if k not in {"records", "ben_annotation"}}
    audit_metadata["ben_annotation"] = {k: v for k, v in ben.items() if k != "judgments"}
    (SUBMISSION_EXPORTS / "dictalm_audit.json").write_text(
        json.dumps(audit_metadata, ensure_ascii=False, indent=2) + "\n")

def export_comparison(ROOT, SUBMISSION_EXPORTS, test_comparison, combined_test_metrics, metric_coverage, baseline_rows):
    for name, rows in {
        "automatic_metrics": combined_test_metrics,
        "metric_coverage": metric_coverage,
        "baselines": baseline_rows,
        "technical_failures": test_comparison["failures"],
        "models_and_settings": [r | {"manifest_path": str(Path(r["manifest_path"]).relative_to(ROOT))}
                                for r in test_comparison["models"]],
        "pilot_results": test_comparison["pilots"],
        "cost_sessions": test_comparison["cost_sessions"],
        "cost_total": [test_comparison["cost"]],
    }.items():
        write_csv(SUBMISSION_EXPORTS / f"{name}.csv", rows)


def reading_table(rows, columns):
    """Escaped, narrow HTML that does not rely on collapsible details."""
    head = "".join(f'<th scope="col">{escape(label)}</th>' for label in columns.values())
    body = "".join("<tr>" + "".join(
        f'<td dir="auto" style="padding:6px 12px;text-align:start;white-space:normal;'
        f'word-break:normal;overflow-wrap:normal">{escape(str(row.get(key, "")))}</td>'
        for key in columns) + "</tr>" for row in rows)
    return ('<table style="width:auto;max-width:100%;table-layout:auto;font-size:14px;'
            'text-align:left"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>')


def comparison_table(metrics, similarity):
    """Keep generation and selection in explicitly distinct metric columns."""
    lookup = {(m["system"], m["task"]): m for m in metrics}
    def score(m):
        return f'{m["n_correct"]}/{m["n_items"]} ({100 * m["accuracy"]:.1f}%)'
    rows = [dict(model="DictaBERT similarity", generation="Not applicable",
                 selection=score(similarity["metric"]))]
    for system in ("dictabert", "qwen", "qwen14", "openai", "anthropic", "xai", "gemini"):
        rows.append(dict(model=MODEL_LABELS[system],
                         generation=score(lookup[(system, "generate")]) if (system, "generate") in lookup else "Not applicable",
                         selection=score(lookup[(system, "select")])))
    return reading_table(rows, {"model": "System", "generation": "Generation: substring match",
                                "selection": "Selection: reference match"})


def dictalm_table(metrics):
    names = {"automatic_generation": "Generation: automatic substring match",
             "ben_generation": "Generation: author annotation (B.C.)",
             "strict_selection": "Selection: strict label + reference match",
             "diagnostic_leading_label_selection": "Selection: leading-label diagnostic"}
    rows = [dict(measurement=names[m["metric"]], count=f'{m["n_correct"]}/{m["n_items"]}',
                 percent=f'{100 * m["accuracy"]:.1f}%') for m in metrics]
    return reading_table(rows, {"measurement": "Separate measurement", "count": "Accepted / all items", "percent": "Rate"})


def failure_summary(failures):
    counts = Counter((r["system"], r["task"], r["status"]) for r in failures)
    rows = [dict(model=MODEL_LABELS[system], task={"generate": "Generation", "select": "Selection"}[task],
                 status=status.replace("_", " "), count=count)
            for (system, task, status), count in sorted(counts.items())]
    return reading_table(rows, {"model": "System", "task": "Task", "status": "Stored status", "count": "Occurrences"})


def failure_examples(failures, limit=2, preview_length=240):
    """Show literal prefixes, visibly marked as excerpts; never normalize responses."""
    examples = []
    for row in [r for r in failures if r["response"]][:limit]:
        raw = row["response"]
        label = f'{MODEL_LABELS[row["system"]]} · {row["task"]} · {row["item_id"]}'
        note = f'First {preview_length} characters of {len(raw)}; full text is in the linked failure records.' if len(raw) > preview_length else 'Complete saved response.'
        examples.append(f'<p><strong>{escape(label)}</strong> — {note}</p>'
                        '<blockquote dir="auto" style="white-space:pre-wrap;word-break:normal;'
                        'overflow-wrap:break-word;max-width:760px">' + escape(raw[:preview_length]) + '</blockquote>')
    return "".join(examples)
