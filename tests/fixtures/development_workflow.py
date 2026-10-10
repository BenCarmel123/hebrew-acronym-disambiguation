# Frozen development orchestration extracted from experimental_study.ipynb at
# 5c6980efee34b18e3fa63e53f7fd5713938571c0 (setup and Appendix A code).
# Test-only fixture; interfaces are guarded and mocked by run_experimental_study.
# %%
from pathlib import Path
from uuid import uuid4
from IPython.display import HTML, display
from hebrew_acronyms import experimental_study as study
from hebrew_acronyms.paper_export import export_paper
from hebrew_acronyms.saved_study_comparison import display_saved_comparison

ROOT = Path.cwd().resolve()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
RUN_DEVELOPMENT_APPENDIX = globals().get("RUN_DEVELOPMENT_APPENDIX", False)
settings = dict(
    mode="preview", run_kind="validation", run_id=str(uuid4()), root=ROOT,
    train_path=ROOT / "data/study_v1/encoder_inputs/train.csv",
    input_path=ROOT / "data/study_v1/encoder_inputs/dev.csv", expected_dev_items=62,
    validation_items=3, device="cpu", checkpoint=None, snapshot_path=None,
    checkpoint_format="manifest", checkpoint_sha256=None, checkpoint_attestation=None,
    enable_encoder=False, saved_encoder=None, saved_encoder_run_id=None,
    enable_qwen=False, qwen_model="qwen2.5:7b", ollama_url="http://localhost:11434",
    qwen_options={}, enable_gemini=False, gemini_model="gemini-3.8-flash",
    gemini_generation_config={"thinkingConfig": {"thinkingLevel": "low"}},
    request_timeout=120, output_root=ROOT / "experiments/runs",
    saved_run=None, saved_run_id=None,
    paper_source_run_id=None,  # Set the exact selected full-dev run ID to enable paper export.
)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    rows, cohort = study.read_study_inputs(settings)
    artifact = study.prepare_study(rows, cohort, settings)
    run = artifact["settings"]
    request_limit = 1 if run.get("run_kind") == "validation" else None
    print("Run:", artifact["run_id"], "Items:", len(rows))


# %%
if RUN_DEVELOPMENT_APPENDIX:
    encoder_predictions = None
    if settings["mode"] == "run" and (run["enable_encoder"] or run["saved_encoder"]):
        if run.get("checkpoint_format") == "colab_state_dict":
            encoder_predictions = study.predict_colab_study(
                artifact, checkpoint=run["checkpoint"], snapshot_path=run["snapshot_path"],
                expected_sha256=run["checkpoint_sha256"], device=run["device"],
            )
        else:
            encoder_predictions = study.predict_encoder(
                artifact, rows, checkpoint=run["checkpoint"], snapshot_path=run["snapshot_path"],
                device=run["device"], saved_path=run["saved_encoder"], saved_run_id=run["saved_encoder_run_id"],
            )
    checkpoint_info, checkpoint_summary = study.checkpoint_report(artifact)
    checkpoint_summary


# %%
if RUN_DEVELOPMENT_APPENDIX:
    if settings["mode"] == "run" and run["enable_qwen"]:
        qwen = study.connect_qwen(artifact)
        study.collect_responses(artifact, "generate", qwen, system="qwen",
                                limit=request_limit, on_record=study.save_study)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    if settings["mode"] == "run" and run["enable_qwen"]:
        study.collect_responses(artifact, "select", qwen, system="qwen",
                                limit=request_limit, on_record=study.save_study)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    if settings["mode"] == "run" and run["enable_gemini"]:
        gemini = study.connect_gemini(artifact)
        study.collect_responses(artifact, "generate", gemini, system="gemini",
                                limit=request_limit, on_record=study.save_study)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    if settings["mode"] == "run" and run["enable_gemini"]:
        study.collect_responses(artifact, "select", gemini, system="gemini",
                                limit=request_limit, on_record=study.save_study)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    artifact_path = None
    if settings["mode"] == "run":
        artifact_path = study.save_study(artifact)
    print("Saved artifact:", artifact_path)


# %%
if RUN_DEVELOPMENT_APPENDIX:
    prediction_view, results = study.display_tables(artifact, start=0, max_items=10)
    display(HTML(prediction_view))
    metrics = results["metrics"]
    disagreements = results["disagreements"]


# %%
if RUN_DEVELOPMENT_APPENDIX:
    paper_manifest = None
    if settings["paper_source_run_id"] is not None:
        paper_manifest = export_paper(
            artifact, expected_run_id=artifact["run_id"],
            paper_source_run_id=settings["paper_source_run_id"],
            output_dir=ROOT / "paper/generated",
        )
        print("Paper source:", paper_manifest["run_id"], paper_manifest["execution_status"])


# %%
if RUN_DEVELOPMENT_APPENDIX:
    saved_sources = []  # [(Path("/path/to/study.json"), "exact-original-run-id"), ...]
    if saved_sources:
        comparison_view, comparison = display_saved_comparison(saved_sources)
        display(HTML(comparison_view))
