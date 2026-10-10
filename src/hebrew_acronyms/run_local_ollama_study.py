"""Run one local Ollama model through the staged evaluation with sampling seed 42.

Ollama must already be serving the exact model tag; this module never starts it or
pulls weights. Local runs cost nothing, so every price and reserve is zero.

    python -m hebrew_acronyms.run_local_ollama_study pilot --output-dir RUN --code-revision SHA --model TAG
    python -m hebrew_acronyms.run_local_ollama_study full  --output-dir RUN --code-revision SHA --model TAG
"""
from __future__ import annotations

import argparse
import json

from hebrew_acronyms.models.qwen.eval import inspect_ollama
from hebrew_acronyms.staged_evaluation import (
    OLLAMA_SYSTEMS, SYSTEM_NAMES, prepare_session, review_system_pilot,
    run_system_full, run_system_pilot, session_summary,
)

OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 512}
BASE_URL = "http://localhost:11434"
STABLE_FIELDS = ("model", "digest", "server_version", "options",
                 "template", "model_parameters", "details", "model_info")


def build_system(name, model, inspection):
    """Describe one Ollama system whose digest was read from the running server."""
    if name not in OLLAMA_SYSTEMS:
        raise ValueError(f"Local runs support only {OLLAMA_SYSTEMS}")
    return {"name": name, "provider": "qwen", "model": model,
            "settings": {"expected_digest": inspection["digest"], "base_url": BASE_URL,
                         "timeout": 120, "options": dict(OPTIONS)}}


def stable_identity(inspection):
    return {key: inspection[key] for key in STABLE_FIELDS}


def free_session(output_dir, dev_path, test_path, code_revision):
    zero = {name: {"input": 0.0, "output": 0.0} for name in SYSTEM_NAMES}
    return prepare_session(output_dir, dev_path, test_path, code_revision=code_revision,
                           prior_spend_ils=0, rates=zero, reserves=dict.fromkeys(SYSTEM_NAMES, 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pilot", "full"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-revision", required=True, help="40-character commit of the code being run")
    ap.add_argument("--model", required=True, help="exact installed Ollama tag; see ollama list")
    ap.add_argument("--name", default="dictalm", choices=OLLAMA_SYSTEMS)
    ap.add_argument("--dev", default="data/splits/dev_items.csv")
    ap.add_argument("--test", default="data/splits/test_items.csv")
    ap.add_argument("--pilot-identity", help="identity printed by the pilot; required for the full run")
    a = ap.parse_args()

    inspection = inspect_ollama(a.model, base_url=BASE_URL, options=OPTIONS)
    system, identity = build_system(a.name, a.model, inspection), stable_identity(inspection)
    free_session(a.output_dir, a.dev, a.test, a.code_revision)
    if a.stage == "pilot":
        result = run_system_pilot(a.output_dir, system, model_identity=identity)
        review = review_system_pilot(a.output_dir, system, model_identity=identity)
        print("pilot identity (pass to the full run):", review["pilot_identity"])
    else:
        if not a.pilot_identity:
            raise ValueError("Inspect the pilot first and pass its identity with --pilot-identity")
        result = run_system_full(a.output_dir, system, inspected_pilot_identity=a.pilot_identity,
                                 model_identity=identity)
    print(json.dumps(session_summary(a.output_dir)["systems"][a.name], indent=2))
    return result


if __name__ == "__main__":
    main()
