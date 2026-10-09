"""Open-generation / candidate-select evaluation against a local Qwen model via Ollama.

Requires Ollama running locally (`ollama serve` or `brew services start ollama`) with
the explicitly selected model already available. The notebook never starts or pulls it.

    python -m hebrew_acronyms.models.qwen.eval --mode generate
    python -m hebrew_acronyms.models.qwen.eval --mode select
"""
from __future__ import annotations

import argparse
import math

import requests

from hebrew_acronyms.models.common.eval import evaluate, write_details_csv
from hebrew_acronyms.models.common.pairs import load_rows

OLLAMA_URL = "http://localhost:11434/api/generate"


def _request_settings(base_url, timeout):
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Ollama timeout must be positive seconds")
    return base_url.rstrip("/"), timeout


def ollama_model_identity(model, *, base_url="http://localhost:11434", timeout=120):
    """Resolve the exact installed tag to its server-reported digest; never pull/start."""
    base_url, timeout = _request_settings(base_url, timeout)
    response = requests.get(base_url + "/api/tags", timeout=timeout)
    response.raise_for_status()
    matches = [entry for entry in response.json()["models"]
               if model in {entry.get("name"), entry.get("model")}]
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str) or not matches[0]["digest"]:
        raise ValueError(f"Exact Ollama tag {model!r} with a digest is not uniquely installed; check ollama list")
    return matches[0]


def inspect_ollama(model, *, base_url="http://localhost:11434", timeout=120, options=None):
    """Read server identity and model defaults only during an explicitly enabled run.

    Documented API: https://docs.ollama.com/api/tags and
    https://docs.ollama.com/api-reference/show-model-details . No generation here.
    """
    base_url, timeout = _request_settings(base_url, timeout)
    identity = ollama_model_identity(model, base_url=base_url, timeout=timeout)
    response = requests.get(base_url + "/api/version", timeout=timeout)
    response.raise_for_status()
    server_version = response.json()["version"]
    response = requests.post(base_url + "/api/show", json={"model": model}, timeout=timeout)
    response.raise_for_status()
    details = response.json()
    confirmed = ollama_model_identity(model, base_url=base_url, timeout=timeout)
    if confirmed["digest"] != identity["digest"]:
        raise ValueError("Ollama model changed while inspecting it; start a new run")
    return {"model": model, "digest": identity["digest"], "identity_verification": "server_reported_before_generation",
            "server_version": server_version, "base_url": base_url, "timeout_seconds": timeout,
            "options": dict(options or {}), "stream": False, "installed_model": identity,
            "model_parameters": details.get("parameters"), "template": details.get("template"),
            "details": details.get("details"), "model_info": details.get("model_info"),
            "defaults_note": "Explicit options plus server-reported model parameters; unspecified server defaults are not inferred"}


def ollama_response(prompt, *, model, expected_digest, base_url="http://localhost:11434", timeout=120, options=None):
    """Generate once with bounded waits and digest checks; preserve raw text after a failed postcheck."""
    base_url, timeout = _request_settings(base_url, timeout)
    before = ollama_model_identity(model, base_url=base_url, timeout=timeout)
    if before["digest"] != expected_digest:
        raise ValueError("Ollama tag digest changed before the request; start a new run")
    body = {"model": model, "prompt": prompt, "stream": False, "options": dict(options or {})}
    response = requests.post(base_url + "/api/generate", json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    raw = payload.get("response")
    if not isinstance(raw, str):
        raise ValueError("Ollama response must contain text")
    result = {"response": raw, "model": payload.get("model"), "expected_digest": expected_digest,
              "request_settings": {key: value for key, value in body.items() if key != "prompt"},
              "response_metadata": {key: value for key, value in payload.items() if key not in {"response", "context"}},
              "identity_status": "verified", "digest_before": before["digest"], "digest_after": None,
              "identity_error": None, "completion_error": None, "error": None}
    # API done_reason is optional. For new study requests, only an explicit stop
    # confirms completion; length, absent and unknown reasons remain unscored.
    # https://docs.ollama.com/api/generate
    complete = payload.get("done") is True and payload.get("done_reason") == "stop"
    result.update(completion_status="complete" if complete else "incomplete",
                  status="response_received" if complete else "incomplete_response")
    if not complete:
        if payload.get("done") is not True:
            reason = "Ollama did not confirm finished generation"
        elif payload.get("done_reason") == "length":
            reason = "Ollama stopped at the output length limit"
        else:
            reason = "Ollama completion reason is missing or unrecognized"
        result["completion_error"] = reason
    try:
        after = ollama_model_identity(model, base_url=base_url, timeout=timeout)
        result["digest_after"] = after["digest"]
        if after["digest"] != expected_digest or payload.get("model") != model:
            raise ValueError("Ollama model identity changed or response model differs")
    except Exception:
        # HTTP exception strings can contain URLs, credentials or response bodies.
        result.update(identity_status="unverified", identity_error="Ollama post-request model identity check failed")
    result["error"] = "; ".join(message for message in
                               (result["completion_error"], result["identity_error"]) if message) or None
    return result


def ollama_generate(prompt: str, model: str = "qwen2.5:7b", *, timeout=120) -> str:
    """Compatibility text interface; study runs use ollama_response for identity evidence."""
    _request_settings("http://localhost:11434", timeout)
    response = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="data/splits/dev_items.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--mode", default="generate", choices=["generate", "select"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/qwen/{a.mode}_details.csv"

    res = evaluate(load_rows(a.items),
                   generate_fn=lambda p: ollama_generate(p, model=a.model),
                   mode=a.mode)
    print(f"{a.items}  (model: {a.model}, mode: {a.mode})\n")
    print(f"  items scored      {res['n_items']}")
    print(f"  accuracy          {res['accuracy']:.3f}")
    print(f"  invalid rate      {res['invalid_rate']:.3f}   response matched no candidate")

    write_details_csv(out, res["details"])
    print(f"\n  per-item details written to {out}")


if __name__ == "__main__":
    main()
