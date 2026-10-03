"""One explicit Gemini text request with safe credentials and response provenance.

No configuration files, credentials, services or model calls are touched on import.
The study notebook owns prompts and evaluation; this module never scores answers.

API references:
https://ai.google.dev/api/generate-content
https://ai.google.dev/gemini-api/docs/api-key
https://ai.google.dev/gemini-api/docs/generate-content/latest-model
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import re

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_BLOCKED_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


def _redact(value, secret):
    """Remove the current credential even if an upstream response unexpectedly echoes it."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


def validate_gemini_settings(model, timeout=120, generation_config=None):
    """Validate the deliberately small text-only configuration without contacting Gemini.

    Gemini 3.8 accepts thinkingLevel low/medium/high. No legacy thinkingBudget,
    candidateCount or sampling controls are inferred or sent. Omitted settings use
    provider defaults, whose resolved values are not reported by generateContent.
    """
    if not isinstance(model, str) or not re.fullmatch(r"gemini-[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise ValueError("Set an explicit Gemini model code without a URL, path or query")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Gemini timeout must be finite positive seconds")
    config = {} if generation_config is None else deepcopy(generation_config)
    allowed = {"thinkingConfig", "maxOutputTokens", "stopSequences", "responseMimeType"}
    if not isinstance(config, dict) or not set(config) <= allowed:
        raise ValueError("Unsupported Gemini generation configuration; use documented text/thinking settings")
    if "maxOutputTokens" in config and (type(config["maxOutputTokens"]) is not int or config["maxOutputTokens"] <= 0):
        raise ValueError("maxOutputTokens must be a positive integer")
    if "stopSequences" in config and (not isinstance(config["stopSequences"], list)
            or any(not isinstance(value, str) or not value for value in config["stopSequences"])):
        raise ValueError("stopSequences must be a list of nonempty strings")
    if "responseMimeType" in config and config["responseMimeType"] != "text/plain":
        raise ValueError("The study requires text/plain responses")
    if "thinkingConfig" in config:
        if not re.match(r"gemini-3[.-]", model):
            raise ValueError("The documented thinkingLevel configuration requires a Gemini 3 model")
        thinking = config["thinkingConfig"]
        if not isinstance(thinking, dict) or not set(thinking) <= {"thinkingLevel", "includeThoughts"}:
            raise ValueError("Use thinkingLevel rather than the historical thinkingBudget flag")
        if "thinkingLevel" in thinking and thinking["thinkingLevel"] not in {"low", "medium", "high", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError("This text study supports documented low, medium or high thinking levels")
        if "includeThoughts" in thinking and type(thinking["includeThoughts"]) is not bool:
            raise ValueError("includeThoughts must be a boolean")
    # A misplaced credential must not enter the request body or saved settings.
    secret = os.environ.get("GEMINI_API_KEY", "")
    if secret and (secret in model or secret in json.dumps(config, ensure_ascii=False)):
        raise ValueError("Credentials belong only in GEMINI_API_KEY, not model or generation settings")
    return config


def gemini_response(prompt, *, model, timeout=120, generation_config=None):
    """Return one final-text response and metadata; never retry or reveal error bodies.

    The model is always explicit. model_version is the provider's modelVersion,
    not an Ollama-style digest. Thought text/signatures are omitted. Partial final
    text is retained when generation is blocked, incomplete or malformed.
    """
    config = validate_gemini_settings(model, timeout, generation_config)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Gemini requires a nonempty text prompt")
    secret = os.environ.get("GEMINI_API_KEY", "")
    result = {"response": "", "status": "service_error", "requested_model": model,
              "model_version": None, "finish_reason": None, "usage_metadata": None,
              "request_settings": {"generationConfig": config, "timeout_seconds": timeout, "max_attempts": 1},
              "settings_resolution": "Only requested settings are recorded; omitted provider defaults are not resolved by this API",
              "attempts": 0, "http_status": None, "prompt_block_reason": None,
              "thought_parts_omitted": 0, "candidate_count": 0, "error": None}
    if not secret:
        result["error"] = "GEMINI_API_KEY is not set in the environment"
        return result
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    result["attempts"] = 1
    try:
        response = requests.post(GEMINI_URL.format(model=model),
                                 headers={"x-goog-api-key": secret, "Content-Type": "application/json"},
                                 json=body, timeout=timeout, allow_redirects=False)
        result["http_status"] = response.status_code
        if not 200 <= response.status_code < 300:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            result["retry_after"] = retry_after if isinstance(retry_after, str) else None
            result["retryable"] = response.status_code in {408, 429, 500, 502, 503, 504}
            result["error"] = f"Gemini HTTP {response.status_code}; no retry attempted"
            return _redact(result, secret)
        payload = response.json()
    except (requests.Timeout, requests.ConnectionError):
        result["retryable"] = True
        result["error"] = "Gemini request timed out or connection failed; no retry attempted"
        return _redact(result, secret)
    except Exception:
        # requests errors and JSON decoders may include URLs, bodies or headers.
        result["error"] = "Gemini request or response decoding failed; no retry attempted"
        return _redact(result, secret)
    if not isinstance(payload, dict):
        result["error"] = "Gemini returned an invalid response object"
        return _redact(result, secret)
    result["model_version"] = payload.get("modelVersion") if isinstance(payload.get("modelVersion"), str) else None
    result["usage_metadata"] = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else None
    feedback = payload.get("promptFeedback")
    block_reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
    if isinstance(block_reason, str) and block_reason != "BLOCK_REASON_UNSPECIFIED":
        result["prompt_block_reason"] = block_reason
    candidates = payload.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    result["candidate_count"] = len(candidates)
    texts = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts", []) if isinstance(content, dict) else []
        final_parts = []
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("thought"):
                result["thought_parts_omitted"] += 1
            elif isinstance(part.get("text"), str):
                final_parts.append(part["text"])
        texts.append("".join(final_parts))
    result["response"] = texts[0] if texts else ""
    if len(texts) > 1:
        result["candidate_texts"] = texts  # Retain evidence; never choose the best candidate.
    first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    finish = first.get("finishReason")
    finish = finish if isinstance(finish, str) else None
    result["finish_reason"] = finish
    if result["prompt_block_reason"] is not None or finish in _BLOCKED_FINISH:
        result.update(status="blocked", error="Gemini blocked the prompt or response")
    elif len(candidates) > 1:
        result.update(status="incomplete_response", error="Unexpected multiple Gemini candidates; no candidate selection performed")
    elif not result["response"].strip() and finish in {None, "STOP", "FINISH_REASON_UNSPECIFIED"}:
        result.update(status="missing_response", error="Gemini returned no final text answer")
    elif finish != "STOP":
        result.update(status="incomplete_response", error="Gemini did not report a completed final answer")
    else:
        result.update(status="response_received", error=None)
    return _redact(result, secret)


def gemini_generate(prompt, model, *, timeout=120, generation_config=None):
    """Compatibility text wrapper with an explicit model and sanitized failures."""
    result = gemini_response(prompt, model=model, timeout=timeout, generation_config=generation_config)
    if result["status"] != "response_received":
        raise RuntimeError(result["error"])
    return result["response"]


def main():
    """One explicit diagnostic prompt, with metadata and no historical scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--thinking-level", choices=["low", "medium", "high"])
    args = parser.parse_args()
    config = {"thinkingConfig": {"thinkingLevel": args.thinking_level}} if args.thinking_level else {}
    result = gemini_response(args.prompt, model=args.model, timeout=args.timeout, generation_config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
