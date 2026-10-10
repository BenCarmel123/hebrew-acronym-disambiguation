"""One GPT-4.1 mini request with a final-text/usage record and no automatic retries.

Verified 2026-10-09:
https://developers.openai.com/api/docs/models/gpt-4.1-mini
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
"""
from __future__ import annotations

import math
import os

import requests

from hebrew_acronyms.models.common.http import inspect_model, request_json, validate_request
from hebrew_acronyms.models.gemini.eval import _redact

MODEL = "gpt-4.1-mini-2025-04-14"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _headers(secret):
    return {"Authorization": "Bearer " + secret, "Content-Type": "application/json"}


def validate_openai_settings(model, timeout=120, max_output_tokens=128, temperature=0):
    validate_request("validation", model, MODEL, timeout, max_output_tokens, 32768)
    if type(temperature) not in {int, float} or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("OpenAI temperature must be finite and between 0 and 2")
    return {"max_completion_tokens": max_output_tokens, "temperature": temperature,
            "stream": False, "n": 1, "store": False}


def openai_response(prompt, *, model, max_output_tokens=128, timeout=120, temperature=0):
    settings = validate_openai_settings(model, timeout, max_output_tokens, temperature)
    validate_request(prompt, model, MODEL, timeout, max_output_tokens, 32768)
    secret = os.environ.get("OPENAI_API_KEY", "")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], **settings}
    payload, result = request_json(requests.post, OPENAI_URL, provider="OpenAI", secret=secret,
                                  env_name="OPENAI_API_KEY", headers=_headers(secret), timeout=timeout, body=body)
    result.update(response="", requested_model=model, model_version=None, finish_reason=None,
                  usage_metadata=None, request_settings={**settings, "timeout_seconds": timeout, "max_attempts": 1})
    if payload is None:
        return _redact(result, secret)
    result["model_version"] = payload.get("model")
    result["usage_metadata"] = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    result["response_id"] = payload.get("id")
    result["system_fingerprint"] = payload.get("system_fingerprint")
    choices = payload.get("choices")
    choices = choices if isinstance(choices, list) else []
    texts = []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        texts.append(text if isinstance(text, str) else "")
    result["response"] = texts[0] if texts else ""
    if len(texts) > 1:
        result["candidate_texts"] = texts
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    finish = choice.get("finish_reason")
    result["finish_reason"] = finish if isinstance(finish, str) else None
    if finish == "content_filter" or message.get("refusal"):
        result.update(status="blocked", error="OpenAI refused or filtered the response")
    elif len(choices) > 1:
        result.update(status="incomplete_response", error="Unexpected multiple OpenAI choices; no answer selected")
    elif finish != "stop":
        result.update(status="incomplete_response", error="OpenAI did not report a completed final answer")
    elif not result["response"].strip():
        result.update(status="missing_response", error="OpenAI returned no final text answer")
    elif result["model_version"] != model:
        result.update(status="incomplete_response", error="OpenAI response model differs from requested snapshot")
    else:
        result.update(status="response_received", error=None)
    return _redact(result, secret)


def inspect_openai_model(*, model, timeout=30):
    """GET exact model metadata; does not generate text or verify available credit."""
    validate_openai_settings(model, timeout)
    secret = os.environ.get("OPENAI_API_KEY", "")
    return inspect_model(requests.get, "https://api.openai.com/v1/models/" + model,
                         model=model, provider="OpenAI", secret=secret, env_name="OPENAI_API_KEY",
                         headers=_headers(secret), timeout=timeout)
