"""One Haiku 5.5 message with bounded thinking/output and sanitized provenance.

Verified 2026-10-09:
https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide
https://platform.claude.com/docs/en/api/messages/create
Thinking counts toward max_tokens; low effort is not disabled thinking. Never send
legacy budget_tokens or temperature/top_p/top_k/seed controls to this model.
"""
from __future__ import annotations

import os

import requests

from hebrew_acronyms.models.common.http import inspect_model, request_json, validate_request
from hebrew_acronyms.models.gemini.eval import _redact

MODEL = "claude-haiku-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _headers(secret):
    return {"x-api-key": secret, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}


def validate_anthropic_settings(model, timeout=120, max_output_tokens=1024, effort="low"):
    validate_request("validation", model, MODEL, timeout, max_output_tokens, 128000)
    if not isinstance(effort, str) or effort not in {"low", "medium", "high"}:
        raise ValueError("This text evaluation supports low, medium or high Anthropic effort")
    return {"max_tokens": max_output_tokens, "stream": False,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def anthropic_response(prompt, *, model, max_output_tokens=1024, timeout=120, effort="low"):
    settings = validate_anthropic_settings(model, timeout, max_output_tokens, effort)
    validate_request(prompt, model, MODEL, timeout, max_output_tokens, 128000)
    secret = os.environ.get("ANTHROPIC_API_KEY", "")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], **settings}
    payload, result = request_json(requests.post, ANTHROPIC_URL, provider="Anthropic", secret=secret,
                                  env_name="ANTHROPIC_API_KEY", headers=_headers(secret), timeout=timeout, body=body)
    result.update(response="", requested_model=model, model_version=None, finish_reason=None,
                  usage_metadata=None, thought_parts_omitted=0,
                  request_settings={**settings, "timeout_seconds": timeout, "max_attempts": 1})
    if payload is None:
        return _redact(result, secret)
    result["model_version"] = payload.get("model")
    result["usage_metadata"] = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    result["response_id"] = payload.get("id")
    content = payload.get("content")
    texts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"thinking", "redacted_thinking"}:
            result["thought_parts_omitted"] += 1
        elif part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    result["response"] = "".join(texts)
    finish = payload.get("stop_reason")
    result["finish_reason"] = finish if isinstance(finish, str) else None
    stop_details = payload.get("stop_details")
    refused = isinstance(stop_details, dict) and stop_details.get("type") == "refusal"
    if finish == "refusal" or refused:
        result.update(status="blocked", error="Anthropic refused the response")
    elif finish != "end_turn":
        result.update(status="incomplete_response", error="Anthropic did not report a completed final answer")
    elif not result["response"].strip():
        result.update(status="missing_response", error="Anthropic returned no final text answer")
    elif result["model_version"] != model:
        result.update(status="incomplete_response", error="Anthropic response model differs from requested model")
    else:
        result.update(status="response_received", error=None)
    return _redact(result, secret)


def inspect_anthropic_model(*, model, timeout=30):
    """GET exact model capabilities; does not generate text or verify available credit."""
    validate_anthropic_settings(model, timeout)
    secret = os.environ.get("ANTHROPIC_API_KEY", "")
    return inspect_model(requests.get, "https://api.anthropic.com/v1/models/" + model,
                         model=model, provider="Anthropic", secret=secret, env_name="ANTHROPIC_API_KEY",
                         headers=_headers(secret), timeout=timeout)
