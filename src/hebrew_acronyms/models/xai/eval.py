"""One Grok 4.7 Responses request, low reasoning and no tools or retries.

API schema and settings verified 2026-10-09:
https://docs.x.ai/developers/models/grok-4.7
https://docs.x.ai/developers/rest-api-reference/inference/responses
"""
from __future__ import annotations

import os
import requests

from hebrew_acronyms.models.common.http import inspect_model, request_json, validate_request
from hebrew_acronyms.models.gemini.eval import _redact

MODEL = "grok-4.7"
URL = "https://api.x.ai/v1/responses"


def _headers(secret):
    return {"Authorization": "Bearer " + secret, "Content-Type": "application/json"}


def validate_xai_settings(model, timeout=120, max_output_tokens=1024, effort="low"):
    # This study's cap is deliberately bounded; this is not the model's context limit.
    validate_request("validation", model, MODEL, timeout, max_output_tokens, 1024)
    if effort != "low":
        raise ValueError("This study requires low reasoning effort")
    return {"max_output_tokens": max_output_tokens, "reasoning": {"effort": effort},
            "tools": [], "tool_choice": "none", "store": False, "stream": False}


def xai_response(prompt, *, model, max_output_tokens=1024, timeout=120, effort="low"):
    settings = validate_xai_settings(model, timeout, max_output_tokens, effort)
    validate_request(prompt, model, MODEL, timeout, max_output_tokens, 1024)
    secret = os.environ.get("XAI_API_KEY", "")
    payload, result = request_json(
        requests.post, URL, provider="xAI", secret=secret, env_name="XAI_API_KEY",
        headers=_headers(secret), timeout=timeout,
        body={"model": model, "input": [{"role": "user", "content": prompt}], **settings})
    result.update(response="", requested_model=model, model_version=None, finish_reason=None,
                  usage_metadata=None, request_settings={**settings, "timeout_seconds": timeout, "max_attempts": 1})
    if payload is None:
        return _redact(result, secret)
    result.update(model_version=payload.get("model"), response_id=payload.get("id"),
                  usage_metadata=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
                  finish_reason=payload.get("status"))
    output = payload.get("output")
    output = output if isinstance(output, list) else []
    messages, texts, unexpected, refused, incomplete = [], [], False, False, False
    result["thought_parts_omitted"] = 0
    for item in output:
        if not isinstance(item, dict):
            unexpected = True
        elif item.get("type") == "reasoning":
            result["thought_parts_omitted"] += 1
        elif item.get("type") == "message" and item.get("role") == "assistant":
            messages.append(item)
            incomplete |= item.get("status") != "completed"
            content = item.get("content")
            if not isinstance(content, list):
                unexpected = True
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, dict) and part.get("type") == "refusal":
                    refused = True
                else:
                    unexpected = True
        else:
            unexpected = True
    result["response"] = "".join(texts)
    details = payload.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") in {"max_output_tokens", "content_filter"}:
        result["finish_reason"] = details["reason"]
    if refused or result["finish_reason"] == "content_filter":
        result.update(status="blocked", error="xAI refused or filtered the response")
    elif (payload.get("status") != "completed" or payload.get("error") or unexpected
          or incomplete or len(messages) != 1):
        result.update(status="incomplete_response", error="xAI did not return one completed final message without tools")
    elif not result["response"].strip():
        result.update(status="missing_response", error="xAI returned no final text answer")
    elif result["model_version"] != model:
        result.update(status="incomplete_response", error="xAI response model differs from requested model")
    else:
        result.update(status="response_received", error=None)
    return _redact(result, secret)


def inspect_xai_model(*, model, timeout=30):
    """Read exact model metadata without generating text or claiming valid billing."""
    validate_xai_settings(model, timeout)
    secret = os.environ.get("XAI_API_KEY", "")
    return inspect_model(requests.get, "https://api.x.ai/v1/models/" + model,
                         model=model, provider="xAI", secret=secret, env_name="XAI_API_KEY",
                         headers=_headers(secret), timeout=timeout)
