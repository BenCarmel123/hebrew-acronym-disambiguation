"""Small HTTP boundary for explicit provider requests; no import-time I/O."""
from __future__ import annotations

import math
import time

import requests

from hebrew_acronyms.models.gemini.eval import _redact
from hebrew_acronyms.models.common.errors import http_error_diagnostics


def validate_request(prompt, model, expected_model, timeout, max_output_tokens, maximum):
    if model != expected_model:
        raise ValueError("Use the exact model authorized for this provider; no fallback is performed")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Request timeout must be finite positive seconds")
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= maximum:
        raise ValueError("Output token cap must be a positive integer within the model limit")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty text prompt is required")


def request_json(method, url, *, provider, secret, env_name, headers, timeout, body=None):
    """One bounded HTTP request. Error bodies and exception strings are never retained."""
    result = {"status": "service_error", "attempts": 0, "http_status": None,
              "retryable": False, "retry_after": None, "request_id": None,
              "error_category": None, "provider_blocked": False,
              "elapsed_seconds": 0.0, "error": None}
    if not secret:
        result.update(error=f"{env_name} is not set in the environment",
                      error_category="missing_credentials", provider_blocked=True)
        return None, result
    result["attempts"] = 1
    started = time.monotonic()
    try:
        kwargs = {"headers": headers, "timeout": timeout, "allow_redirects": False}
        if body is not None:
            kwargs["json"] = body
        response = method(url, **kwargs)
        result["http_status"] = response.status_code
        response_headers = getattr(response, "headers", {})
        if not 200 <= response.status_code < 300:
            result.update(http_error_diagnostics(response, provider=provider))
            result["error"] = f"{provider} HTTP {response.status_code}; no retry attempted"
            return None, result
        request_id = response_headers.get("request-id") or response_headers.get("x-request-id")
        result["request_id"] = request_id if isinstance(request_id, str) else None
        payload = response.json()
        if not isinstance(payload, dict):
            result.update(error=f"{provider} returned an invalid response object", error_category="invalid_response")
            return None, result
        return _redact(payload, secret), result
    except (requests.Timeout, requests.ConnectionError):
        result.update(retryable=True, error_category="connection",
                      error=f"{provider} request timed out or connection failed; no retry attempted")
        return None, result
    except Exception:
        result.update(error=f"{provider} request or response decoding failed; no retry attempted",
                      error_category="request_error")
        return None, result
    finally:
        result["elapsed_seconds"] = time.monotonic() - started


def inspect_model(method, url, *, model, provider, secret, env_name, headers, timeout):
    """Read model metadata without generation; a successful lookup is not a billing check."""
    payload, result = request_json(method, url, provider=provider, secret=secret,
                                  env_name=env_name, headers=headers, timeout=timeout)
    result.update(requested_model=model, model_version=None, model_metadata=None,
                  availability="unverified")
    if payload is not None:
        result["model_version"] = payload.get("id")
        result["model_metadata"] = payload
        if payload.get("id") == model:
            result.update(status="available", availability="model_lookup_succeeded", error=None)
        else:
            result["error"] = f"{provider} returned a different or missing model ID; no substitution allowed"
    return _redact(result, secret)
