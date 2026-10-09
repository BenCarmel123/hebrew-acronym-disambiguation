"""Allowlisted HTTP failure diagnostics; never return upstream error text.

429 alone cannot distinguish rate limits from exhausted quota. Google QuotaFailure
details can identify per-minute, per-day or zero allocation; daily/zero quota stops
the current collection but does not establish that billing is disabled. See:
https://ai.google.dev/gemini-api/docs/troubleshooting
https://developers.openai.com/api/docs/guides/error-codes
https://platform.claude.com/docs/en/api/errors
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re


def retry_after_seconds(value, *, now=None):
    """Parse seconds or an HTTP-date without retaining the untrusted header.

    Do not cap valid delays here: the runner must stop rather than retry earlier
    when the server's delay exceeds its wait allowance.
    """
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    if isinstance(value, str) and len(value) > 128:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError, OverflowError):
        if not isinstance(value, str):
            return None
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - (now or datetime.now(timezone.utc))).total_seconds()
            return max(0.0, seconds) if math.isfinite(seconds) else None
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def http_error_diagnostics(response, *, provider):
    """Inspect only known fields in memory and emit fixed labels and numbers."""
    status = response.status_code
    headers = getattr(response, "headers", {})
    delay = retry_after_seconds(headers.get("Retry-After"))
    error = {}
    try:
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
    except Exception:
        pass  # Decoding exceptions can themselves contain credentials or bodies.
    codes = {value.lower() for key in ("code", "type", "status")
             if isinstance((value := error.get(key)), str)}
    message = error.get("message", "")
    message = message.lower() if isinstance(message, str) else ""
    quota = rate = billing = False
    if provider == "OpenAI":
        quota = bool(codes & {"insufficient_quota", "quota_exceeded"})
        billing = bool(codes & {"billing_hard_limit_reached", "billing_not_active"})
        rate = "rate_limit_exceeded" in codes
    elif provider == "Anthropic":
        billing = ("credit balance is too low" in message or
                   bool(codes & {"billing_error", "credit_balance_too_low"}))
        rate = "rate_limit_error" in codes
    elif provider == "Gemini":
        # Some Gemini responses put the zero allocation only in the message.
        # Recognize this narrow numeric fact without retaining any message text.
        quota = bool(re.search(r"quota exceeded for metric: [a-z0-9_./]+, limit: 0(?:[\s,]|$)", message))
        details = error.get("details", [])
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            kind = detail.get("@type")
            if kind == "type.googleapis.com/google.rpc.ErrorInfo":
                billing |= detail.get("reason") in ("BILLING_DISABLED", "BILLING_NOT_ACTIVE", "BILLING_ACCOUNT_DISABLED")
            if kind == "type.googleapis.com/google.rpc.QuotaFailure":
                violations = detail.get("violations", [])
                for violation in violations if isinstance(violations, list) else []:
                    if not isinstance(violation, dict):
                        continue
                    quota_id = violation.get("quotaId", "")
                    quota_id = quota_id.lower() if isinstance(quota_id, str) else ""
                    allocation = violation.get("quotaValue")
                    quota |= "perday" in quota_id or (type(allocation) in {int, float, str} and allocation in (0, "0"))
                    rate |= "perminute" in quota_id or "persecond" in quota_id
            if kind == "type.googleapis.com/google.rpc.RetryInfo":
                value = detail.get("retryDelay")
                if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?s", value):
                    parsed = retry_after_seconds(value[:-1])
                    if parsed is not None:
                        delay = max(delay or 0, parsed)
    if billing or status == 402:
        category, blocked, retryable = "billing_blocked", True, False
    elif quota:
        category, blocked, retryable = "quota_exhausted", True, False
    elif status == 401:
        category, blocked, retryable = "authentication", True, False
    elif status == 403:
        category, blocked, retryable = "permission", True, False
    elif status == 429:
        category, blocked, retryable = "rate_limit" if rate else "rate_or_quota_unknown", False, True
    elif status in {408, 500, 502, 503, 504, 529}:
        category, blocked, retryable = "transient_service", False, True
    else:
        category, blocked, retryable = "http_error", False, False
    return {"error_category": category, "provider_blocked": blocked,
            "retryable": retryable, "retry_after": delay}
