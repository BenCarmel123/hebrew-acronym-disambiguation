"""Throttled, retrying client for the Sefaria API.

Mirrors `wiki_client.WikiAPI`'s shape (throttle + exponential backoff, one
`get` chokepoint) so mining code that already knows that pattern reads the
same way here. Sefaria serves rabbinic/Talmudic texts, where acronyms like
מהר"ש name a specific rabbi through real, natural usage — unlike Wikipedia
prose, which mostly cannot disambiguate that kind of acronym at all (see
`data/splits/README.md`'s `unsure` rabbinic-name rows).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

LOG = logging.getLogger(__name__)

API_URL = "https://www.sefaria.org/api"
USER_AGENT = (
    "TAU-NLP-2526b-HebrewAcronymDisambiguation/0.1 "
    "(research project; contact: bencarmel123@gmail.com) python-requests"
)


class SefariaAPI:
    """Minimal wrapper around Sefaria's public REST API. No auth required."""

    def __init__(
        self,
        api_url: str = API_URL,
        delay: float = 0.3,
        timeout: float = 30.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.api_url = api_url
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.monotonic()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """One HTTP call to `{api_url}{path}`, with throttling and retries."""
        url = f"{self.api_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                # Any other 4xx (e.g. a malformed or too-specific ref) will not
                # succeed on retry — fail immediately rather than burning the
                # whole backoff schedule on a request that can never work.
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Sefaria call failed: {exc}") from exc
                last_error = exc
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            sleep_for = min(30.0, (2**attempt) * 0.5) + random.uniform(0, 0.25)
            retry_after = getattr(getattr(last_error, "response", None), "headers", {}).get("Retry-After")
            if retry_after:
                try:
                    sleep_for = max(sleep_for, min(60.0, float(retry_after)))
                except ValueError:
                    pass
            LOG.warning("Sefaria call failed (%s); retrying in %.1fs", last_error, sleep_for)
            time.sleep(sleep_for)
        raise RuntimeError(f"Sefaria call failed after {self.max_retries} attempts: {last_error}")

    def search(self, query: str, size: int = 20) -> list[dict]:
        """Full-text search. Returns `[{ref, snippets}, ...]`.

        `snippets` are highlighted fragments from Sefaria's search index —
        useful for finding *which refs* mention the query, but frequently
        clause-fragmented (Sefaria's tokenizer breaks on the internal geresh
        of an acronym like מהר"ש). Fetch the ref's own text via `text()` for
        real sentence boundaries; never mine snippets directly, same reason
        `mine_sentences.py`'s docstring gives for Wikipedia search snippets.
        """
        data = self._request(
            "POST", "/search-wrapper",
            json={"query": query, "type": "text", "size": size, "field": "naive_lemmatizer"},
        )
        hits = data.get("hits", {}).get("hits", [])
        return [
            {"ref": h["_id"], "snippets": h.get("highlight", {}).get("naive_lemmatizer", [])}
            for h in hits
        ]

    def text(self, ref: str) -> list[str]:
        """Hebrew text segments for one ref (a book section, e.g. a daf or siman).

        Returns the `he` field: a list of segment strings, each roughly a
        paragraph/verse. `ref` normally comes from `search()`'s `ref` field.
        Segments may carry markup (`<small>`, `<sup>`, footnote `<i>` blocks)
        that the caller should strip before sentence-splitting.
        """
        import urllib.parse

        data = self._request("GET", f"/texts/{urllib.parse.quote(ref)}", params={"lang": "he"})
        he = data.get("he")
        if he is None:
            return []
        return he if isinstance(he, list) else [he]
