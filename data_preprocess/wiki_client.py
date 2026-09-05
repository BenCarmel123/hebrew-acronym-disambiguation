"""Throttled, retrying client for the Hebrew Wikipedia action API."""
from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Iterator

import requests

LOG = logging.getLogger(__name__)

API_URL = "https://he.wikipedia.org/w/api.php"
USER_AGENT = (
    "TAU-NLP-2526b-HebrewAcronymDisambiguation/0.1 "
    "(research project; contact: bencarmel123@gmail.com) python-requests"
)


class WikiAPI:
    """Minimal wrapper around the MediaWiki action API.

    Politeness is built in: a fixed inter-request delay plus exponential
    backoff with jitter on 429/5xx, which is what the API etiquette guide asks
    for. Continuation is handled transparently by `query_all`.
    """

    def __init__(
        self,
        api_url: str = API_URL,
        delay: float = 0.25,
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

    def get(self, **params: Any) -> dict:
        """One API call, with throttling and retries. Returns parsed JSON."""
        params.setdefault("format", "json")
        params.setdefault("formatversion", 2)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                sleep_for = min(30.0, (2**attempt) * 0.5) + random.uniform(0, 0.25)
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
                if retry_after:
                    try:
                        sleep_for = max(sleep_for, min(60.0, float(retry_after)))
                    except ValueError:
                        pass
                LOG.warning("API call failed (%s); retrying in %.1fs", exc, sleep_for)
                time.sleep(sleep_for)
                continue
            if "error" in data:
                raise RuntimeError(f"MediaWiki error: {data['error']}")
            return data
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def query_all(self, **params: Any) -> Iterator[dict]:
        """Yield each `query` payload, following `continue` to the last page."""
        params.setdefault("action", "query")
        cont: dict[str, Any] = {}
        while True:
            data = self.get(**{**params, **cont})
            if "query" in data:
                yield data["query"]
            if "continue" not in data:
                return
            cont = data["continue"]

    # -- convenience endpoints -------------------------------------------------

    def category_members(self, category: str, limit: int = 500) -> Iterator[dict]:
        """Yield page dicts (`pageid`, `title`) for a category, main namespace."""
        for chunk in self.query_all(
            list="categorymembers",
            cmtitle=category,
            cmlimit=limit,
            cmnamespace=0,
            cmtype="page",
        ):
            yield from chunk.get("categorymembers", [])

    def wikitext(self, title: str) -> str:
        """Raw wikitext of a page ('' when the page is missing)."""
        data = self.get(action="parse", page=title, prop="wikitext", redirects=1)
        return data.get("parse", {}).get("wikitext", "") or ""

    def wikitext_batch(self, titles: list[str], batch_size: int = 50) -> dict[str, str]:
        """Wikitext for many pages at once.

        The action API accepts up to 50 titles per query, so fetching an entire
        category costs ~1/50th of the requests that per-page `parse` calls do.
        Missing pages are simply absent from the result.
        """
        out: dict[str, str] = {}
        for i in range(0, len(titles), batch_size):
            chunk = titles[i : i + batch_size]
            for payload in self.query_all(
                prop="revisions",
                rvprop="content",
                rvslots="main",
                titles="|".join(chunk),
            ):
                for page in payload.get("pages", []):
                    if page.get("missing"):
                        continue
                    revisions = page.get("revisions") or []
                    if not revisions:
                        continue
                    content = revisions[0].get("slots", {}).get("main", {}).get("content")
                    if content:
                        out[page["title"]] = content
        return out

    def search_hits(self, phrase: str) -> int:
        """Total number of articles matching an exact-phrase search."""
        data = self.get(
            action="query",
            list="search",
            srsearch=f'"{phrase}"',
            srlimit=1,
            srnamespace=0,
            srinfo="totalhits",
            srprop="",
        )
        return int(data.get("query", {}).get("searchinfo", {}).get("totalhits", 0))

    def search_snippets(self, phrase: str, limit: int = 10) -> list[dict]:
        """Full-text search hits for an exact phrase, with a highlighted snippet.

        Returns `[{title, snippet}, ...]`; `snippet` carries `<span
        class="searchmatch">` around the matched phrase, same as the web search
        UI. This is a source of real usage contexts, unlike `search_hits`, which
        only counts.

        A phrase containing a literal `"` (gershayim-marked acronyms are
        normally rendered with an ASCII double quote, e.g. `שב"ת`) breaks the
        `"phrase"` exact-match syntax — the embedded quote splits it into two
        adjacent quoted fragments instead of one. `insource:/regex/` matches the
        literal text without that ambiguity, but CirrusSearch's regex parser
        also treats a bare `"` as a delimiter inside the pattern itself, so it
        is put in a one-character class (`["]`) rather than escaped with a
        backslash — both are equivalent regexes, but only the character-class
        form parses cleanly server-side.
        """
        if '"' in phrase:
            pattern = re.escape(phrase).replace('"', '["]')
            srsearch = f"insource:/{pattern}/"
        else:
            srsearch = f'"{phrase}"'
        data = self.get(
            action="query",
            list="search",
            srsearch=srsearch,
            srlimit=limit,
            srnamespace=0,
            srprop="snippet",
        )
        return [
            {"title": r["title"], "snippet": r.get("snippet", "")}
            for r in data.get("query", {}).get("search", [])
        ]

    def search_cooccurrence(self, acronym: str, expansion: str, limit: int = 20) -> list[str]:
        """Titles of pages containing both `acronym` and `expansion`.

        Mining an acronym alone yields whichever sense dominates the corpus —
        every `מ"מ` sentence comes back as מילימטר rainfall, and the rarer
        senses never surface. Requiring the expansion to appear on the same
        page biases the fetch toward pages actually about that sense, so each
        expansion gets its own pool of candidate pages.

        The page is evidence about the sense, not proof of it: a page about
        ממלא מקום can still use `מ"מ` for something else. The label this
        supports is provisional, for a human to confirm.

        Both terms go through `insource:` because the acronym's embedded quote
        breaks `"phrase"` matching (see `search_snippets`).
        """
        acr_pattern = re.escape(acronym).replace('"', '["]')
        terms = f'insource:/{acr_pattern}/ insource:"{expansion}"'
        data = self.get(
            action="query",
            list="search",
            srsearch=terms,
            srlimit=limit,
            srnamespace=0,
            srprop="",
        )
        return [r["title"] for r in data.get("query", {}).get("search", [])]
