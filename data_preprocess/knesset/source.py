"""Mine natural acronym usage from the Knesset Proceedings Corpus.

Unlike Wikipedia/Sefaria, this source is not queried per-acronym over an API
— it is a bulk dataset of already-segmented sentences
(huggingface.co/datasets/HaifaCLGroup/KnessetCorpus), downloaded as .jsonl.bz2
shards and scanned locally. Each shard is one protocol (a plenary or committee
session) containing a `protocol_sentences` list; every sentence is already a
clean unit (`sentence_text`), so there is no page-to-sentence splitting step
like `mine_sentences.page_sentences` — only the acronym/prose filters apply.

Register: formal parliamentary speech, decades of sessions (1992-2024),
frequent real acronym usage (ministries, laws, military and organisational
names) in genuinely disambiguating context — unlike Wikipedia's encyclopedic
prose, a Knesset speaker uses an acronym the way a reader must actually
resolve it from what is being debated.

    # fetch some shards (see download_shards), then:
    python -m data_preprocess knesset-mine \\
        --shards-dir data/raw/knesset_shards \\
        --acronyms data/mined/knesset/target_types.txt \\
        --out data/mined/knesset/knesset_mined.csv
"""
from __future__ import annotations

import bz2
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

import requests

from ..common import hebrew_text
from ..common.filters import ACRONYM_TOKEN_RE, ContextRow, is_clean_sentence

LOG = logging.getLogger(__name__)

REPO_ID = "HaifaCLGroup/KnessetCorpus"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

_CONFIG_DIRS = {
    "plenary_protocols": "protocols_sentences/plenary_protocols/data",
    "committee_protocols": "protocols_sentences/committee_protocols/data",
}


def shard_paths(config: str = "plenary_protocols") -> str:
    """Directory (within the dataset repo) holding one config's .jsonl.bz2 shards."""
    return _CONFIG_DIRS[config]


def list_remote_shards(config: str = "plenary_protocols") -> list[str]:
    """Full repo-relative paths of every .jsonl.bz2 shard for one config.

    `plenary_protocols/data/` is flat (~1,000 files); `committee_protocols/data/`
    is one directory per Knesset number (15-25, ~1,000 files each, ~11,000
    total) — so listing has to recurse one level for committee protocols, and
    the result mixes shards from every Knesset number rather than one
    contiguous block, which matters for `download_shards`' `n` truncation:
    "the first n" spans Knesset numbers rather than exhausting one before the
    next.
    """
    base = shard_paths(config)
    resp = requests.get(f"{API_URL}/{base}", timeout=30)
    resp.raise_for_status()
    entries = resp.json()
    files = [f["path"] for f in entries if f["type"] == "file"]
    subdirs = [f["path"] for f in entries if f["type"] == "directory"]
    for subdir in subdirs:
        sub_resp = requests.get(f"{API_URL}/{subdir}", timeout=30)
        sub_resp.raise_for_status()
        files.extend(f["path"] for f in sub_resp.json() if f["type"] == "file")
    return files


def download_shards(
    config: str = "plenary_protocols", *, n: int = 30, out_dir: str = "data/raw/knesset_shards",
    max_retries: int = 3,
) -> list[Path]:
    """Download the first `n` shards for a config into `out_dir`. Skips existing files.

    The corpus is a fixed public dump, not something mining code should
    re-fetch on every run — shards already on disk are left alone, so a
    second call with a larger `n` only fetches what is new.

    A batch of thousands of downloads (committee_protocols alone is ~9,000
    files) will hit an occasional connection reset or read timeout — a
    transient network condition, not a reason to abandon everything already
    fetched. Each shard gets its own small retry budget; a shard that still
    fails is logged and skipped so the run reaches the end of the list rather
    than dying on file #500 of #9,166 and losing no progress but wasting the
    time already spent.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    remote_paths = list_remote_shards(config)[:n]
    local_paths = []
    failed = []
    for i, remote_path in enumerate(remote_paths, 1):
        local_path = out / Path(remote_path).name
        if local_path.exists():
            local_paths.append(local_path)
            continue
        for attempt in range(max_retries):
            try:
                resp = requests.get(f"{RESOLVE_URL}/{remote_path}", timeout=120)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                local_paths.append(local_path)
                break
            except requests.RequestException as exc:
                LOG.warning("download failed (attempt %d/%d) for %s: %s",
                            attempt + 1, max_retries, remote_path, exc)
        else:
            failed.append(remote_path)
        if i % 200 == 0:
            LOG.info("downloaded %d/%d shards (%d failed so far)", i, len(remote_paths), len(failed))
    if failed:
        LOG.warning("%d/%d shards could not be downloaded after retries", len(failed), len(remote_paths))
    return local_paths


def iter_shard_sentences(shard_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (sentence_text, protocol_name) for every sentence in one shard file."""
    opener = bz2.open if shard_path.suffix == ".bz2" else open
    with opener(shard_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            protocol = json.loads(line)
            name = protocol.get("protocol_name", shard_path.name)
            for sent in protocol.get("protocol_sentences", []):
                text = sent.get("sentence_text", "")
                if text:
                    yield text, name


def mine_knesset(
    shard_paths: Iterable[Path],
    acronyms: Iterable[str],
    *,
    max_per_acronym: int = 15,
) -> Iterator[ContextRow]:
    """Scan local Knesset shards for clean sentences mentioning each acronym.

    All shards are scanned once. Rather than checking every target acronym
    against every sentence (O(sentences x types), too slow once the target
    list covers the full ~550-type inventory), each sentence's own
    acronym-shaped tokens are extracted once via `ACRONYM_TOKEN_RE` and looked
    up in a normalised-acronym -> canonical-type index — a sentence usually
    carries zero or one such token, so this turns the inner loop from O(types)
    into O(tokens actually present).
    """
    acronyms = list(acronyms)
    found = {a: 0 for a in acronyms}
    # A sentence's raw token (mixed gershayim/ASCII quote, no niqqud
    # normalisation) maps to the canonical acronym spelling it should be
    # scored against — built once, not per sentence.
    by_normalized: dict[str, str] = {}
    for acronym in acronyms:
        for variant in hebrew_text.variants(acronym) or [acronym]:
            by_normalized[variant.replace("״", '"')] = acronym

    for shard_path in shard_paths:
        if all(found[a] >= max_per_acronym for a in acronyms):
            break
        try:
            sentences = list(iter_shard_sentences(shard_path))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("failed to read %s: %s", shard_path, exc)
            continue
        for text, protocol_name in sentences:
            tokens = {t.replace("״", '"') for t in ACRONYM_TOKEN_RE.findall(text)}
            candidates = {by_normalized[t] for t in tokens if t in by_normalized}
            for acronym in candidates:
                if found[acronym] >= max_per_acronym:
                    continue
                if not is_clean_sentence(text, acronym):
                    continue
                yield ContextRow(acronym, text, "knesset", protocol_name)
                found[acronym] += 1
    for acronym, n in found.items():
        LOG.info("%s: %d sentences from Knesset shards", acronym, n)
