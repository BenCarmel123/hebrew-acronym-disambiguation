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
        --acronyms data/mined/knesset/rabbinic_types.txt \\
        --out data/mined/knesset/knesset_mined.csv
"""
from __future__ import annotations

import bz2
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

import requests

from ..common.filters import ContextRow, is_clean_sentence, mentions_acronym

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
    """Filenames of every .jsonl.bz2 shard for one config, via the HF tree API."""
    resp = requests.get(f"{API_URL}/{shard_paths(config)}", timeout=30)
    resp.raise_for_status()
    return [f["path"] for f in resp.json() if f["type"] == "file"]


def download_shards(
    config: str = "plenary_protocols", *, n: int = 30, out_dir: str = "data/raw/knesset_shards"
) -> list[Path]:
    """Download the first `n` shards for a config into `out_dir`. Skips existing files.

    The corpus is a fixed public dump, not something mining code should
    re-fetch on every run — shards already on disk are left alone, so a
    second call with a larger `n` only fetches what is new.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    remote_paths = list_remote_shards(config)[:n]
    local_paths = []
    for remote_path in remote_paths:
        local_path = out / Path(remote_path).name
        local_paths.append(local_path)
        if local_path.exists():
            continue
        LOG.info("downloading %s", remote_path)
        resp = requests.get(f"{RESOLVE_URL}/{remote_path}", timeout=120)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
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

    All shards are scanned once, checking every acronym against every
    sentence — cheaper than one pass per acronym over the same files, and the
    corpus is small enough (shards are ~0.5MB each) that this is fast even
    for dozens of shards.
    """
    acronyms = list(acronyms)
    found = {a: 0 for a in acronyms}
    for shard_path in shard_paths:
        if all(found[a] >= max_per_acronym for a in acronyms):
            break
        try:
            sentences = list(iter_shard_sentences(shard_path))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("failed to read %s: %s", shard_path, exc)
            continue
        for text, protocol_name in sentences:
            for acronym in acronyms:
                if found[acronym] >= max_per_acronym:
                    continue
                if not mentions_acronym(text, acronym):
                    continue
                if not is_clean_sentence(text, acronym):
                    continue
                yield ContextRow(acronym, text, "knesset", protocol_name)
                found[acronym] += 1
    for acronym, n in found.items():
        LOG.info("%s: %d sentences from Knesset shards", acronym, n)
