"""Mine usage-example sentences already carried inside Wiktionary entries."""
from __future__ import annotations

from typing import Iterator

from ..common import hebrew_text
from ..common.filters import ContextRow, is_clean_sentence
from .parser import extract_examples


def mine_wiktionary_sentences(entries: dict[str, str]) -> Iterator[ContextRow]:
    """Usage-example sentences carried in Wiktionary entries themselves.

    `entries` is `{page_title: wikitext}`. Unlike a search snippet, a `#:`/`#*`
    example line is written by a lexicographer to illustrate the word, so it
    needs no prose filtering — only the same gloss and single-acronym checks
    every other item is held to.
    """
    for title, wikitext in entries.items():
        acronym = hebrew_text.normalize_acronym(title)
        for example in extract_examples(wikitext):
            if is_clean_sentence(example, acronym):
                yield ContextRow(acronym, example, "wiktionary", title)
