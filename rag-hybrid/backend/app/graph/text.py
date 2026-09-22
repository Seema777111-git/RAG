"""Entity-name normalisation shared by the extractor, the graph store and the entity linker."""

from __future__ import annotations

import re
import unicodedata

_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been being what which who whom whose "
    "when where why how does do did has have had this that these those it its as about into than then there "
    "their they them he she his her we our you your i me my tell say says said explain describe list give "
    "between not can could should would will may might".split()
)


def normalize_entity(text: str) -> str:
    """Canonical key for an entity: NFKC, lowercase, no edge punctuation, no leading article."""
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = _EDGE_PUNCT.sub("", text)
    text = _LEADING_ARTICLE.sub("", text)
    return " ".join(text.lower().split())


def normalize_predicate(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).lower()).split() if t]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]
