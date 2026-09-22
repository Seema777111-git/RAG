"""Entity linking: map a natural-language question onto nodes of the knowledge graph.

Three complementary signals, best score wins per node:
  exact     the entity name appears verbatim in the question              (score 1.0)
  partial   most of a multi-word entity's content words appear             (score = coverage)
  semantic  embedding similarity between the question and the entity name  (score = cosine)
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

from backend.app.domain.models import LinkedEntity
from backend.app.graph.kg_store import KnowledgeGraphStore
from backend.app.graph.text import content_tokens, tokenize
from backend.app.vector.embeddings import Embedder

logger = logging.getLogger(__name__)

_MIN_EXACT_CHARS = 3
_PARTIAL_MIN_COVERAGE = 0.6


class EntityLinker:
    def __init__(
        self,
        store: KnowledgeGraphStore,
        embedder: Optional[Embedder],
        *,
        max_entities: int = 8,
        semantic: bool = True,
        semantic_threshold: float = 0.55,
    ):
        self._store = store
        self._embedder = embedder
        self._max = max_entities
        self._semantic = semantic and embedder is not None
        self._threshold = semantic_threshold
        self._cache_lock = threading.Lock()
        self._cache_version = -1
        self._cache_nodes: list[str] = []
        self._cache_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ semantic cache
    def _node_matrix(self) -> tuple[list[str], Optional[np.ndarray]]:
        """Embed every node label once per graph version."""
        with self._cache_lock:
            if self._cache_version != self._store.version:
                nodes = self._store.node_ids()
                matrix = self._embedder.embed_documents(nodes) if nodes else None  # type: ignore[union-attr]
                self._cache_nodes, self._cache_matrix = nodes, matrix
                self._cache_version = self._store.version
            return self._cache_nodes, self._cache_matrix

    # ------------------------------------------------------------------ linking
    def link(self, question: str) -> list[LinkedEntity]:
        nodes = self._store.node_ids()
        if not nodes:
            return []
        padded = f" {' '.join(tokenize(question))} "
        q_tokens = set(content_tokens(question))
        found: dict[str, LinkedEntity] = {}

        def offer(node: str, score: float, method: str) -> None:
            current = found.get(node)
            if current is None or score > current.score:
                found[node] = LinkedEntity(node_id=node, label=self._store.label(node), score=round(score, 4), method=method)  # type: ignore[arg-type]

        for node in nodes:
            tokens = tokenize(node)
            if not tokens:
                continue
            if len(node) >= _MIN_EXACT_CHARS and re.search(rf"(?<!\w){re.escape(' '.join(tokens))}(?!\w)", padded):
                offer(node, 1.0, "exact")
                continue
            node_content = content_tokens(node)
            if len(node_content) >= 2:
                coverage = sum(t in q_tokens for t in node_content) / len(node_content)
                if coverage >= _PARTIAL_MIN_COVERAGE:
                    offer(node, 0.9 * coverage, "partial")

        if self._semantic:
            try:
                names, matrix = self._node_matrix()
                if matrix is not None and len(names):
                    sims = matrix @ self._embedder.embed_query(question)  # type: ignore[union-attr]
                    for idx in np.argsort(-sims)[: self._max]:
                        if float(sims[idx]) >= self._threshold:
                            offer(names[int(idx)], float(sims[idx]), "semantic")
            except Exception as exc:  # noqa: BLE001 - lexical linking still works
                logger.warning("Semantic entity linking skipped: %s", exc)

        graph = self._store.graph
        ranked = sorted(found.values(), key=lambda e: (-e.score, -graph.degree(e.node_id)))
        return ranked[: self._max]
