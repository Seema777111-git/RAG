"""NetworkX knowledge graph with GraphML persistence and per-document removal."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import networkx as nx

from backend.app.domain.models import Triple
from backend.app.graph.text import normalize_entity, normalize_predicate

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """A `MultiDiGraph` where nodes are normalised entity names.

    Edge key = "<doc_id>::<predicate>", so one edge exists per (subject, predicate, object, document)
    and removing a document is a simple filter. Every edge keeps the chunk ids that support it,
    which is what lets the UI link a graph fact back to its source passage.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._graph = nx.MultiDiGraph()
        self.version = 0  # bumped on every mutation; caches key off it
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = nx.read_graphml(self._path, node_type=str, edge_key_type=str, force_multigraph=True)
        graph = nx.MultiDiGraph()
        for node, data in raw.nodes(data=True):
            graph.add_node(node, label=data.get("label", node), doc_ids=json.loads(data.get("doc_ids", "[]")))
        for u, v, key, data in raw.edges(keys=True, data=True):
            graph.add_edge(
                u,
                v,
                key=key,
                predicate=data.get("predicate", ""),
                doc_id=data.get("doc_id", ""),
                chunk_ids=json.loads(data.get("chunk_ids", "[]")),
            )
        self._graph = graph
        self.version += 1
        logger.info("Loaded knowledge graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    def save(self) -> None:
        with self._lock:
            out = nx.MultiDiGraph()
            for node, data in self._graph.nodes(data=True):
                out.add_node(node, label=data["label"], doc_ids=json.dumps(data["doc_ids"]))
            for u, v, key, data in self._graph.edges(keys=True, data=True):
                out.add_edge(
                    u, v, key=key, predicate=data["predicate"], doc_id=data["doc_id"], chunk_ids=json.dumps(data["chunk_ids"])
                )
            tmp = self._path.with_suffix(".graphml.tmp")
            nx.write_graphml(out, tmp)
            os.replace(tmp, self._path)

    # ------------------------------------------------------------------ mutation
    def add_triples(self, triples: list[Triple]) -> int:
        """Add triples, merging duplicates. Returns the number of triples actually stored."""
        stored = 0
        with self._lock:
            for t in triples:
                s, o = normalize_entity(t.subject), normalize_entity(t.object)
                p = normalize_predicate(t.predicate)
                if not (s and o and p) or s == o:
                    continue
                for node_id, label in ((s, t.subject.strip()), (o, t.object.strip())):
                    if node_id not in self._graph:
                        self._graph.add_node(node_id, label=label, doc_ids=[t.doc_id])
                    elif t.doc_id not in self._graph.nodes[node_id]["doc_ids"]:
                        self._graph.nodes[node_id]["doc_ids"].append(t.doc_id)
                key = f"{t.doc_id}::{p}"
                if self._graph.has_edge(s, o, key):
                    chunk_ids = self._graph.edges[s, o, key]["chunk_ids"]
                    if t.chunk_id not in chunk_ids:
                        chunk_ids.append(t.chunk_id)
                else:
                    self._graph.add_edge(s, o, key=key, predicate=t.predicate.strip().lower(), doc_id=t.doc_id, chunk_ids=[t.chunk_id])
                stored += 1
            if stored:
                self.version += 1
        return stored

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            doomed = [(u, v, k) for u, v, k, d in self._graph.edges(keys=True, data=True) if d["doc_id"] == doc_id]
            self._graph.remove_edges_from(doomed)
            for node in list(self._graph.nodes):
                data = self._graph.nodes[node]
                if doc_id in data["doc_ids"]:
                    data["doc_ids"].remove(doc_id)
                if self._graph.degree(node) == 0:
                    self._graph.remove_node(node)
            if doomed:
                self.version += 1
            return len(doomed)

    # ------------------------------------------------------------------ read access
    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def label(self, node_id: str) -> str:
        return self._graph.nodes[node_id]["label"]

    def node_ids(self) -> list[str]:
        with self._lock:
            return list(self._graph.nodes)

    def incident_edges(self, node_id: str) -> Iterator[tuple[str, str, str, dict]]:
        """All edges touching a node, in either direction: (u, v, key, data)."""
        yield from self._graph.out_edges(node_id, keys=True, data=True)
        yield from self._graph.in_edges(node_id, keys=True, data=True)

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def edge_count_for(self, doc_id: str) -> int:
        with self._lock:
            return sum(1 for _, _, d in self._graph.edges(data=True) if d["doc_id"] == doc_id)
