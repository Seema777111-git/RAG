"""Graph retrieval: link entities in the question, then expand a k-hop neighbourhood around them."""

from __future__ import annotations

from collections import deque

from backend.app.domain.models import GraphEdge, GraphNode, KGResult, KGTriple
from backend.app.graph.kg_store import KnowledgeGraphStore
from backend.app.graph.linker import EntityLinker

_MAX_EDGES_PER_NODE = 60  # keeps hub nodes from flooding the result


class KnowledgeGraphRetriever:
    def __init__(self, store: KnowledgeGraphStore, linker: EntityLinker):
        self._store = store
        self._linker = linker

    def empty_result(self) -> KGResult:
        return KGResult(graph_nodes_total=self._store.node_count, graph_edges_total=self._store.edge_count)

    def retrieve(self, question: str, hops: int = 1, max_triples: int = 15) -> KGResult:
        store = self._store
        empty = self.empty_result()
        entities = self._linker.link(question)
        if not entities:
            return empty

        with store.lock:
            seed_score = {e.node_id: e.score for e in entities}
            # BFS. `origin[node]` = best seed score that reaches the node, used to weight edges.
            depth = {n: 0 for n in seed_score}
            origin = dict(seed_score)
            queue = deque(seed_score)
            candidates: dict[tuple[str, str, str], KGTriple] = {}

            while queue:
                node = queue.popleft()
                if depth[node] >= hops:
                    continue
                edges = sorted(store.incident_edges(node), key=lambda e: -len(e[3]["chunk_ids"]))[:_MAX_EDGES_PER_NODE]
                for u, v, key, data in edges:
                    hop = depth[node] + 1
                    other = v if u == node else u
                    score = origin[node] / hop + 0.05 * min(len(data["chunk_ids"]), 5)
                    if u in seed_score and v in seed_score:
                        score += 0.25  # connects two entities the user asked about
                    edge_id = (u, v, key)
                    triple = KGTriple(
                        subject=store.label(u),
                        predicate=data["predicate"],
                        object=store.label(v),
                        chunk_id=data["chunk_ids"][0],
                        doc_id=data["doc_id"],
                        score=round(score, 4),
                        hop=hop,
                    )
                    if edge_id not in candidates or triple.score > candidates[edge_id].score:
                        candidates[edge_id] = triple
                    if other not in depth:
                        depth[other] = hop
                        origin[other] = origin[node]
                        queue.append(other)

            ranked = sorted(candidates.items(), key=lambda kv: (-kv[1].score, kv[1].hop))[:max_triples]
            triples = [t for _, t in ranked]
            node_ids: dict[str, GraphNode] = {n: GraphNode(id=n, label=store.label(n), is_seed=True) for n in seed_score}
            edges_out: list[GraphEdge] = []
            for (u, v, _), t in ranked:
                for n in (u, v):
                    node_ids.setdefault(n, GraphNode(id=n, label=store.label(n), is_seed=False))
                edges_out.append(GraphEdge(source=u, target=v, label=t.predicate))

        return KGResult(
            entities=entities,
            triples=triples,
            nodes=list(node_ids.values()),
            edges=edges_out,
            graph_nodes_total=store.node_count,
            graph_edges_total=store.edge_count,
        )
