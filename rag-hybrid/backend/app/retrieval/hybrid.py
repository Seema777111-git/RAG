"""Hybrid retriever: FAISS and the knowledge graph run concurrently, then get fused."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from langsmith import traceable

from backend.app.core.config import Settings
from backend.app.domain.models import ContextBundle, KGResult, QueryRequest, VectorHit
from backend.app.graph.retriever import KnowledgeGraphRetriever
from backend.app.retrieval.fusion import fuse
from backend.app.vector.embeddings import Embedder
from backend.app.vector.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalOutput:
    vector_hits: list[VectorHit] = field(default_factory=list)
    kg: KGResult = field(default_factory=KGResult)
    context: ContextBundle = field(default_factory=ContextBundle)
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        kg_retriever: KnowledgeGraphRetriever,
    ):
        self._settings = settings
        self._embedder = embedder
        self._vectors = vector_store
        self._kg = kg_retriever

    # Sync workers: FAISS / NumPy / NetworkX are CPU-bound, so they run in threads.
    @traceable(name="vector_retrieval", run_type="retriever")
    def _vector_search(self, question: str, top_k: int) -> list[VectorHit]:
        query = self._embedder.embed_query(question)
        return [VectorHit(rank=i, score=round(score, 4), chunk=chunk) for i, (chunk, score) in enumerate(self._vectors.search(query, top_k), start=1)]

    @traceable(name="kg_retrieval", run_type="retriever")
    def _kg_search(self, question: str, hops: int, max_triples: int) -> KGResult:
        return self._kg.retrieve(question, hops=hops, max_triples=max_triples)

    async def _timed(self, name: str, timings: dict[str, float], func, *args):
        start = time.perf_counter()
        try:
            return await asyncio.to_thread(func, *args)
        finally:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)

    @traceable(name="hybrid_retrieval", run_type="chain")
    async def retrieve(self, request: QueryRequest) -> RetrievalOutput:
        s = self._settings
        top_k = request.top_k or s.vector_top_k
        hops = request.kg_hops or s.kg_hops
        max_triples = request.kg_max_triples or s.kg_max_triples
        out = RetrievalOutput()

        jobs = {}
        if request.mode in ("hybrid", "vector"):
            jobs["vector"] = self._timed("vector_ms", out.timings_ms, self._vector_search, request.question, top_k)
        if request.mode in ("hybrid", "graph"):
            jobs["kg"] = self._timed("kg_ms", out.timings_ms, self._kg_search, request.question, hops, max_triples)

        results = await asyncio.gather(*jobs.values())
        by_name = dict(zip(jobs, results))
        out.vector_hits = by_name.get("vector", [])
        out.kg = by_name.get("kg", self._kg.empty_result())

        start = time.perf_counter()
        out.context = fuse(
            out.vector_hits,
            out.kg,
            self._vectors,
            max_chars=s.context_max_chars,
            kg_chunk_boost=s.kg_chunk_boost,
            kg_only_chunks=s.kg_only_chunks,
        )
        out.timings_ms["fusion_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return out
