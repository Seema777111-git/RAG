"""Ingestion orchestration: one PDF in, two indexes out (FAISS vectors + NetworkX knowledge graph)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
from langsmith import traceable
from pydantic import BaseModel

from backend.app.core.config import Settings
from backend.app.domain.models import Chunk, DocumentInfo, Triple
from backend.app.graph.extractor import TripleExtractor
from backend.app.graph.kg_store import KnowledgeGraphStore
from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.pdf_loader import LoadedPdf, load_pdf
from backend.app.services.registry import DocumentRegistry
from backend.app.vector.embeddings import Embedder
from backend.app.vector.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


class PipelineStatus(BaseModel):
    status: str  # ok | failed | skipped
    count: int = 0
    seconds: float = 0.0
    detail: Optional[str] = None


class IngestResult(BaseModel):
    document: DocumentInfo
    replaced_existing: bool = False
    vector: PipelineStatus
    graph: PipelineStatus
    warnings: list[str] = []


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        kg_store: KnowledgeGraphStore,
        extractor: TripleExtractor,
        registry: DocumentRegistry,
    ):
        self._s = settings
        self._embedder = embedder
        self._vectors = vector_store
        self._kg = kg_store
        self._extractor = extractor
        self._registry = registry
        self._write_lock = asyncio.Lock()  # one ingestion at a time keeps both indexes consistent

    # ------------------------------------------------------------------ pipelines
    async def _vector_pipeline(self, chunks: list[Chunk], progress: Progress) -> np.ndarray:
        progress("Embedding chunks", 0.35)
        return await asyncio.to_thread(self._embedder.embed_documents, [c.text for c in chunks])

    async def _graph_pipeline(self, chunks: list[Chunk], progress: Progress) -> tuple[list[Triple], int]:
        def on_progress(done: int, total: int) -> None:
            progress(f"Extracting knowledge graph ({done}/{total} chunks)", 0.35 + 0.5 * done / max(total, 1))

        return await self._extractor.extract_all(chunks, on_progress)

    # ------------------------------------------------------------------ public API
    @traceable(name="ingest_pdf", run_type="chain")
    async def ingest(self, path: Path, filename: str, progress: Progress = lambda *_: None) -> IngestResult:
        progress("Reading PDF", 0.05)
        pdf: LoadedPdf = await asyncio.to_thread(load_pdf, path, filename)

        progress("Chunking", 0.15)
        chunks = chunk_document(pdf, self._s.chunk_size, self._s.chunk_overlap, self._s.min_chunk_chars)
        if not chunks:
            raise ValueError("The PDF produced no usable text chunks.")

        warnings: list[str] = []
        async with self._write_lock:
            replaced = self._registry.get(pdf.doc_id) is not None
            if replaced:
                progress("Replacing previous version", 0.2)
                self._remove_from_indexes(pdf.doc_id)

            t_vec = t_kg = time.perf_counter()
            vec_task = asyncio.create_task(self._vector_pipeline(chunks, progress))
            kg_task = (
                asyncio.create_task(self._graph_pipeline(chunks, progress)) if self._s.kg_extraction_enabled else None
            )

            # Vector pipeline
            try:
                vectors = await vec_task
                await asyncio.to_thread(self._vectors.add, chunks, vectors)
                vector_status = PipelineStatus(status="ok", count=len(chunks), seconds=round(time.perf_counter() - t_vec, 2))
            except Exception as exc:  # noqa: BLE001 - without vectors the document is unusable: abort the whole ingest
                logger.exception("Vector pipeline failed")
                if kg_task:
                    kg_task.cancel()
                raise RuntimeError(f"Vector pipeline failed: {exc}") from exc

            # Graph pipeline (best effort: a failure degrades to vector-only, it does not lose the document)
            graph_status = PipelineStatus(status="skipped", detail="KG extraction disabled (KG_EXTRACTION_ENABLED=false)")
            stored = 0
            if kg_task:
                try:
                    triples, failed_chunks = await kg_task
                    stored = self._kg.add_triples(triples)
                    detail = f"{failed_chunks} chunk(s) failed extraction" if failed_chunks else None
                    if failed_chunks:
                        reason = f" First error: {self._extractor.last_error}" if self._extractor.last_error else ""
                        warnings.append(f"Knowledge-graph extraction failed for {failed_chunks} of {len(chunks)} chunks.{reason}")
                    graph_status = PipelineStatus(status="ok", count=stored, seconds=round(time.perf_counter() - t_kg, 2), detail=detail)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Graph pipeline failed")
                    graph_status = PipelineStatus(status="failed", detail=str(exc), seconds=round(time.perf_counter() - t_kg, 2))
                    warnings.append(f"Knowledge graph was not built: {exc}. Vector search still works for this document.")

            progress("Saving indexes", 0.92)
            await asyncio.to_thread(self._vectors.save)
            await asyncio.to_thread(self._kg.save)

            document = DocumentInfo(
                doc_id=pdf.doc_id,
                filename=pdf.filename,
                pages=pdf.page_count,
                chunks=len(chunks),
                triples=stored,
                size_bytes=pdf.size_bytes,
            )
            self._registry.upsert(document)

        progress("Done", 1.0)
        return IngestResult(document=document, replaced_existing=replaced, vector=vector_status, graph=graph_status, warnings=warnings)

    def _remove_from_indexes(self, doc_id: str) -> None:
        self._vectors.delete_document(doc_id)
        self._kg.remove_document(doc_id)

    async def delete(self, doc_id: str) -> bool:
        async with self._write_lock:
            if self._registry.get(doc_id) is None:
                return False
            self._remove_from_indexes(doc_id)
            await asyncio.to_thread(self._vectors.save)
            await asyncio.to_thread(self._kg.save)
            self._registry.remove(doc_id)
            return True
