"""Dependency wiring. One `AppContainer` is built at start-up and shared by all requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.core.config import Settings
from backend.app.core.env import sanitize_environment
from backend.app.core.llm import LLM, create_llm
from backend.app.evaluation.llm_judge import LLMJudge
from backend.app.evaluation.runner import EvaluationRunner
from backend.app.generation.generator import AnswerGenerator
from backend.app.graph.extractor import TripleExtractor
from backend.app.graph.kg_store import KnowledgeGraphStore
from backend.app.graph.linker import EntityLinker
from backend.app.graph.retriever import KnowledgeGraphRetriever
from backend.app.guardrails.engine import GuardrailsEngine
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.jobs import JobManager
from backend.app.services.rag_service import RAGService
from backend.app.services.registry import DocumentRegistry
from backend.app.vector.embeddings import Embedder, SentenceTransformerEmbedder
from backend.app.vector.faiss_store import FaissVectorStore


@dataclass
class AppContainer:
    settings: Settings
    llm: LLM
    embedder: Embedder
    vector_store: FaissVectorStore
    kg_store: KnowledgeGraphStore
    registry: DocumentRegistry
    guardrails: GuardrailsEngine
    ingestion: IngestionService
    rag: RAGService
    evaluator: EvaluationRunner
    jobs: JobManager


def build_container(
    settings: Settings,
    *,
    llm: Optional[LLM] = None,
    embedder: Optional[Embedder] = None,
    guardrails: Optional[GuardrailsEngine] = None,
) -> AppContainer:
    """Wire every component. Tests pass fakes for `llm`, `embedder` and `guardrails`."""
    sanitize_environment()  # libraries imported above may have copied empty .env lines into os.environ
    settings.ensure_dirs()
    llm = llm or create_llm(settings)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device, settings.embedding_batch_size)

    vector_store = FaissVectorStore(settings.vector_dir, settings.embedding_model)
    kg_store = KnowledgeGraphStore(settings.graph_path)
    registry = DocumentRegistry(settings.registry_path)

    linker = EntityLinker(
        kg_store,
        embedder,
        max_entities=settings.kg_max_entities,
        semantic=settings.kg_semantic_linking,
        semantic_threshold=settings.kg_semantic_threshold,
    )
    retriever = HybridRetriever(settings, embedder, vector_store, KnowledgeGraphRetriever(kg_store, linker))
    generator = AnswerGenerator(llm, settings.llm_model, settings.llm_max_tokens, settings.llm_temperature)
    guardrails = guardrails or GuardrailsEngine(settings)
    rag = RAGService(settings, retriever, generator, guardrails)

    extractor = TripleExtractor(llm, settings.extraction_model, settings.kg_max_triples_per_chunk, settings.kg_extraction_concurrency)
    ingestion = IngestionService(settings, embedder, vector_store, kg_store, extractor, registry)
    evaluator = EvaluationRunner(settings, rag, LLMJudge(llm, settings.judge_model))

    return AppContainer(
        settings=settings,
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        kg_store=kg_store,
        registry=registry,
        guardrails=guardrails,
        ingestion=ingestion,
        rag=rag,
        evaluator=evaluator,
        jobs=JobManager(),
    )
