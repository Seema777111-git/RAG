"""The online RAG pipeline: input rail -> hybrid retrieval -> generation -> output rail."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from langsmith import traceable

from backend.app.core.config import Settings
from backend.app.core.tracing import current_trace
from backend.app.domain.models import GuardrailResult, QueryRequest, RAGAnswer
from backend.app.generation.generator import AnswerGenerator, extract_citations
from backend.app.generation.prompts import NO_CONTEXT_ANSWER
from backend.app.guardrails.engine import GuardrailsEngine
from backend.app.retrieval.fusion import evidence_strings
from backend.app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

BLOCKED_FALLBACK = "I'm sorry, I can't respond to that."


class RAGService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        guardrails: GuardrailsEngine,
    ):
        self._s = settings
        self._retriever = retriever
        self._generator = generator
        self._guardrails = guardrails

    @contextmanager
    def _timer(self, timings: dict[str, float], name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)

    def _models(self) -> dict[str, str]:
        return {
            "generation": self._generator.model,
            "guardrails": self._s.guardrail_model if self._guardrails.enabled else "disabled",
            "embeddings": self._s.embedding_model,
        }

    @traceable(name="rag_query", run_type="chain")
    async def answer(self, request: QueryRequest) -> RAGAnswer:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        result = RAGAnswer(question=request.question, answer="", mode=request.mode, models=self._models(), trace=current_trace())

        # 1. Input rail --------------------------------------------------------------------------------
        input_check = await self._guardrails.check_input(request.question)
        result.guardrails["input"] = input_check
        timings["input_guardrail_ms"] = input_check.latency_ms
        if not input_check.allowed:
            return self._blocked(result, input_check, timings, started)
        # A rail may rewrite the input (for example mask sensitive data); use the rewritten text downstream.
        question = input_check.message if input_check.status == "modified" and input_check.message else request.question
        if question != request.question:
            request = request.model_copy(update={"question": question})

        # 2. Hybrid retrieval (vector and graph in parallel) -----------------------------------------
        retrieval = await self._retriever.retrieve(request)
        result.vector_hits, result.kg, result.context = retrieval.vector_hits, retrieval.kg, retrieval.context
        timings.update(retrieval.timings_ms)

        if retrieval.context.is_empty:
            result.answer = NO_CONTEXT_ANSWER
            return self._finish(result, timings, started)

        # 3. Generation ------------------------------------------------------------------------------
        with self._timer(timings, "generation_ms"):
            answer = await self._generator.generate(request.question, retrieval.context.text)

        # 4. Output rails (safety + grounding against the evidence the model saw) --------------------
        output_check = await self._guardrails.check_output(request.question, answer, evidence_strings(retrieval.context))
        result.guardrails["output"] = output_check
        timings["output_guardrail_ms"] = output_check.latency_ms
        if not output_check.allowed:
            return self._blocked(result, output_check, timings, started)

        result.answer = answer
        result.citations = extract_citations(answer)
        return self._finish(result, timings, started)

    # ------------------------------------------------------------------ helpers
    def _blocked(self, result: RAGAnswer, check: GuardrailResult, timings: dict[str, float], started: float) -> RAGAnswer:
        result.blocked = True
        result.blocked_by = check.rail or f"{check.stage} guardrail"
        result.answer = check.message or BLOCKED_FALLBACK
        return self._finish(result, timings, started)

    @staticmethod
    def _finish(result: RAGAnswer, timings: dict[str, float], started: float) -> RAGAnswer:
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result.timings_ms = timings
        return result
