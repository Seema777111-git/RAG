from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_container
from backend.app.core.tracing import tracing_enabled
from backend.app.services.container import AppContainer

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/stats")
async def stats(c: AppContainer = Depends(get_container)) -> dict:
    s = c.settings
    return {
        "documents": len(c.registry.list()),
        "chunks": c.vector_store.count,
        "graph_nodes": c.kg_store.node_count,
        "graph_edges": c.kg_store.edge_count,
        "config": {
            "llm_provider": s.llm_provider,
            "llm_configured": s.llm_configured,
            "api_key_env": s.api_key_env,
            "guardrails_enabled": c.guardrails.enabled,
            "langsmith_tracing": tracing_enabled(),
            "langsmith_project": s.langsmith_project,
            "llm_model": s.llm_model,
            "extraction_model": s.extraction_model,
            "judge_model": s.judge_model,
            "embedding_model": s.embedding_model,
            "vector_top_k": s.vector_top_k,
            "kg_hops": s.kg_hops,
            "kg_max_triples": s.kg_max_triples,
            "eval_threshold": s.eval_threshold,
        },
    }


@router.post("/llm/check")
async def check_llm(c: AppContainer = Depends(get_container)) -> dict:
    """Make one tiny call to the configured LLM. Errors come back as a readable 502/503 message."""
    started = time.perf_counter()
    reply = await c.llm.complete(system="Reply with the single word: ok", user="ping", model=c.settings.llm_model, max_tokens=16)
    return {
        "ok": True,
        "provider": c.settings.llm_provider,
        "model": c.settings.llm_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "reply": reply[:40],
    }
