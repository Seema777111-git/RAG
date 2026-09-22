"""FastAPI application factory.

Run with:  uvicorn backend.app.main:create_app --factory --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.errors import register_exception_handlers
from backend.app.api.router import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import setup_logging
from backend.app.core.tracing import configure_langsmith, flush
from backend.app.services.container import AppContainer, build_container

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


def create_app(settings: Optional[Settings] = None, container: Optional[AppContainer] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        configure_langsmith(settings)
        app.state.container = container or build_container(settings)
        if not settings.llm_configured:
            logger.warning("%s is not set: uploads work but knowledge-graph extraction, answers and evaluation will fail.", settings.api_key_env)
        logger.info("API ready | docs=%d chunks=%d graph=%d/%d", len(app.state.container.registry.list()), app.state.container.vector_store.count, app.state.container.kg_store.node_count, app.state.container.kg_store.edge_count)
        yield
        flush()

    app = FastAPI(
        title="Hybrid RAG API",
        version="1.0.0",
        description="PDF question answering with FAISS + knowledge-graph retrieval, NeMo Guardrails, DeepEval and LangSmith.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router, prefix=API_PREFIX)
    return app
