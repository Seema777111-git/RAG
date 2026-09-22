from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_container
from backend.app.domain.models import QueryRequest, RAGAnswer
from backend.app.services.container import AppContainer

router = APIRouter(tags=["query"])


@router.post("/query", response_model=RAGAnswer)
async def query(request: QueryRequest, c: AppContainer = Depends(get_container)) -> RAGAnswer:
    """Answer a question with hybrid (vector + knowledge graph) retrieval. Returns every intermediate artefact."""
    return await c.rag.answer(request)
