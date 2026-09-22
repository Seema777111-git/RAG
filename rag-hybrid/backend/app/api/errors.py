"""Map domain exceptions to HTTP responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.core.errors import IndexMismatchError, LLMConfigError, LLMError, NotFoundError, PdfError, RAGError

logger = logging.getLogger(__name__)

_STATUS = {
    LLMConfigError: 503,
    LLMError: 502,
    IndexMismatchError: 409,
    PdfError: 422,
    NotFoundError: 404,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError) -> JSONResponse:
        status = next((code for cls, code in _STATUS.items() if isinstance(exc, cls)), 400)
        return JSONResponse(status_code=status, content={"detail": str(exc), "type": exc.__class__.__name__})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error.", "type": exc.__class__.__name__})
