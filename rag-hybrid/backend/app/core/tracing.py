"""LangSmith integration: environment wiring, trace ids and evaluation feedback."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from langsmith.run_helpers import get_current_run_tree

from backend.app.core.config import Settings
from backend.app.domain.models import TraceInfo

logger = logging.getLogger(__name__)

_state = {"enabled": False, "project": ""}


def configure_langsmith(settings: Settings) -> bool:
    """Export LangSmith settings to the environment (the SDK reads them from there).

    Returns True when tracing is active. Also sets the legacy LANGCHAIN_* names because
    LangChain-based components (NeMo Guardrails' LLM adapter) may still look at those.
    """
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key.get_secret_value())
    flag = "true" if enabled else "false"
    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    elif settings.langsmith_tracing:
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; tracing disabled")
    _state.update(enabled=enabled, project=settings.langsmith_project)
    return enabled


def tracing_enabled() -> bool:
    return bool(_state["enabled"])


def current_trace() -> Optional[TraceInfo]:
    """Return the id of the run currently being traced, if tracing is enabled."""
    if not tracing_enabled():
        return None
    run = get_current_run_tree()
    if run is None:
        return None
    return TraceInfo(run_id=str(run.id), project=str(_state["project"]))


@lru_cache(maxsize=1)
def _client():
    from langsmith import Client

    return Client()


def log_feedback(run_id: str | None, key: str, score: float | None, comment: str | None = None) -> None:
    """Attach an evaluation score to a traced RAG run. Never raises."""
    if not (tracing_enabled() and run_id) or score is None:
        return
    try:
        _client().create_feedback(run_id=run_id, key=key, score=float(score), comment=(comment or "")[:1000])
    except Exception as exc:  # noqa: BLE001 - tracing must never break the request
        logger.warning("Could not log LangSmith feedback (%s): %s", key, exc)


def flush() -> None:
    if not tracing_enabled():
        return
    try:
        _client().flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith flush failed: %s", exc)
