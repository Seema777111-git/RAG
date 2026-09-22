"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from backend.app.services.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
