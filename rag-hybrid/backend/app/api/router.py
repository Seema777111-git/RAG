from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes import documents, evaluate, health, jobs, query

api_router = APIRouter()
for module in (health, documents, jobs, query, evaluate):
    api_router.include_router(module.router)
