"""Tiny in-memory job tracker for long-running work (ingestion, evaluation)."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from backend.app.domain.models import utcnow

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed"]
ProgressFn = Callable[[str, float], None]


class Job(BaseModel):
    job_id: str
    kind: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0  # 0..1
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None


class JobManager:
    def __init__(self, max_jobs: int = 200):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max = max_jobs

    def create(self, kind: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > self._max:  # drop the oldest finished jobs
                for old_id in [j.job_id for j in sorted(self._jobs.values(), key=lambda j: j.created_at) if j.status in ("succeeded", "failed")][: len(self._jobs) - self._max]:
                    self._jobs.pop(old_id, None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in changes.items():
                    setattr(job, key, value)

    async def run(self, job_id: str, work: Callable[[ProgressFn], Awaitable[Any]]) -> None:
        """Execute `work(progress)`; store its (JSON-serialisable) result or the error."""
        self._update(job_id, status="running", stage="starting")

        def progress(stage: str, fraction: float) -> None:
            self._update(job_id, stage=stage, progress=max(0.0, min(1.0, fraction)))

        try:
            result = await work(progress)
            self._update(job_id, status="succeeded", stage="done", progress=1.0, result=result, finished_at=utcnow())
        except asyncio.CancelledError:
            self._update(job_id, status="failed", error="Job was cancelled.", finished_at=utcnow())
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the client through the job record
            logger.exception("Job %s failed", job_id)
            self._update(job_id, status="failed", error=str(exc) or exc.__class__.__name__, finished_at=utcnow())
