from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_container
from backend.app.services.container import AppContainer
from backend.app.services.jobs import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str, c: AppContainer = Depends(get_container)) -> Job:
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
