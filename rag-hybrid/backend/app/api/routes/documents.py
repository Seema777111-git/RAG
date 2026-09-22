from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from backend.app.api.deps import get_container
from backend.app.domain.models import DocumentInfo
from backend.app.ingestion.pdf_loader import file_sha256
from backend.app.services.container import AppContainer

router = APIRouter(prefix="/documents", tags=["documents"])

_CHUNK = 1 << 20


@router.get("", response_model=list[DocumentInfo])
async def list_documents(c: AppContainer = Depends(get_container)) -> list[DocumentInfo]:
    return c.registry.list()


@router.post("", status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    c: AppContainer = Depends(get_container),
) -> dict:
    """Store the PDF and start ingestion in the background. Poll `GET /jobs/{job_id}` for progress."""
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only .pdf files are supported.")

    limit = c.settings.max_upload_mb * 1024 * 1024
    tmp = c.settings.uploads_dir / f"upload-{uuid.uuid4().hex}.tmp"
    size = 0
    try:
        with tmp.open("wb") as out:
            while block := await file.read(_CHUNK):
                if size == 0 and not block.startswith(b"%PDF-"):
                    raise HTTPException(status_code=415, detail="The file is not a valid PDF.")
                size += len(block)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File is larger than {c.settings.max_upload_mb} MB.")
                out.write(block)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        stored = c.settings.uploads_dir / f"{await asyncio.to_thread(file_sha256, tmp)}.pdf"
        os.replace(tmp, stored)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    job = c.jobs.create("ingest")

    async def work(progress):
        result = await c.ingestion.ingest(stored, filename, progress)
        return result.model_dump(mode="json")

    background.add_task(c.jobs.run, job.job_id, work)
    return {"job_id": job.job_id, "filename": filename}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, c: AppContainer = Depends(get_container)) -> dict:
    if not await c.ingestion.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    (c.settings.uploads_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"deleted": doc_id}
