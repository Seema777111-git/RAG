from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from backend.app.api.deps import get_container
from backend.app.evaluation.dataset import generate_cases
from backend.app.evaluation.metrics import available_metrics
from backend.app.evaluation.schemas import EvalCase, EvalReport, EvalRequest, GenerateRequest
from backend.app.services.container import AppContainer

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.get("/metrics")
async def list_metrics() -> dict:
    return {"metrics": available_metrics()}


@router.post("", status_code=202)
async def start_evaluation(request: EvalRequest, background: BackgroundTasks, c: AppContainer = Depends(get_container)) -> dict:
    """Run DeepEval over the given questions as a background job. Poll `GET /jobs/{job_id}`."""
    unknown = [m for m in request.metrics if m not in available_metrics()]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown metrics: {', '.join(unknown)}")

    job = c.jobs.create("evaluate")

    async def work(progress):
        report = await c.evaluator.run(request, progress)
        return report.model_dump(mode="json")

    background.add_task(c.jobs.run, job.job_id, work)
    return {"job_id": job.job_id}


@router.post("/generate", response_model=list[EvalCase])
async def generate_test_cases(request: GenerateRequest, c: AppContainer = Depends(get_container)) -> list[EvalCase]:
    cases = await generate_cases(c.llm, c.settings.llm_model, c.vector_store, request.num_questions, request.doc_id)
    if not cases:
        raise HTTPException(status_code=409, detail="No suitable passages found. Ingest a document first.")
    return cases


@router.get("/reports")
async def list_reports(c: AppContainer = Depends(get_container)) -> list[dict]:
    return c.evaluator.list_reports()


@router.get("/reports/{name}", response_model=EvalReport)
async def get_report(name: str, c: AppContainer = Depends(get_container)) -> EvalReport:
    report = c.evaluator.load_report(name)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report
