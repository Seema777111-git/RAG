"""Run the RAG pipeline over a set of questions and score each answer with DeepEval."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from statistics import mean
from typing import Optional

from deepeval.test_case import LLMTestCase

from backend.app.core.config import Settings
from backend.app.core.tracing import flush, log_feedback
from backend.app.domain.models import QueryRequest
from backend.app.evaluation.llm_judge import LLMJudge
from backend.app.evaluation.metrics import build_metric
from backend.app.evaluation.schemas import (
    NEEDS_EXPECTED,
    CaseResult,
    EvalReport,
    EvalRequest,
    MetricScore,
    MetricSummary,
)
from backend.app.retrieval.fusion import evidence_strings
from backend.app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


class EvaluationRunner:
    def __init__(self, settings: Settings, rag: RAGService, judge: LLMJudge):
        self._s = settings
        self._rag = rag
        self._judge = judge

    async def _score(self, metric_name: str, test_case: LLMTestCase, threshold: float) -> MetricScore:
        try:
            metric = build_metric(metric_name, self._judge, threshold)
            await metric.a_measure(test_case, _show_indicator=False)
            return MetricScore(metric=metric_name, score=round(float(metric.score), 4), passed=bool(metric.success), reason=metric.reason)
        except Exception as exc:  # noqa: BLE001 - one bad metric must not sink the run
            logger.warning("Metric %s failed: %s", metric_name, exc)
            return MetricScore(metric=metric_name, error=str(exc)[:500])

    async def _run_case(self, case, request: EvalRequest, threshold: float, semaphore: asyncio.Semaphore) -> CaseResult:
        async with semaphore:
            answer = await self._rag.answer(QueryRequest(question=case.question, mode=request.mode))
            contexts = evidence_strings(answer.context)
            run_id = answer.trace.run_id if answer.trace else None
            result = CaseResult(
                question=case.question,
                expected_output=case.expected_output,
                answer=answer.answer,
                blocked=answer.blocked,
                contexts=contexts,
                trace_run_id=run_id,
            )
            if answer.blocked or not contexts:
                reason = "Blocked by guardrails" if answer.blocked else "No context was retrieved"
                result.scores = [MetricScore(metric=m, error=reason) for m in request.metrics]
                return result

            test_case = LLMTestCase(
                input=case.question,
                actual_output=answer.answer,
                expected_output=case.expected_output,
                retrieval_context=contexts,
            )
            names = []
            for name in request.metrics:
                if name in NEEDS_EXPECTED and not case.expected_output:
                    result.scores.append(MetricScore(metric=name, error="Skipped: requires an expected answer"))
                else:
                    names.append(name)
            result.scores += await asyncio.gather(*(self._score(n, test_case, threshold) for n in names))
            for s in result.scores:  # link scores back to the LangSmith trace of this exact RAG run
                log_feedback(run_id, f"deepeval_{s.metric}", s.score, s.reason)
            return result

    @staticmethod
    def _summarise(cases: list[CaseResult], metrics: list[str]) -> list[MetricSummary]:
        out = []
        for name in metrics:
            scored = [s for c in cases for s in c.scores if s.metric == name and s.score is not None]
            out.append(
                MetricSummary(
                    metric=name,
                    evaluated=len(scored),
                    mean=round(mean(s.score for s in scored), 4) if scored else None,  # type: ignore[arg-type]
                    pass_rate=round(sum(bool(s.passed) for s in scored) / len(scored), 4) if scored else None,
                )
            )
        return out

    async def run(self, request: EvalRequest, progress: Optional[Progress] = None) -> EvalReport:
        progress = progress or (lambda *_: None)
        threshold = request.threshold if request.threshold is not None else self._s.eval_threshold
        semaphore = asyncio.Semaphore(self._s.eval_concurrency)
        total = len(request.cases)
        results: list[Optional[CaseResult]] = [None] * total
        done = 0

        async def worker(i: int) -> None:
            nonlocal done
            results[i] = await self._run_case(request.cases[i], request, threshold, semaphore)
            done += 1
            progress(f"Evaluated {done}/{total} questions", done / total)

        progress("Starting evaluation", 0.0)
        await asyncio.gather(*(worker(i) for i in range(total)))
        cases = [r for r in results if r is not None]

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report = EvalReport(
            name=f"eval-{stamp}",
            threshold=threshold,
            mode=request.mode,
            judge_model=self._judge.get_model_name(),
            cases=cases,
            summary=self._summarise(cases, request.metrics),
        )
        self._s.eval_dir.mkdir(parents=True, exist_ok=True)
        (self._s.eval_dir / f"{report.name}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        flush()
        return report

    # ------------------------------------------------------------------ saved reports
    def list_reports(self) -> list[dict]:
        rows = []
        for path in sorted(self._s.eval_dir.glob("eval-*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"name": data["name"], "created_at": data["created_at"], "cases": len(data["cases"]), "summary": data["summary"]})
            except Exception:  # noqa: BLE001 - ignore corrupt files
                continue
        return rows

    def load_report(self, name: str) -> Optional[EvalReport]:
        path = (self._s.eval_dir / f"{name}.json").resolve()
        if not path.is_file() or path.parent != self._s.eval_dir.resolve():
            return None  # also rejects path-traversal attempts such as "../secrets"
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
