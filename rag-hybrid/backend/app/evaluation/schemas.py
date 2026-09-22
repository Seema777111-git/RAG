"""Evaluation request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.app.domain.models import QueryMode, utcnow

ALL_METRICS = ["faithfulness", "answer_relevancy", "contextual_relevancy", "contextual_precision", "contextual_recall"]
# These two need a reference ("expected") answer to compare against.
NEEDS_EXPECTED = {"contextual_precision", "contextual_recall"}


class EvalCase(BaseModel):
    question: str = Field(min_length=1)
    expected_output: Optional[str] = None


class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=200)
    metrics: list[str] = Field(default_factory=lambda: ["faithfulness", "answer_relevancy", "contextual_relevancy"])
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mode: QueryMode = "hybrid"


class MetricScore(BaseModel):
    metric: str
    score: Optional[float] = None
    passed: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class CaseResult(BaseModel):
    question: str
    expected_output: Optional[str] = None
    answer: str = ""
    blocked: bool = False
    contexts: list[str] = Field(default_factory=list)
    scores: list[MetricScore] = Field(default_factory=list)
    trace_run_id: Optional[str] = None


class MetricSummary(BaseModel):
    metric: str
    mean: Optional[float] = None
    pass_rate: Optional[float] = None
    evaluated: int = 0


class EvalReport(BaseModel):
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    threshold: float
    mode: QueryMode
    judge_model: str
    cases: list[CaseResult]
    summary: list[MetricSummary]


class GenerateRequest(BaseModel):
    num_questions: int = Field(default=5, ge=1, le=30)
    doc_id: Optional[str] = None
