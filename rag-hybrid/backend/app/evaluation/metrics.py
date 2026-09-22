"""DeepEval metric factory."""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from backend.app.evaluation.llm_judge import LLMJudge

_METRICS = {
    "faithfulness": FaithfulnessMetric,  # is the answer supported by the retrieved context?
    "answer_relevancy": AnswerRelevancyMetric,  # does the answer address the question?
    "contextual_relevancy": ContextualRelevancyMetric,  # is the retrieved context relevant to the question?
    "contextual_precision": ContextualPrecisionMetric,  # are the relevant chunks ranked first? (needs expected output)
    "contextual_recall": ContextualRecallMetric,  # does the context cover the expected answer? (needs expected output)
}


def available_metrics() -> list[str]:
    return list(_METRICS)


def build_metric(name: str, judge: LLMJudge, threshold: float) -> BaseMetric:
    try:
        cls = _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown metric '{name}'. Available: {', '.join(_METRICS)}") from exc
    return cls(threshold=threshold, model=judge, include_reason=True, async_mode=True)
