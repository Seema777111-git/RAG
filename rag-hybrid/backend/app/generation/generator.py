"""Answer generation from the fused context."""

from __future__ import annotations

import re

from langsmith import traceable

from backend.app.core.llm import LLM
from backend.app.generation.prompts import ANSWER_SYSTEM, ANSWER_USER

_CITATION = re.compile(r"\[([CK]\d+)\]")


def extract_citations(answer: str) -> list[str]:
    """Unique citation labels in order of first appearance."""
    seen: dict[str, None] = {}
    for label in _CITATION.findall(answer):
        seen.setdefault(label, None)
    return list(seen)


class AnswerGenerator:
    def __init__(self, llm: LLM, model: str, max_tokens: int, temperature: float):
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    @traceable(name="generate_answer", run_type="chain")
    async def generate(self, question: str, context: str) -> str:
        return await self._llm.complete(
            system=ANSWER_SYSTEM,
            user=ANSWER_USER.format(context=context, question=question),
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
