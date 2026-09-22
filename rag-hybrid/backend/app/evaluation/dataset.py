"""Synthetic question/answer generation so evaluation can start without a hand-written golden set."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Optional

from backend.app.core.llm import LLM
from backend.app.evaluation.schemas import EvalCase
from backend.app.vector.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)

_SYSTEM = """You write evaluation questions for a document Q&A system.
Given one passage, write ONE question that the passage answers fully and specifically, plus the correct answer taken only from the passage.
Avoid questions that need other context ("according to the text", "the author"). Respond with JSON only:
{"question": "...", "answer": "..."}"""


async def generate_cases(
    llm: LLM,
    model: str,
    store: FaissVectorStore,
    count: int,
    doc_id: Optional[str] = None,
    seed: int = 7,
) -> list[EvalCase]:
    candidates = store.iter_chunks(doc_id)
    chunks = [c for c in candidates if len(c.text) > 200] or [c for c in candidates if len(c.text) > 80]  # prefer substantial passages
    if not chunks:
        return []
    random.Random(seed).shuffle(chunks)
    cases: list[EvalCase] = []
    for chunk in chunks[: count * 2]:  # oversample: some generations will be unusable
        if len(cases) >= count:
            break
        try:
            raw = await llm.complete(system=_SYSTEM, user=f"Passage:\n{chunk.text}", model=model, max_tokens=400, temperature=0.3)
            payload = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))  # type: ignore[union-attr]
            q, a = str(payload["question"]).strip(), str(payload["answer"]).strip()
            if q and a:
                cases.append(EvalCase(question=q, expected_output=a))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unusable generated case: %s", exc)
    return cases
