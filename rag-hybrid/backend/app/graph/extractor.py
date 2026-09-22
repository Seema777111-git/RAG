"""LLM-based knowledge-graph triple extraction (one call per chunk, bounded concurrency)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Optional

from backend.app.core.llm import LLM
from backend.app.domain.models import Chunk, Triple
from backend.app.graph.text import normalize_entity

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You extract knowledge-graph triples from a passage of a document.

Rules:
- Only extract facts that the passage states explicitly. Never add outside knowledge.
- subject and object are specific entities or concepts (people, organisations, products, places, dates, amounts, technical terms). Never use pronouns.
- Use the same surface form for the same entity every time (for example always "Acme Corp", never "the company").
- predicate is a short lowercase verb phrase of 1 to 4 words (for example "acquired", "is located in", "reported revenue of").
- Prefer fewer, high-quality triples. Return at most {max_triples}.
- Respond with JSON only, no prose and no code fences, in exactly this shape:
{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
- If the passage contains no extractable facts, respond with {{"triples": []}}."""

MAX_ENTITY_CHARS = 100
MAX_PREDICATE_CHARS = 60

ProgressCallback = Callable[[int, int], Awaitable[None] | None]


def parse_triples_json(raw: str) -> list[dict]:
    """Tolerantly pull the JSON object out of an LLM reply."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = payload.get("triples", []) if isinstance(payload, dict) else []
    return [i for i in items if isinstance(i, dict)]


class TripleExtractor:
    def __init__(self, llm: LLM, model: str, max_triples_per_chunk: int = 12, concurrency: int = 4):
        self._llm = llm
        self._model = model
        self._max = max_triples_per_chunk
        self._concurrency = concurrency
        self.last_error: Optional[str] = None  # first failure of the most recent extract_all run

    async def extract_chunk(self, chunk: Chunk) -> list[Triple]:
        raw = await self._llm.complete(
            system=EXTRACTION_SYSTEM.format(max_triples=self._max),
            user=f"Passage:\n{chunk.text}",
            model=self._model,
            max_tokens=1200,
            temperature=0.0,
        )
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for item in parse_triples_json(raw):
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not (subject and predicate and obj):
                continue
            if max(len(subject), len(obj)) > MAX_ENTITY_CHARS or len(predicate) > MAX_PREDICATE_CHARS:
                continue
            key = (normalize_entity(subject), predicate.lower(), normalize_entity(obj))
            if not key[0] or not key[2] or key[0] == key[2] or key in seen:
                continue
            seen.add(key)
            triples.append(
                Triple(subject=subject, predicate=predicate, object=obj, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
            )
            if len(triples) >= self._max:
                break
        return triples

    async def extract_all(
        self, chunks: list[Chunk], on_progress: Optional[ProgressCallback] = None
    ) -> tuple[list[Triple], int]:
        """Extract from every chunk. Returns (triples, failed_chunk_count); a bad chunk never aborts the run."""
        self.last_error = None
        semaphore = asyncio.Semaphore(self._concurrency)
        done = 0
        failures = 0
        results: list[list[Triple]] = [[] for _ in chunks]

        async def worker(position: int, chunk: Chunk) -> None:
            nonlocal done, failures
            async with semaphore:
                try:
                    results[position] = await self.extract_chunk(chunk)
                except Exception as exc:  # noqa: BLE001 - keep going, report the count
                    failures += 1
                    self.last_error = self.last_error or str(exc)
                    logger.warning("Triple extraction failed for %s: %s", chunk.chunk_id, exc)
                done += 1
                if on_progress:
                    maybe = on_progress(done, len(chunks))
                    if asyncio.iscoroutine(maybe):
                        await maybe

        await asyncio.gather(*(worker(i, c) for i, c in enumerate(chunks)))
        return [t for group in results for t in group], failures
