"""Fuse vector hits and knowledge-graph triples into one numbered, size-bounded prompt context."""

from __future__ import annotations

from typing import Optional

from backend.app.domain.models import ContextBundle, ContextChunk, KGResult, KGTriple, VectorHit
from backend.app.vector.faiss_store import FaissVectorStore


def _format_chunk(item: ContextChunk) -> str:
    c = item.chunk
    return f"[{item.ref}] ({c.source}, page {c.page})\n{c.text}"


def _format_triple(t: KGTriple, chunk_ref: Optional[str]) -> str:
    support = f" (supported by {chunk_ref})" if chunk_ref else ""
    return f"[{t.ref}] {t.subject} | {t.predicate} | {t.object}{support}"


def fuse(
    vector_hits: list[VectorHit],
    kg: KGResult,
    vector_store: FaissVectorStore,
    *,
    max_chars: int = 12000,
    kg_chunk_boost: float = 0.10,
    kg_only_chunks: int = 3,
) -> ContextBundle:
    """Merge both retrievers.

    - A chunk found by vector search AND cited by a KG triple is marked `both` and gets a score boost.
    - A chunk cited only by a KG triple is pulled in as `graph` evidence (bounded by `kg_only_chunks`),
      so every graph fact the model sees can be traced back to real source text.
    - Passages get 75% of the character budget; graph facts get the rest.
    """
    kg_chunk_ids = [t.chunk_id for t in kg.triples]  # already ordered by triple score
    kg_chunk_set = set(kg_chunk_ids)
    seen: set[str] = set()
    scored: list[tuple[float, str, ContextChunk]] = []

    for hit in vector_hits:
        both = hit.chunk.chunk_id in kg_chunk_set
        score = hit.score + (kg_chunk_boost if both else 0.0)
        scored.append((score, hit.chunk.chunk_id, ContextChunk(ref="", chunk=hit.chunk, score=round(score, 4), origin="both" if both else "vector")))
        seen.add(hit.chunk.chunk_id)

    extra = 0
    for chunk_id in kg_chunk_ids:
        if chunk_id in seen or extra >= kg_only_chunks:
            continue
        chunk = vector_store.get_chunk(chunk_id)
        if chunk is None:
            continue
        seen.add(chunk_id)
        extra += 1
        scored.append((-1.0 - extra, chunk_id, ContextChunk(ref="", chunk=chunk, score=0.0, origin="graph")))

    scored.sort(key=lambda row: -row[0])

    passage_budget = int(max_chars * 0.75)
    used = 0
    truncated = False
    chunks: list[ContextChunk] = []
    for n, (_, _, item) in enumerate(scored, start=1):
        item.ref = f"C{n}"
        cost = len(item.chunk.text) + len(item.chunk.source) + 32
        if used + cost > passage_budget and any(c.in_prompt for c in chunks):
            item.in_prompt = False
            truncated = True
        else:
            used += cost
        chunks.append(item)

    ref_of = {c.chunk.chunk_id: c.ref for c in chunks if c.in_prompt}
    triple_budget = max_chars - used
    triples: list[KGTriple] = []
    triple_lines: list[str] = []
    spent = 0
    for n, t in enumerate(kg.triples, start=1):
        t = t.model_copy(update={"ref": f"K{n}"})
        line = _format_triple(t, ref_of.get(t.chunk_id))
        if spent + len(line) > triple_budget:
            truncated = True
            break
        spent += len(line) + 1
        triples.append(t)
        triple_lines.append(line)

    sections: list[str] = []
    passages = [_format_chunk(c) for c in chunks if c.in_prompt]
    if passages:
        sections.append("PASSAGES\n" + "\n\n".join(passages))
    if triple_lines:
        sections.append("KNOWLEDGE GRAPH FACTS (subject | relation | object)\n" + "\n".join(triple_lines))

    return ContextBundle(text="\n\n".join(sections), chunks=chunks, triples=triples, truncated=truncated)


def evidence_strings(bundle: ContextBundle) -> list[str]:
    """Plain-text evidence list used by the grounding rail and by DeepEval's retrieval_context."""
    items = [c.chunk.text for c in bundle.chunks if c.in_prompt]
    items += [f"{t.subject} {t.predicate} {t.object}." for t in bundle.triples]
    return items
