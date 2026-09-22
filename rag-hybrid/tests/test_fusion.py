import numpy as np

from backend.app.domain.models import Chunk, KGResult, KGTriple, VectorHit
from backend.app.retrieval.fusion import evidence_strings, fuse
from backend.app.vector.faiss_store import FaissVectorStore


def chunk(i, text="passage text"):
    return Chunk(chunk_id=f"c{i}", doc_id="d", source="f.pdf", page=i, text=f"{text} {i}")


def triple(chunk_id, score=1.0):
    return KGTriple(subject="Acme", predicate="acquired", object="Widget", chunk_id=chunk_id, doc_id="d", score=score, hop=1)


def make_store(tmp_path, chunks):
    store = FaissVectorStore(tmp_path, "m")
    store.add(chunks, np.eye(len(chunks), dtype="float32"))
    return store


def test_agreement_boost_graph_only_chunks_and_refs(tmp_path):
    chunks = [chunk(i) for i in range(4)]
    store = make_store(tmp_path, chunks)
    hits = [VectorHit(rank=1, score=0.80, chunk=chunks[0]), VectorHit(rank=2, score=0.75, chunk=chunks[1])]
    kg = KGResult(triples=[triple("c1"), triple("c3")])
    bundle = fuse(hits, kg, store, kg_chunk_boost=0.1)

    origins = {c.chunk.chunk_id: c.origin for c in bundle.chunks}
    assert origins == {"c1": "both", "c0": "vector", "c3": "graph"}
    assert [c.ref for c in bundle.chunks] == ["C1", "C2", "C3"]
    assert bundle.chunks[0].chunk.chunk_id == "c1"  # 0.75 + 0.10 outranks 0.80
    assert [t.ref for t in bundle.triples] == ["K1", "K2"]
    assert "[K1] Acme | acquired | Widget (supported by C1)" in bundle.text
    assert "PASSAGES" in bundle.text and "KNOWLEDGE GRAPH FACTS" in bundle.text


def test_kg_only_chunk_limit(tmp_path):
    chunks = [chunk(i) for i in range(5)]
    store = make_store(tmp_path, chunks)
    kg = KGResult(triples=[triple(f"c{i}") for i in range(5)])
    bundle = fuse([], kg, store, kg_only_chunks=2)
    assert len(bundle.chunks) == 2


def test_budget_truncates_passages(tmp_path):
    chunks = [chunk(i, "x" * 400) for i in range(4)]
    store = make_store(tmp_path, chunks)
    hits = [VectorHit(rank=i + 1, score=1 - i * 0.1, chunk=c) for i, c in enumerate(chunks)]
    bundle = fuse(hits, KGResult(), store, max_chars=1000)
    assert bundle.truncated and 0 < sum(c.in_prompt for c in bundle.chunks) < 4
    assert sum(c.in_prompt for c in bundle.chunks) >= 1


def test_empty_inputs(tmp_path):
    bundle = fuse([], KGResult(), FaissVectorStore(tmp_path, "m"))
    assert bundle.is_empty and bundle.text == ""


def test_evidence_strings(tmp_path):
    chunks = [chunk(0)]
    store = make_store(tmp_path, chunks)
    bundle = fuse([VectorHit(rank=1, score=0.9, chunk=chunks[0])], KGResult(triples=[triple("c0")]), store)
    assert evidence_strings(bundle) == ["passage text 0", "Acme acquired Widget."]
