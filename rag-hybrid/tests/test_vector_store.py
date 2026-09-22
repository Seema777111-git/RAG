import numpy as np
import pytest

from backend.app.core.errors import IndexMismatchError
from backend.app.domain.models import Chunk
from backend.app.vector.faiss_store import FaissVectorStore


def _chunks(doc, n):
    return [Chunk(chunk_id=f"{doc}-{i}", doc_id=doc, source=f"{doc}.pdf", page=1, text=f"text {i}") for i in range(n)]


def _unit(n, dim=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_returns_best_match_first(tmp_path):
    store = FaissVectorStore(tmp_path, "m")
    vecs = _unit(5)
    store.add(_chunks("a", 5), vecs)
    hits = store.search(vecs[3], 3)
    assert hits[0][0].chunk_id == "a-3" and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 3


def test_empty_store_returns_nothing(tmp_path):
    assert FaissVectorStore(tmp_path, "m").search(_unit(1)[0], 3) == []


def test_persistence_round_trip(tmp_path):
    store = FaissVectorStore(tmp_path, "m")
    vecs = _unit(4)
    store.add(_chunks("a", 4), vecs)
    store.save()
    reloaded = FaissVectorStore(tmp_path, "m")
    assert reloaded.count == 4
    assert reloaded.search(vecs[2], 1)[0][0].chunk_id == "a-2"
    assert reloaded.get_chunk("a-1").text == "text 1"


def test_delete_document_removes_only_that_document(tmp_path):
    store = FaissVectorStore(tmp_path, "m")
    store.add(_chunks("a", 3), _unit(3, seed=1))
    store.add(_chunks("b", 2), _unit(2, seed=2))
    assert store.delete_document("a") == 3
    assert store.count == 2 and store.count_for("a") == 0
    assert all(c.doc_id == "b" for c, _ in store.search(_unit(1, seed=9)[0], 10))
    assert store.get_chunk("a-0") is None


def test_ids_stay_unique_after_delete_and_add(tmp_path):
    store = FaissVectorStore(tmp_path, "m")
    store.add(_chunks("a", 2), _unit(2))
    store.delete_document("a")
    store.add(_chunks("b", 2), _unit(2, seed=3))
    assert store.count == 2 and len(store.search(_unit(1)[0], 5)) == 2


def test_embedding_model_mismatch_is_detected(tmp_path):
    store = FaissVectorStore(tmp_path, "model-a")
    store.add(_chunks("a", 2), _unit(2))
    store.save()
    with pytest.raises(IndexMismatchError):
        FaissVectorStore(tmp_path, "model-b").search(_unit(1)[0], 1)


def test_dimension_mismatch_is_rejected(tmp_path):
    store = FaissVectorStore(tmp_path, "m")
    store.add(_chunks("a", 1), _unit(1, dim=8))
    with pytest.raises(IndexMismatchError):
        store.add(_chunks("b", 1), _unit(1, dim=4))
