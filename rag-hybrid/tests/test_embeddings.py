"""Checks SentenceTransformerEmbedder's plumbing against a stub of the library (the real model needs a download)."""

import sys
import types

import numpy as np

from backend.app.vector.embeddings import SentenceTransformerEmbedder


def _install_stub(monkeypatch):
    calls = {}

    class StubModel:
        def __init__(self, name, device=None):
            calls["init"] = (name, device)

        def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
            calls["encode"] = dict(n=len(texts), batch_size=batch_size, normalize=normalize_embeddings)
            return np.tile(np.array([[3.0, 4.0]]), (len(texts), 1))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_lazy_load_and_shapes(monkeypatch):
    calls = _install_stub(monkeypatch)
    emb = SentenceTransformerEmbedder("some/model", device="cpu", batch_size=8)
    assert "init" not in calls  # nothing loaded until first use
    docs = emb.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, 2) and docs.dtype == np.float32
    assert calls["init"] == ("some/model", "cpu")
    assert calls["encode"] == {"n": 3, "batch_size": 8, "normalize": True}
    assert emb.embed_query("q").shape == (2,)
    assert emb.embed_documents([]).shape[0] == 0
    assert emb.name == "some/model"
