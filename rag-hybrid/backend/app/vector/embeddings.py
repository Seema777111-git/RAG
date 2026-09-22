"""Embedding models. `Embedder` is a protocol so tests (and other backends) can plug in."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised vector."""


class SentenceTransformerEmbedder:
    """Local sentence-transformers model. Loaded lazily on first use (model download can be slow)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s on %s", self.name, self._device)
                    self._model = SentenceTransformer(self.name, device=self._device)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        with self._lock:  # SentenceTransformer.encode is not guaranteed thread-safe
            vectors = model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]
