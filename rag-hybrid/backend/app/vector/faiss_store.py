"""Persistent FAISS vector store (exact cosine search) with chunk metadata and per-document deletion."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from backend.app.core.errors import IndexMismatchError
from backend.app.domain.models import Chunk

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"


class FaissVectorStore:
    """`IndexIDMap2(IndexFlatIP)` over L2-normalised vectors, i.e. cosine similarity.

    Exact search is the right default up to a few hundred thousand chunks. For larger corpora,
    swap the index for `IndexHNSWFlat` / `IndexIVFFlat`; the rest of the class stays the same.
    """

    def __init__(self, directory: Path, embedding_model: str):
        self._dir = Path(directory)
        self._embedding_model = embedding_model
        self._lock = threading.RLock()
        self._index: Optional[faiss.Index] = None
        self._chunks: dict[int, Chunk] = {}
        self._by_chunk_id: dict[str, int] = {}
        self._next_id = 0
        self._stored_model: Optional[str] = None
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        index_path, meta_path = self._dir / INDEX_FILE, self._dir / META_FILE
        if not (index_path.exists() and meta_path.exists()):
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._index = faiss.read_index(str(index_path))
        self._next_id = int(meta.get("next_id", 0))
        self._stored_model = meta.get("embedding_model")
        for key, value in meta.get("chunks", {}).items():
            chunk = Chunk(**value)
            self._chunks[int(key)] = chunk
            self._by_chunk_id[chunk.chunk_id] = int(key)
        logger.info("Loaded FAISS index: %d vectors (model=%s)", len(self._chunks), self._stored_model)

    def save(self) -> None:
        with self._lock:
            if self._index is None:
                return
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_index = self._dir / f"{INDEX_FILE}.tmp"
            faiss.write_index(self._index, str(tmp_index))
            os.replace(tmp_index, self._dir / INDEX_FILE)
            meta = {
                "embedding_model": self._embedding_model,
                "dim": int(self._index.d),
                "next_id": self._next_id,
                "chunks": {str(i): c.model_dump() for i, c in self._chunks.items()},
            }
            tmp_meta = self._dir / f"{META_FILE}.tmp"
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_meta, self._dir / META_FILE)
            self._stored_model = self._embedding_model

    # ------------------------------------------------------------------ guards
    def check_compatible(self) -> None:
        if self._stored_model and self._stored_model != self._embedding_model and self._chunks:
            raise IndexMismatchError(
                f"The FAISS index was built with '{self._stored_model}' but EMBEDDING_MODEL is "
                f"'{self._embedding_model}'. Re-ingest your documents or restore the original model."
            )

    # ------------------------------------------------------------------ mutation
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        with self._lock:
            self.check_compatible()
            if self._index is None:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
            elif vectors.shape[1] != self._index.d:
                raise IndexMismatchError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self._index.d}.")
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
            self._index.add_with_ids(vectors, ids)
            for i, chunk in zip(ids.tolist(), chunks):
                self._chunks[i] = chunk
                self._by_chunk_id[chunk.chunk_id] = i
            self._next_id += len(chunks)

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [i for i, c in self._chunks.items() if c.doc_id == doc_id]
            if not ids or self._index is None:
                return 0
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            for i in ids:
                chunk = self._chunks.pop(i)
                self._by_chunk_id.pop(chunk.chunk_id, None)
            return len(ids)

    # ------------------------------------------------------------------ queries
    def search(self, query: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            self.check_compatible()
            if self._index is None or self._index.ntotal == 0:
                return []
            q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            scores, ids = self._index.search(q, min(k, self._index.ntotal))
            return [(self._chunks[int(i)], float(s)) for s, i in zip(scores[0], ids[0]) if int(i) in self._chunks]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            i = self._by_chunk_id.get(chunk_id)
            return self._chunks.get(i) if i is not None else None

    def iter_chunks(self, doc_id: Optional[str] = None) -> list[Chunk]:
        with self._lock:
            return [c for c in self._chunks.values() if doc_id is None or c.doc_id == doc_id]

    def count_for(self, doc_id: str) -> int:
        with self._lock:
            return sum(1 for c in self._chunks.values() if c.doc_id == doc_id)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def dimension(self) -> Optional[int]:
        return int(self._index.d) if self._index is not None else None
