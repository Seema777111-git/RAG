"""JSON registry of ingested documents."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from backend.app.domain.models import DocumentInfo


class DocumentRegistry:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._docs: dict[str, DocumentInfo] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._docs = {d["doc_id"]: DocumentInfo(**d) for d in data}

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([d.model_dump(mode="json") for d in self._docs.values()], indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def upsert(self, doc: DocumentInfo) -> None:
        with self._lock:
            self._docs[doc.doc_id] = doc
            self._flush()

    def remove(self, doc_id: str) -> Optional[DocumentInfo]:
        with self._lock:
            doc = self._docs.pop(doc_id, None)
            if doc:
                self._flush()
            return doc

    def get(self, doc_id: str) -> Optional[DocumentInfo]:
        with self._lock:
            return self._docs.get(doc_id)

    def list(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.ingested_at, reverse=True)
