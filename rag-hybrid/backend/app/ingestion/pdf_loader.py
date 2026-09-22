"""PDF text extraction with PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.errors import PdfError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str


@dataclass(frozen=True)
class LoadedPdf:
    doc_id: str
    filename: str
    page_count: int
    pages: list[PageText]  # only pages that contain text
    size_bytes: int


def file_sha256(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def load_pdf(path: Path, filename: str | None = None) -> LoadedPdf:
    """Extract text page by page. Blocks are joined with blank lines so the chunker sees paragraphs."""
    import pymupdf  # imported lazily: keeps app start-up fast

    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several unrelated types
        raise PdfError(f"Cannot open PDF: {exc}") from exc

    pages: list[PageText] = []
    with doc:
        if doc.needs_pass:
            raise PdfError("The PDF is password protected.")
        page_count = doc.page_count
        for index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks", sort=True)
            paragraphs = []
            for block in blocks:
                # (x0, y0, x1, y1, text, block_no, block_type); type 0 = text, 1 = image
                if len(block) >= 7 and block[6] != 0:
                    continue
                text = " ".join(str(block[4]).split())
                if text:
                    paragraphs.append(text)
            if paragraphs:
                pages.append(PageText(page=index, text="\n\n".join(paragraphs)))

    if not pages:
        raise PdfError(
            "No extractable text found. The PDF may be a scan; run OCR on it first (e.g. `ocrmypdf`) and upload again."
        )
    logger.info("Loaded %s: %d/%d pages with text", path.name, len(pages), page_count)
    return LoadedPdf(
        doc_id=file_sha256(path),
        filename=filename or path.name,
        page_count=page_count,
        pages=pages,
        size_bytes=path.stat().st_size,
    )
