"""Paragraph-aware text chunker with overlap. Chunks never cross page boundaries so citations stay exact."""

from __future__ import annotations

import re

from backend.app.domain.models import Chunk
from backend.app.ingestion.pdf_loader import LoadedPdf

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort for a single 'sentence' longer than `size`: cut on whitespace."""
    parts: list[str] = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        if len(sentence) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, size))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= size:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space != -1 else tail


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of roughly `chunk_size` characters (overlap is added on top)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())
        pieces.extend([paragraph] if len(paragraph) <= chunk_size else _split_long_paragraph(paragraph, chunk_size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= chunk_size:
            current = f"{current}\n{piece}"
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {piece}".strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_document(pdf: LoadedPdf, chunk_size: int, overlap: int, min_chars: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pdf.pages:
        index = 0
        for text in split_text(page.text, chunk_size, overlap):
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{pdf.doc_id}-p{page.page}-c{index}",
                    doc_id=pdf.doc_id,
                    source=pdf.filename,
                    page=page.page,
                    text=text,
                )
            )
            index += 1
    return chunks
