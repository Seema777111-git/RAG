from backend.app.ingestion.chunker import chunk_document, split_text
from backend.app.ingestion.pdf_loader import LoadedPdf, PageText


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", 200, 20) == ["Hello world."]


def test_long_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 + 40 for c in chunks)
    # overlap: the start of each chunk repeats words from the end of the previous one
    first_words = chunks[1].split()[:3]
    assert all(w in chunks[0] for w in first_words)


def test_paragraphs_are_packed_together():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert split_text(text, 500, 0) == ["First paragraph here.\nSecond paragraph here."]


def test_giant_token_is_hard_split():
    chunks = split_text("x" * 1000, 300, 0)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks) == "x" * 1000


def test_chunk_document_ids_and_pages():
    pdf = LoadedPdf("abc123", "f.pdf", 2, [PageText(1, "A" * 80), PageText(2, "B" * 90)], 1)
    chunks = chunk_document(pdf, 500, 0, min_chars=20)
    assert [c.chunk_id for c in chunks] == ["abc123-p1-c0", "abc123-p2-c0"]
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.source == "f.pdf" for c in chunks)


def test_tiny_chunks_are_dropped():
    pdf = LoadedPdf("d", "f.pdf", 1, [PageText(1, "tiny")], 1)
    assert chunk_document(pdf, 500, 0, min_chars=20) == []
