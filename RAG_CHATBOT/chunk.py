from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(text, chunk_size=900, chunk_overlap=150):
    """
    Split extracted document text into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    print("Chunking completed!")
    print(f"Total chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    from pdf_reader import read_pdf

    pdf_path = "Resource/Blessed child parenting.pdf"

    # Read PDF
    text = read_pdf(pdf_path)

    # Create chunks
    chunks = chunk_pages(text)

    # Display first 3 chunks
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)