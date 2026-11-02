print("data_processor.py started")

from pdf_reader import read_pdf
from chunk import chunk_pages
from embedder import embed_chunks
from vector_store import store_in_chroma, verify_storage


pdf_path = "Resource/Blessed child parenting.pdf"


def run():

    print("Step 1: Reading PDF...")
    text = read_pdf(pdf_path)

    print("Step 2: Creating chunks...")
    chunks = chunk_pages(text)

    print("Step 3: Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Step 4: Storing in ChromaDB...")
    store_in_chroma(chunks, embeddings)

    print("Step 5: Verifying vectors...")
    verify_storage()

    print("RAG ingestion completed successfully!")


if __name__ == "__main__":
    run()