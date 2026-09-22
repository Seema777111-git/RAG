import faiss
import pickle
from pathlib import Path

from loader import load_structured_data
from chunker import create_chunks
from embedder import load_embedding_model, create_embeddings


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss.index"
CHUNKS_FILE = VECTORSTORE_DIR / "chunks.pkl"


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks, embeddings):
    """
    Create and save a FAISS vector index.

    Args:
        chunks: RAG text chunks
        embeddings: BGE embedding vectors
    """

    # --------------------------------------------------------
    # Convert embeddings to FAISS format
    # --------------------------------------------------------

    embeddings = embeddings.astype("float32")

    # --------------------------------------------------------
    # Get vector dimension
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print("\nVector dimension:", dimension)

    # --------------------------------------------------------
    # Create FAISS index
    #
    # IndexFlatIP = Inner Product similarity.
    # Because BGE embeddings are normalized,
    # this works as cosine similarity.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(dimension)

    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    index.add(embeddings)

    print("Vectors added to FAISS:", index.ntotal)

    # --------------------------------------------------------
    # Create vectorstore directory
    # --------------------------------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE)
    )

    # --------------------------------------------------------
    # Save chunks + metadata
    # --------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\nFAISS index saved to:")
    print(FAISS_INDEX_FILE)

    print("\nChunks and metadata saved to:")
    print(CHUNKS_FILE)

    return index


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    """
    Load an existing FAISS index and chunks.
    """

    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{FAISS_INDEX_FILE}"
        )

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_FILE}"
        )

    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )

    with open(
        CHUNKS_FILE,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return index, chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - VECTOR STORE")
    print("===================================")

    # --------------------------------------------------------
    # STEP 1: Load structured data
    # --------------------------------------------------------

    records = load_structured_data()

    print("\nRecords loaded:", len(records))

    # --------------------------------------------------------
    # STEP 2: Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(records)

    print("Chunks created:", len(chunks))

    # --------------------------------------------------------
    # STEP 3: Load BGE model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # STEP 4: Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        chunks,
        model
    )

    print(
        "Embeddings created:",
        len(embeddings)
    )

    # --------------------------------------------------------
    # STEP 5: Create FAISS vector store
    # --------------------------------------------------------

    index = create_vector_store(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n===================================")
    print("VECTOR STORE SUMMARY")
    print("===================================")

    print("Records :", len(records))
    print("Chunks  :", len(chunks))
    print("Vectors :", index.ntotal)
    print("Dimension:", index.d)

    print("\n===================================")
    print("FAISS VECTOR STORE COMPLETED")
    print("===================================")