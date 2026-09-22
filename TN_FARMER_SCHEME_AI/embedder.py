from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# LOAD MODEL
# ============================================================

def load_embedding_model():
    """
    Load the BGE embedding model.
    """

    print("\nLoading embedding model...")
    print("Model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):
    """
    Convert chunk text into numerical vectors.

    Args:
        chunks: Output from chunker.py
        model: BGE embedding model

    Returns:
        embeddings: Numerical vectors
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nCreating embeddings...")
    print("Input chunks:", len(texts))

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embeddings created.")

    return embeddings


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from loader import load_structured_data
    from chunker import create_chunks

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - EMBEDDER")
    print("===================================")

    # --------------------------------------------------------
    # STEP 1: Load records
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

    # --------------------------------------------------------
    # Show embedding information
    # --------------------------------------------------------

    print("\n===================================")
    print("EMBEDDING RESULT")
    print("===================================")

    print("Number of chunks :", len(chunks))
    print("Number of vectors:", len(embeddings))
    print("Vector dimension :", len(embeddings[0]))

    print("\nFirst vector:")
    print(embeddings[0])

    print("\n===================================")
    print("EMBEDDING COMPLETED")
    print("===================================")