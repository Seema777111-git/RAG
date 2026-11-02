from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks):
    """
    Convert text chunks into embeddings.
    """

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    print("Embedding completed!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return embeddings