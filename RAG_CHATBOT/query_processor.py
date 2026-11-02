from sentence_transformers import SentenceTransformer
from vector_store import get_collection


MODEL_NAME = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def search_query(question, top_k=3):
    """
    Convert the user's question into an embedding
    and retrieve the most relevant chunks from ChromaDB.
    """

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    print("\nSearch completed!")
    print(f"Question: {question}")
    print(f"Chunks retrieved: {len(results['documents'][0])}")

    return results

def get_context(question, top_k=3):
    """
    Retrieve relevant chunks and combine them into one context.
    """

    results = search_query(question, top_k)

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    return context