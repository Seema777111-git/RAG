from embedder import load_embedding_model
from vector_store import load_vector_store


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def retrieve(query, index, chunks, model, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, position in zip(scores[0], indices[0]):

        if position == -1:
            continue

        chunk = chunks[position]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results


def display_results(query, results):

    print("\n===================================")
    print("RETRIEVAL RESULTS")
    print("===================================")

    print("\nQuestion:")
    print(query)

    print("\nResults found:", len(results))

    for number, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nRESULT {number}")
        print("-----------------------------------")

        print(
            "Similarity Score:",
            round(result["score"], 4)
        )

        print(
            "Scheme ID:",
            metadata.get("id")
        )

        print(
            "Department:",
            metadata.get("department")
        )

        print(
            "Scheme:",
            metadata.get("scheme_name")
        )

        print(
            "Chunk:",
            metadata.get("chunk_number")
        )

        print("\nText:")
        print(result["text"])


if __name__ == "__main__":

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - RETRIEVER")
    print("===================================")

    print("\nLoading FAISS vector store...")

    index, chunks = load_vector_store()

    print("FAISS vectors:", index.ntotal)
    print("Stored chunks:", len(chunks))

    print("\nLoading BGE embedding model...")

    model = load_embedding_model()

    query = (
        "What subsidy is available for farmers "
        "for agricultural machinery?"
    )

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        model=model,
        top_k=5
    )

    display_results(
        query,
        results
    )

    print("\n===================================")
    print("RETRIEVAL COMPLETED")
    print("===================================")