from sentence_transformers import CrossEncoder


# ============================================================
# RERANKER MODEL
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# LOAD RERANKER
# ============================================================

def load_reranker():
    """
    Load the Cross-Encoder reranking model.
    """

    print("\nLoading reranker model...")
    print("Model:", MODEL_NAME)

    model = CrossEncoder(
        MODEL_NAME
    )

    print("Reranker model loaded.")

    return model


# ============================================================
# RERANK RESULTS
# ============================================================

def rerank(
    query,
    results,
    model,
    top_k=3
):
    """
    Rerank retrieved chunks using a Cross-Encoder.

    Args:
        query: Farmer's question
        results: Results returned by retriever.py
        model: Cross-Encoder model
        top_k: Number of final results

    Returns:
        Reranked results
    """

    if not results:
        return []

    # --------------------------------------------------------
    # Create question + document pairs
    # --------------------------------------------------------

    pairs = []

    for result in results:

        pairs.append([
            query,
            result["text"]
        ])

    # --------------------------------------------------------
    # Calculate relevance scores
    # --------------------------------------------------------

    scores = model.predict(
        pairs
    )

    # --------------------------------------------------------
    # Add reranking score
    # --------------------------------------------------------

    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        updated_result = result.copy()

        updated_result["rerank_score"] = float(
            score
        )

        reranked_results.append(
            updated_result
        )

    # --------------------------------------------------------
    # Sort by reranking score
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Return top results
    # --------------------------------------------------------

    return reranked_results[:top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_reranked_results(
    query,
    results
):

    print("\n===================================")
    print("RERANKED RESULTS")
    print("===================================")

    print("\nQuestion:")
    print(query)

    print(
        "\nFinal results:",
        len(results)
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(
            f"\nRESULT {number}"
        )

        print("-----------------------------------")

        print(
            "Original Score:",
            round(
                result["score"],
                4
            )
        )

        print(
            "Rerank Score:",
            round(
                result["rerank_score"],
                4
            )
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


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from retriever import (
        load_vector_store,
        load_embedding_model,
        retrieve
    )

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - RERANKER")
    print("===================================")

    # --------------------------------------------------------
    # STEP 1: Load FAISS
    # --------------------------------------------------------

    print("\nLoading vector store...")

    index, chunks = load_vector_store()

    print(
        "FAISS vectors:",
        index.ntotal
    )

    # --------------------------------------------------------
    # STEP 2: Load BGE
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # STEP 3: Test question
    # --------------------------------------------------------

    query = (
        "What subsidy is available for farmers "
        "for agricultural machinery?"
    )

    # --------------------------------------------------------
    # STEP 4: Retrieve candidates
    # --------------------------------------------------------

    retrieved_results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        model=embedding_model,
        top_k=5
    )

    print(
        "\nRetrieved candidates:",
        len(retrieved_results)
    )

    # --------------------------------------------------------
    # STEP 5: Load reranker
    # --------------------------------------------------------

    reranker_model = load_reranker()

    # --------------------------------------------------------
    # STEP 6: Rerank
    # --------------------------------------------------------

    final_results = rerank(
        query=query,
        results=retrieved_results,
        model=reranker_model,
        top_k=3
    )

    # --------------------------------------------------------
    # STEP 7: Display
    # --------------------------------------------------------

    display_reranked_results(
        query,
        final_results
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n===================================")
    print("RERANKING COMPLETED")
    print("===================================")