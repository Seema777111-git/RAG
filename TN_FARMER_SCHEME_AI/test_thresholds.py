"""
test_thresholds.py

Run this from your project root:
    python test_thresholds.py

Purpose:
    Prints the retrieval score and rerank score for a mix of
    on-topic (real scheme) questions and off-topic (unrelated)
    questions, so you can see where the score gap actually is
    and set MIN_RETRIEVAL_SCORE / MIN_RERANK_SCORE in
    guardrails.py based on real numbers instead of guesses.
"""

from retriever import retrieve
from vector_store import load_vector_store
from embedder import load_embedding_model
from reranker import load_reranker, rerank


RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3


# ------------------------------------------------------------
# TEST QUESTIONS
# ------------------------------------------------------------

ON_TOPIC_QUESTIONS = [
    "What subsidy is available for farmers for agricultural machinery?",
    "Enakku drip irrigation-ku eligibility enna?",
    "விவசாயிகளுக்கு டிராக்டர் மானியம் என்ன?",
]

OFF_TOPIC_QUESTIONS = [
    "What is the capital of France?",
    "How do I apply for a passport?",
    "Tell me a joke about cricket.",
]


def run_question(label, query, index, chunks, embedding_model, reranker_model):

    print("\n" + "=" * 70)
    print(f"[{label}] {query}")
    print("=" * 70)

    retrieved = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        model=embedding_model,
        top_k=RETRIEVAL_TOP_K
    )

    if not retrieved:
        print("No retrieval results at all.")
        return

    reranked = rerank(
        query=query,
        results=retrieved,
        model=reranker_model,
        top_k=RERANK_TOP_K
    )

    for i, result in enumerate(reranked, start=1):
        metadata = result.get("metadata", {})
        scheme_name = metadata.get("scheme_name", "UNKNOWN")
        retrieval_score = result.get("score", "N/A")
        rerank_score = result.get("rerank_score", "N/A")

        print(f"  #{i} scheme={scheme_name!r}")
        print(f"      retrieval_score={retrieval_score}")
        print(f"      rerank_score={rerank_score}")


def main():

    print("Loading components (this may take a moment)...")

    index, chunks = load_vector_store()
    embedding_model = load_embedding_model()
    reranker_model = load_reranker()

    print("\nComponents loaded. Running test questions...\n")

    for query in ON_TOPIC_QUESTIONS:
        run_question("ON-TOPIC", query, index, chunks, embedding_model, reranker_model)

    for query in OFF_TOPIC_QUESTIONS:
        run_question("OFF-TOPIC", query, index, chunks, embedding_model, reranker_model)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        "\nLook at the gap between ON-TOPIC and OFF-TOPIC scores above.\n"
        "Set MIN_RETRIEVAL_SCORE and MIN_RERANK_SCORE in guardrails.py\n"
        "to a value that sits between the two groups \u2014 high enough to\n"
        "reject the off-topic scores, low enough to keep the on-topic ones."
    )


if __name__ == "__main__":
    main()