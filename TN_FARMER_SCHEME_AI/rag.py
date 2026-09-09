from retriever import retrieve
from vector_store import load_vector_store
from embedder import load_embedding_model
from reranker import load_reranker, rerank
from guardrails import apply_guardrails
from generator import generate_from_evidence


# ============================================================
# RAG CONFIGURATION
# ============================================================

RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

def load_rag_components():
    """
    Load all RAG components once.

    Components:
    - FAISS vector store
    - BGE embedding model
    - CrossEncoder reranker
    """

    print("Loading RAG components...")

    # Load FAISS vector store
    index, chunks = load_vector_store()

    # Load BGE embedding model
    embedding_model = load_embedding_model()

    # Load CrossEncoder reranker
    reranker_model = load_reranker()

    print("RAG components loaded successfully.")

    return {
        "index": index,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model
    }


# ============================================================
# RUN RAG PIPELINE
# ============================================================

def run_rag(query, components):
    """
    Run the complete RAG pipeline.

    Flow:

    Farmer Question
          ↓
    BGE Embedding
          ↓
    FAISS Semantic Retrieval
          ↓
    Top 5 Results
          ↓
    CrossEncoder Reranking
          ↓
    Top 3 Results
          ↓
    Guardrails
          ↓
    Official Evidence
          ↓
    Qwen LLM
          ↓
    Final Answer
    """

    # --------------------------------------------------------
    # STEP 1: SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    retrieved_results = retrieve(
        query=query,
        index=components["index"],
        chunks=components["chunks"],
        model=components["embedding_model"],
        top_k=RETRIEVAL_TOP_K
    )

    # --------------------------------------------------------
    # STEP 2: RERANKING
    # --------------------------------------------------------

    reranked_results = rerank(
        query=query,
        results=retrieved_results,
        model=components["reranker_model"],
        top_k=RERANK_TOP_K
    )

    # --------------------------------------------------------
    # STEP 3: GUARDRAILS
    # --------------------------------------------------------

    guardrail_result = apply_guardrails(
        query=query,
        results=reranked_results
    )

    # --------------------------------------------------------
    # STEP 4: STOP IF QUESTION IS NOT ALLOWED
    # --------------------------------------------------------

    if not guardrail_result["allowed"]:

        return {
            "query": query,
            "answer": guardrail_result["message"],
            "retrieved_results": retrieved_results,
            "reranked_results": reranked_results,
            "evidence": [],
            "generated": False,
            "question_status": guardrail_result["question_status"]
        }

    # --------------------------------------------------------
    # STEP 5: GENERATE ANSWER FROM GUARDRAIL RESULT
    # --------------------------------------------------------

    generator_result = generate_from_evidence(
        query=query,
        guardrail_result=guardrail_result
    )

    # --------------------------------------------------------
    # STEP 6: FINAL RESULT
    # --------------------------------------------------------

    return {
        "query": query,

        "answer": generator_result["answer"],

        "retrieved_results": retrieved_results,

        "reranked_results": reranked_results,

        "evidence": generator_result.get(
            "evidence",
            guardrail_result["evidence"]
        ),

        "generated": generator_result.get(
            "generated",
            False
        ),

        "question_status": guardrail_result.get(
            "question_status",
            "VALID"
        )
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    components = load_rag_components()

    question = input("\nEnter farmer question: ")

    result = run_rag(
        query=question,
        components=components
    )

    print("\n" + "=" * 70)
    print("RAG RESULT")
    print("=" * 70)

    print("\nQuestion:")
    print(result["query"])

    print("\nQuestion Status:")
    print(result["question_status"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nGenerated:")
    print(result["generated"])

    print("\nEvidence:")

    for evidence in result["evidence"]:
        print("-" * 70)
        print(evidence)

    print("\n" + "=" * 70)
    print("RAG COMPLETED")
    print("=" * 70)