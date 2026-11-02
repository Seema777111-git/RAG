import re

from rank_bm25 import BM25Okapi

from embedder import load_embedding_model
from vector_store import load_vector_store


MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Reciprocal Rank Fusion constant (standard value from Cormack et al., 2009).
RRF_K = 60

# Keyword-search index, built once per chunk list.
_bm25_cache = {"key": None, "bm25": None}


# ============================================================
# BM25 KEYWORD SEARCH
# ============================================================

def tokenize(text):
    """Lowercase word tokens; keeps Tamil script and Latin words."""
    return re.findall(r"[\w஀-௿]+", str(text).lower())


def get_bm25(chunks):
    key = (id(chunks), len(chunks))
    if _bm25_cache["key"] != key:
        _bm25_cache["bm25"] = BM25Okapi([tokenize(c["text"]) for c in chunks])
        _bm25_cache["key"] = key
    return _bm25_cache["bm25"]


# ============================================================
# HYBRID RETRIEVAL: FAISS (dense) + BM25 (keyword) -> RRF
# ============================================================

def retrieve(query, index, chunks, model, top_k=5):
    """
    Hybrid retrieval.

    1. Dense: BGE embedding + FAISS cosine similarity
    2. Keyword: BM25 over the same chunks
    3. Fusion: Reciprocal Rank Fusion of both rankings

    Each result keeps "score" = dense cosine similarity, so the
    guardrail thresholds in guardrails.py keep their meaning.
    Extra fields: bm25_score, dense_rank, bm25_rank, rrf_score.
    """

    if not chunks:
        return []

    # ---- Dense search (score every chunk; the index is small) ----
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    dense_scores, dense_indices = index.search(
        query_embedding,
        min(index.ntotal, len(chunks))
    )

    dense_score = {}
    dense_rank = {}
    for rank, (score, position) in enumerate(
        zip(dense_scores[0], dense_indices[0]), start=1
    ):
        if position == -1:
            continue
        dense_score[int(position)] = float(score)
        dense_rank[int(position)] = rank

    # ---- BM25 keyword search ----
    bm25_scores = get_bm25(chunks).get_scores(tokenize(query))
    bm25_order = sorted(
        (i for i in range(len(chunks)) if bm25_scores[i] > 0),
        key=lambda i: bm25_scores[i],
        reverse=True
    )
    bm25_rank = {i: r for r, i in enumerate(bm25_order, start=1)}

    # ---- Reciprocal Rank Fusion ----
    fused = {}
    for position in set(dense_rank) | set(bm25_rank):
        rrf = 0.0
        if position in dense_rank:
            rrf += 1.0 / (RRF_K + dense_rank[position])
        if position in bm25_rank:
            rrf += 1.0 / (RRF_K + bm25_rank[position])
        fused[position] = rrf

    top = sorted(fused, key=lambda p: fused[p], reverse=True)[:top_k]

    results = []
    for position in top:
        chunk = chunks[position]
        results.append({
            "score": dense_score.get(position, 0.0),
            "bm25_score": float(bm25_scores[position]),
            "dense_rank": dense_rank.get(position),
            "bm25_rank": bm25_rank.get(position),
            "rrf_score": round(fused[position], 6),
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
            "BM25 Score:", round(result.get("bm25_score", 0), 3),
            "| Dense rank:", result.get("dense_rank"),
            "| BM25 rank:", result.get("bm25_rank"),
            "| RRF:", result.get("rrf_score")
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