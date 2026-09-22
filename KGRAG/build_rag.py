import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase
import config


# ============================================================
# STEP 3: FAISS VECTOR STORE BUILDER
# ============================================================

def build_vector_store(schemes):
    print("=" * 60)
    print("STEP 3: BUILDING FAISS VECTOR STORE")
    print("=" * 60)

    documents = []
    for item in schemes:
        doc = Document(
            page_content=item.get("composite_text", ""),
            metadata={
                "id": item.get("id", ""),
                "scheme_name": item.get("scheme_name", ""),
                "department": item.get("department", ""),
                "detail_url": item.get("detail_url", "")
            }
        )
        documents.append(doc)

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    print(f"Embedding {len(documents)} documents using: {config.EMBEDDING_MODEL}...")
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(config.FAISS_INDEX_DIR))
    print(f"FAISS index successfully saved to: {config.FAISS_INDEX_DIR}/")


# ============================================================
# STEP 4: NEO4J AURA KG INGESTION
# ============================================================

def build_knowledge_graph(schemes):
    print("\n" + "=" * 60)
    print("STEP 4: INGESTING KNOWLEDGE GRAPH INTO NEO4J AURA")
    print("=" * 60)

    if not config.NEO4J_URI or not config.NEO4J_PASSWORD:
        raise ValueError("NEO4J_URI or NEO4J_PASSWORD missing in config/.env.")

    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    constraint_query = """
    CREATE CONSTRAINT scheme_id_unique IF NOT EXISTS
    FOR (s:Scheme) REQUIRE s.id IS UNIQUE;
    """

    ingest_query = """
    MERGE (s:Scheme {id: $id})
    SET s.name = $scheme_name,
        s.url = $detail_url,
        s.description = $description,
        s.how_to_avail = $how_to_avail,
        s.age_min = $age_min,
        s.age_max = $age_max,
        s.income_max_inr = $income_max_inr

    MERGE (d:Department {name: $department})
    MERGE (s)-[:ADMINISTERED_BY]->(d)

    WITH s
    UNWIND $beneficiaries AS b_name
    MERGE (b:Beneficiary {name: b_name})
    MERGE (s)-[:TARGETS_BENEFICIARY]->(b)

    WITH s
    UNWIND $communities AS c_name
    MERGE (c:Community {name: c_name})
    MERGE (s)-[:TARGETS_COMMUNITY]->(c)

    WITH s
    UNWIND $benefits AS ben_name
    MERGE (ben:Benefit {name: ben_name})
    MERGE (s)-[:PROVIDES_BENEFIT]->(ben)
    """

    with driver.session() as session:
        print("Ensuring unique constraints on (:Scheme)...")
        session.run(constraint_query)

        print(f"Ingesting {len(schemes)} schemes and creating graph relationships...")
        for s in schemes:
            params = {
                "id": s["id"],
                "scheme_name": s["scheme_name"],
                "department": s["department"],
                "detail_url": s["detail_url"],
                "description": s.get("description") or "",
                "how_to_avail": s.get("how_to_avail") or "",
                "age_min": s.get("age_min"),
                "age_max": s.get("age_max"),
                "income_max_inr": s.get("income_max_inr"),
                "beneficiaries": s.get("beneficiaries", []),
                "communities": s.get("communities", []),
                "benefits": s.get("types_of_benefits", []),
            }
            session.run(ingest_query, params)

        # Verification query
        stats = session.run("""
        RETURN 
            COUNT { MATCH (s:Scheme) } AS scheme_count,
            COUNT { MATCH (d:Department) } AS dept_count,
            COUNT { MATCH (b:Beneficiary) } AS ben_count,
            COUNT { MATCH ()-[r]->() } AS rel_count
        """).single()

        print("\nGraph Ingestion Verification:")
        print(f"  Schemes       : {stats['scheme_count']}")
        print(f"  Departments   : {stats['dept_count']}")
        print(f"  Beneficiaries : {stats['ben_count']}")
        print(f"  Total Edges   : {stats['rel_count']}")

    driver.close()
    print("=" * 60)
    print("STEP 4 COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    if not config.CLEAN_DATA_PATH.exists():
        print(f"Error: {config.CLEAN_DATA_PATH} not found. Ensure schemes_cleaned.json exists.")
        return

    with open(config.CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    # Step 3
    build_vector_store(schemes)

    # Step 4
    build_knowledge_graph(schemes)


if __name__ == "__main__":
    main()