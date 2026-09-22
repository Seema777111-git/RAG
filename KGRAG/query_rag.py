import os
import json
import faiss
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class HybridRAGPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        # Load FAISS index and metadata if available
        self.index_path = "faiss_index.bin"
        self.meta_path = "faiss_metadata.json"
        
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.faiss_index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.faiss_metadata = json.load(f)
        else:
            self.faiss_index = None
            self.faiss_metadata = []

    def close(self):
        if self.driver:
            self.driver.close()

    def guardrail_input(self, query: str, selected_department: str = "Agriculture Department") -> tuple[bool, str]:
        """Validates query safety and relevance."""
        query_lower = query.lower()
        blocked_keywords = ["hack", "jailbreak", "override", "weather", "movie", "recipe", "sql injection"]
        if any(word in query_lower for word in blocked_keywords):
            return False, json.dumps({
                "status": "blocked",
                "message": "Query blocked: Please ask questions strictly related to Tamil Nadu Government welfare schemes."
            }, ensure_ascii=False)
        return True, "Passed"

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def search_vector_store(self, query: str, top_k: int = 5):
        if not self.faiss_index or not self.faiss_metadata:
            return [], []
        
        query_embedding = np.array([self.get_embedding(query)], dtype=np.float32)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        retrieved_contexts = []
        extracted_schemes = []
        
        for idx in indices[0]:
            if idx != -1 and idx < len(self.faiss_metadata):
                item = self.faiss_metadata[idx]
                retrieved_contexts.append(item.get("composite_text", ""))
                scheme_name = item.get("scheme_name")
                if scheme_name and scheme_name not in extracted_schemes:
                    extracted_schemes.append(scheme_name)
                    
        return retrieved_contexts, extracted_schemes

    def search_knowledge_graph(self, extracted_schemes: list):
        graph_facts = []
        if not extracted_schemes:
            return graph_facts
        
        query_cypher = """
        MATCH (s:Scheme)
        WHERE s.name IN $schemes
        OPTIONAL MATCH (s)-[:OFFERED_BY]->(d:Department)
        OPTIONAL MATCH (s)-[:TARGETS]->(b:Beneficiary)
        OPTIONAL MATCH (s)-[:PROVIDES]->(bt:BenefitType)
        RETURN s.name AS scheme, d.name AS department, collect(DISTINCT b.name) AS beneficiaries, collect(DISTINCT bt.name) AS benefits
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query_cypher, schemes=extracted_schemes)
                for record in result:
                    scheme = record["scheme"]
                    dept = record["department"] or "Unknown"
                    beneficiaries = ", ".join(record["beneficiaries"]) if record["beneficiaries"] else "General"
                    benefits = ", ".join(record["benefits"]) if record["benefits"] else "General Support"
                    fact = f"Scheme: {scheme} | Department: {dept} | Target Beneficiaries: {beneficiaries} | Benefits: {benefits}"
                    graph_facts.append(fact)
        except Exception as e:
            print(f"Graph search error: {e}")
            
        return graph_facts

    def evaluate_faithfulness(self, contexts: list, answer: str) -> str:
        if not contexts:
            return "No contexts available for evaluation."
        
        context_str = "\n".join(contexts)
        eval_prompt = f"""You are an evaluation judge. Determine whether the generated answer is fully grounded in and supported by the provided context. Answer with 'Supported' or 'Unsupported' followed by a short reason.

Context:
{context_str}

Answer:
{answer}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Evaluation error: {e}"

    def run_pipeline(self, query: str, selected_department: str = "Agriculture Department") -> str:
        """Executes full pipeline constrained by the selected department outputting structured JSON."""
        is_safe, msg = self.guardrail_input(query, selected_department)
        if not is_safe:
            return msg

        text_contexts, extracted_schemes = self.search_vector_store(query, top_k=5)
        graph_facts = self.search_knowledge_graph(extracted_schemes)

        vector_context_str = "\n\n".join(text_contexts) if text_contexts else "No vector text retrieved."
        graph_context_str = "\n".join(graph_facts) if graph_facts else "No direct graph relationships found."

        combined_prompt = f"""You are an expert advisor on Tamil Nadu Government Schemes.
The user has specifically selected the department: **{selected_department}**.
Answer the user query using the retrieved context, ensuring the response aligns with this department context. If the query asks about a completely different department or topic outside of {selected_department} and the retrieved data, politely inform the user that the query does not match the selected department.

---
UNSTRUCTURED SCHEME DESCRIPTIONS (Vector Search):
{vector_context_str}

---
STRUCTURED RELATIONSHIPS & ELIGIBILITY (Knowledge Graph):
{graph_context_str}

---
USER QUERY:
{query}

Provide a well-structured response detailing eligible schemes, target beneficiaries, and key benefits based strictly on the provided context."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and precise government welfare scheme assistant."},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content

        all_contexts = text_contexts + graph_facts
        eval_status = self.evaluate_faithfulness(all_contexts, answer)

        output_data = {
            "query": query,
            "selected_department": selected_department,
            "extracted_schemes": extracted_schemes,
            "response": answer,
            "evaluation_status": eval_status
        }

        return json.dumps(output_data, indent=2, ensure_ascii=False)