# KGRAG — Knowledge Graph + Retrieval-Augmented Generation

**Level: Base.** My base Knowledge Graph RAG project. It answers questions about **Tamil Nadu Government
schemes** by combining **vector search (FAISS)** with a **knowledge graph (Neo4j)**, then generating a grounded
answer with GPT-4o and checking it for faithfulness.

The advanced projects in this repo build on these ideas:
[rag-hybrid](../rag-hybrid) (vector + knowledge graph with NeMo guardrails and DeepEval) and
[TN_FARMER_SCHEME_AI](../TN_FARMER_SCHEME_AI) (FAISS + BM25 hybrid, reranking, guardrails, multilingual).

## 🎯 What it does

* Scrapes scheme information from the official tn.gov.in portal (54 schemes)
* Cleans it into structured records (department, beneficiaries, communities, benefits, age and income limits)
* Indexes every scheme twice: as vectors in FAISS and as a graph in Neo4j
* For each question: guardrail → vector search → graph facts → GPT-4o answer → faithfulness check

## 🏗️ Architecture

```text
tn.gov.in scheme pages
          ↓
   scrape.py (Playwright)
          ↓
 schemes.json → schemes_cleaned.json
          ↓
     build_rag.py
     ├── FAISS index (OpenAI text-embedding-3-small)   → faiss_index/
     └── Neo4j Aura knowledge graph
           (Scheme)-[:ADMINISTERED_BY]->(Department)
           (Scheme)-[:TARGETS_BENEFICIARY]->(Beneficiary)
           (Scheme)-[:TARGETS_COMMUNITY]->(Community)
           (Scheme)-[:PROVIDES_BENEFIT]->(Benefit)
```

### Question flow (`query_rag.py`)

```text
User question
     ↓
Keyword guardrail (blocks off-topic / unsafe queries)
     ↓
FAISS vector search → top 5 scheme texts + scheme names
     ↓
Neo4j lookup for those schemes → department, beneficiaries, communities, benefits
     ↓
GPT-4o answer from both contexts
     ↓
GPT-4o-mini faithfulness check (Supported / Unsupported)
     ↓
Structured JSON → Streamlit UI (app.py)
```

## 📁 Project Structure

```text
KGRAG/
├── app.py                 # Streamlit UI
├── scrape.py              # Scrapes and cleans scheme data from tn.gov.in
├── build_rag.py           # Builds the FAISS index and the Neo4j graph
├── query_rag.py           # Hybrid pipeline: guardrail, vector + graph retrieval, answer, evaluation
├── config.py              # Settings and paths (reads .env)
├── checkdb.py             # Checks the Neo4j connection and graph contents
├── schemes.json           # Raw scraped data
├── schemes_cleaned.json   # Cleaned records (also the metadata for the FAISS rows)
├── faiss_index/           # index.faiss + index.pkl
└── requirements.txt
```

## ⚙️ Technology Stack

* **Python**, **Streamlit**
* **Playwright** for web scraping
* **OpenAI** embeddings (`text-embedding-3-small`), **GPT-4o** answers, **GPT-4o-mini** evaluation
* **FAISS** vector search
* **Neo4j Aura** knowledge graph (Cypher)
* **LangChain** (index building)

## 🚀 Setup

```powershell
git clone https://github.com/Seema777111-git/RAG.git
cd RAG/KGRAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium     # only needed to re-scrape
```

Create a local `.env` (never commit it):

```text
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

A free Neo4j Aura instance works. Free instances pause after a few days without use; resume them in the Aura
console.

```powershell
python checkdb.py        # check the Neo4j connection
python build_rag.py      # build FAISS + the knowledge graph
streamlit run app.py     # run the app
```

## 🛠️ Fixes in this version

* `query_rag.py` now loads the index that `build_rag.py` writes (`faiss_index/index.faiss`, with
  `schemes_cleaned.json` as row metadata). Before, it looked for `faiss_index.bin`, so vector search returned nothing.
* The graph query now uses the same relationship names as the build (`ADMINISTERED_BY`, `TARGETS_BENEFICIARY`,
  `TARGETS_COMMUNITY`, `PROVIDES_BENEFIT`), so graph facts are returned.
* Credentials come from `config.py` (`NEO4J_USERNAME`), the same as the build step.

## 📌 Next steps (done in the advanced projects)

* Keyword (BM25) + vector hybrid retrieval and reranking
* Score-based guardrails instead of a keyword list
* A labelled evaluation set
* Multilingual questions (Tamil / Tanglish)

## 👤 Author

Built by [Seema777111-git](https://github.com/Seema777111-git)
