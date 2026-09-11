# KGRAG — Knowledge Graph + Retrieval-Augmented Generation

KGRAG is an AI-powered **Knowledge Graph + Retrieval-Augmented Generation (RAG)** project designed to retrieve reliable information from structured and unstructured government scheme data and provide grounded answers.

The project combines **data ingestion, cleaning, vector embeddings, FAISS retrieval, and LLM-based question answering** to create a domain-focused AI assistant.

## 🎯 Project Objective

The goal of KGRAG is to build an AI system that can:

* Collect government scheme information
* Clean and structure the source data
* Build a searchable knowledge base
* Retrieve relevant information for a user query
* Generate answers grounded in the retrieved evidence
* Reduce hallucinations by using retrieved source information

## 🏗️ Architecture

```text
Government Scheme Sources
          ↓
       Scraping
          ↓
    Data Cleaning
          ↓
   Structured JSON
          ↓
   RAG Knowledge Base
          ↓
    Vector Embeddings
          ↓
        FAISS
          ↓
      Retrieval
          ↓
    Relevant Context
          ↓
        LLM
          ↓
   Grounded AI Answer
```

## 📁 Project Structure

```text
KGRAG/
│
├── app.py
├── build_rag.py
├── checkdb.py
├── config.py
├── query_rag.py
├── scrape.py
│
├── schemes.json
├── schemes_cleaned.json
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── requirements.txt
└── .gitignore
```

## 🔍 Key Components

### `scrape.py`

Collects government scheme information from the configured source.

Responsibilities include:

* Web data extraction
* URL handling
* Data collection
* Initial data preparation

### `schemes.json`

Contains the collected scheme data before cleaning and normalization.

### `schemes_cleaned.json`

Contains cleaned and normalized scheme information used by the RAG pipeline.

### `build_rag.py`

Builds the searchable RAG knowledge base.

Typical flow:

```text
Cleaned Data
     ↓
Documents
     ↓
Embeddings
     ↓
FAISS Index
```

### `faiss_index/`

Stores the generated FAISS vector index used for semantic retrieval.

```text
index.faiss
index.pkl
```

These files allow the application to retrieve relevant scheme information efficiently.

### `query_rag.py`

Handles user queries and retrieves relevant information from the RAG knowledge base.

### `app.py`

Provides the application interface and connects the user query workflow with the RAG pipeline.

### `config.py`

Contains configurable application and RAG settings.

### `checkdb.py`

Utility for checking the generated knowledge base/vector database.

## 🧠 RAG Workflow

KGRAG follows the following conceptual workflow:

```text
User Question
     ↓
Query Processing
     ↓
Semantic Retrieval
     ↓
FAISS Vector Search
     ↓
Relevant Scheme Information
     ↓
Context Construction
     ↓
LLM Generation
     ↓
Grounded Answer
```

The important principle is:

> **Retrieve relevant information first, then generate the answer using that context.**

This helps reduce unsupported responses and improves domain-specific accuracy.

## ⚙️ Technology Stack

* **Python**
* **RAG — Retrieval-Augmented Generation**
* **FAISS**
* **Vector Embeddings**
* **JSON**
* **Web Scraping**
* **Streamlit**
* **LLM-based Question Answering**

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Seema777111-git/RAG.git
cd RAG/KGRAG
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a local `.env` file if required by the application.

**Do not commit `.env` to GitHub.**

The repository `.gitignore` excludes sensitive environment files.

### 6. Build the RAG index

```powershell
python build_rag.py
```

### 7. Run the application

```powershell
streamlit run app.py
```

## 🧪 Testing and Validation

Before deploying changes, validate:

* Source data is available
* Cleaned data is generated correctly
* FAISS index is created successfully
* Retrieval returns relevant documents
* Generated answers are grounded in retrieved information
* Environment secrets are not committed

## 🔐 Security

Sensitive configuration should remain local.

The following files/directories should **not** be committed:

```text
.env
.venv/
venv/
__pycache__/
*.pyc
```

Never place API keys, passwords, tokens, or other credentials directly in source code.

## 📌 Future Enhancements

Potential improvements include:

* Knowledge Graph integration using Neo4j
* Hybrid keyword + vector retrieval
* Reranking
* Source citation and evidence display
* Conversational memory
* Agentic RAG
* Query classification
* Automated data refresh
* Evaluation framework
* Retrieval quality metrics
* Hallucination/grounding checks
* Multilingual government-scheme assistance

## 🎓 Portfolio Value

KGRAG demonstrates practical AI engineering concepts including:

* Retrieval-Augmented Generation
* Semantic Search
* Vector Databases
* Data Engineering
* Web Data Ingestion
* LLM Application Development
* AI Grounding
* Knowledge Base Construction
* AI Agent architecture foundations

This project is intended as a portfolio demonstration of building an **end-to-end domain-specific AI application** rather than a simple chatbot.

## 👤 Author

**Seema**

GitHub: `Seema777111-git`

Repository:

https://github.com/Seema777111-git/RAG/tree/main/KGRAG
