# 📄 RAG Chatbot (Basic RAG)

**Level: Basic.** My first Retrieval-Augmented Generation project: upload a PDF, ask a question, and get an answer
grounded only in that document.

It covers the core RAG loop with no extras: **PDF → chunks → embeddings → vector store → retrieve → LLM answer**.
The advanced versions in this repo build on it:
[rag-hybrid](../rag-hybrid) (vector + knowledge graph, guardrails, evaluation) and
[TN_FARMER_SCHEME_AI](../TN_FARMER_SCHEME_AI) (FAISS + BM25 hybrid, reranking, guardrails, multilingual).

---

## Two versions inside

### 1. Streamlit app (`app.py`): HR Policy Assistant
A LangChain pipeline with a web UI.

```
Upload PDF → PyPDFLoader → RecursiveCharacterTextSplitter (1000 / 200)
          → OpenAI embeddings (text-embedding-3-small) → FAISS
          → top 4 chunks → GPT-4o-mini → answer
```

Sample document: `HRPolicy.pdf`.

### 2. Step-by-step scripts: the pipeline built by hand
The same idea, with each RAG step written as its own module so it is easy to follow.

| Step | File | What it does |
|---|---|---|
| 1. Read | `pdf_reader.py` | Extracts text from the PDF (pypdf) |
| 2. Chunk | `chunk.py` | Splits text into 900-character chunks with 150 overlap |
| 3. Embed | `embedder.py` | Local BGE embeddings (`BAAI/bge-base-en-v1.5`) |
| 4. Store | `vector_store.py` | Saves chunks and vectors in ChromaDB (`./chroma_db`) |
| Ingest all | `data_processor.py` | Runs steps 1 to 4 in order |
| 5. Retrieve | `query_processor.py` | Embeds the question, gets the top 3 chunks |
| 6. Answer | `llm.py` | GPT-4o-mini answers only from the retrieved context |
| Test | `test_rag.py` | Asks one question end to end |

---

## 🛠️ Tech Stack

- Python, Streamlit
- LangChain (loaders, text splitters, prompts, LCEL chain)
- Embeddings: OpenAI `text-embedding-3-small`, or local `BAAI/bge-base-en-v1.5` (sentence-transformers)
- Vector stores: FAISS (app), ChromaDB (scripts)
- LLM: OpenAI `gpt-4o-mini`

---

## 🚀 Run it

```bash
cd RAG_CHATBOT
pip install -r requirements.txt
echo OPENAI_API_KEY=sk-... > .env
```

**Streamlit app**
```bash
streamlit run app.py
```

**Scripts**: put a PDF in `Resource/` and set its path in `data_processor.py`, then:
```bash
python data_processor.py   # read, chunk, embed, store
python test_rag.py         # ask a question
```

---

## What this version does not have yet

These are added in the advanced projects in this repo:

- Hybrid retrieval (keyword BM25 or a knowledge graph next to vectors)
- Reranking
- Guardrails for off-topic or vague questions
- Evaluation and source citations

---

## 👤 Author

Built by [Seema777111-git](https://github.com/Seema777111-git)
