# 🌾 TN Farmer Scheme AI Assistant

**An AI-powered assistant that helps Tamil Nadu farmers discover government schemes they're eligible for — in English, தமிழ், or Tanglish.**

Government scheme information is scattered, hard to search, and locked behind bureaucratic language. Most farmers never find out they qualify for subsidies that exist specifically for them. This project fixes that with a RAG (Retrieval-Augmented Generation) pipeline built directly on live data from the official **TN Agrisnet portal**.

---

## 🏆 Highlights

- 🗣️ **True multilingual support** — not just translation, but real query understanding in English, தமிழ், and Tanglish, validated against a dedicated multilingual test set
- 🏛️ **100% official, live government data** — 55+ real schemes ingested directly from the TN Agrisnet portal, not synthetic or scraped-once data
- 🔍 **Hybrid retrieval + reranking** — combines dense vector search with keyword (BM25) search, then reranks with a cross-encoder for precision most basic RAG demos skip
- 🛡️ **Guardrails that actually refuse to answer** — vague or off-topic questions are rejected instead of hallucinated, with thresholds tuned from real scored test data (`test_thresholds.py`)
- 📚 **Full traceability** — every answer ships with the exact source scheme behind it, plus a human-readable PDF export of the underlying data for audit
- 🧾 **Rich preserved metadata** — department, eligibility, required documents, guidelines link, retrieval date, and scheme year are all kept intact end-to-end, not flattened away
- 📊 **Built-in evaluation harness** — ships with its own labeled question bank (valid / insufficient / multilingual) to measure accuracy, not just a demo script
- 🎨 **Production-feel UI** — a polished, government-styled Streamlit interface designed for non-technical, first-time users

---

## 🎯 The Problem

- Tamil Nadu government publishes dozens of farmer welfare schemes, but the information sits in disconnected, hard-to-navigate government pages
- Farmers often can't tell which schemes they're eligible for, what documents are needed, or where to apply
- Language is a barrier — most portals are English-only, while many farmers are more comfortable in Tamil or Tanglish

## 💡 The Solution

A chatbot that:
- Answers scheme-related questions in **English, Tamil, or Tanglish**
- Pulls answers only from **verified, official scheme data** — not general AI guesswork
- Shows the **exact source scheme** behind every answer, so users can verify and apply
- **Refuses to answer** off-topic or unclear questions instead of hallucinating a response

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌐 **Multilingual queries** | Ask in English, தமிழ், or Tanglish — the assistant understands all three |
| 🔍 **Hybrid retrieval** | Combines dense vector search (FAISS) with cross-encoder reranking for accurate matches |
| 🛡️ **Guardrails** | Rejects vague, off-topic, or insufficiently specific questions instead of guessing |
| 📄 **Grounded answers** | Every answer is generated only from retrieved scheme evidence, with the source scheme cited |
| 🏛️ **Live government data** | Scheme data ingested directly from the official [TN Agrisnet portal](https://www.tnagrisnet.tn.gov.in/people_app/goScheme/) |
| 📊 **Built-in evaluation** | Includes a test suite of valid, invalid, and multilingual questions to measure accuracy |
| 🎨 **Clean, accessible UI** | Government-style Streamlit interface designed for non-technical users |

---

## 🏗️ How It Works

### Data ingestion — from the government portal to structured evidence

```
TN Agrisnet GoScheme
        │
        ▼
 pending_ajax_list
        │
        ▼
       JSON
        │
        ▼
 55+ scheme records
        │
        ├──────────────┐
        ▼              ▼
     JSON             PDF
        │              │
        ▼              ▼
 Structured RAG    Human-readable
   pipeline         source document
        │
        ▼
    Chunking
        ▼
   Embeddings
        ▼
      FAISS
        ▼
       RAG
```

Every record keeps its original metadata intact, so answers can always be traced back to source:

- `source_url`
- `retrieved_date`
- `scheme_list_year`
- `department`
- `scheme_name`
- `eligibility`
- `documents`
- `guidelines_url`

### End-to-end pipeline

```
📥 Web ingestion
      ↓
🧹 Cleaning
      ↓
🧩 Structure detection
      ↓
✂️ Smart chunking
      ↓
🧠 BGE embeddings
      ↓
🗄️ FAISS
      ↓
🔎 Hybrid retrieval
      ↓
🔄 Reranking
      ↓
🛡️ Guardrails
      ↓
🤖 LLM
      ↓
💬 Farmer's answer
      ↓
📚 Official evidence
```

Every answer is delivered **alongside its official evidence** — the farmer never just gets a claim, they get the scheme it came from.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers` (`BAAI/bge-base-en-v1.5`)
- **Vector store**: FAISS
- **Hybrid retrieval**: Dense vector search + BM25 keyword search
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **Generation**: Qwen2.5 (via local Ollama)
- **Chunking**: LangChain text splitters
- **Data ingestion**: `requests` + `BeautifulSoup`
- **Evaluation**: Custom scriptable test harness + LangSmith tracing
- **Visualization**: Plotly

---

## 📂 Project Structure

```
TN_FARMER_SCHEME_AI/
├── app.py                  # Streamlit UI
├── ingestion.py             # Scrapes scheme data from TN Agrisnet
├── loader.py                 # Loads structured scheme data
├── chunker.py                 # Splits scheme text into chunks
├── embedder.py                # Generates embeddings
├── vector_store.py            # Builds & loads the FAISS index
├── retriever.py                # Dense retrieval logic
├── reranker.py                 # Cross-encoder reranking
├── guardrails.py                # Relevance & language filtering
├── generator.py                 # LLM answer generation
├── rag.py                        # Orchestrates the full RAG pipeline
├── evaluator.py                  # Runs the pipeline against test questions
├── test_thresholds.py            # Tunes guardrail score thresholds
├── questions.json                # Test question bank (EN/Tamil/Tanglish)
├── raw_data.json                 # Raw scraped scheme data
├── structured_data.json          # Cleaned scheme records
└── vectorstore/                  # Saved FAISS index + chunks
```

---

## 🚀 Getting Started

### 1. Clone and enter the project
```bash
git clone https://github.com/Seema777111-git/RAG.git
cd RAG/TN_FARMER_SCHEME_AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Ollama locally (for answer generation)
```bash
ollama pull qwen2.5:3b
ollama serve
```

### 4. Build the vector index (first run only)
```bash
python vector_store.py
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 🧪 Evaluating the Pipeline

The project ships with a question bank covering valid, insufficient, and multilingual queries:

```bash
python evaluator.py
```

Use `test_thresholds.py` to inspect retrieval/rerank scores across question types and tune the guardrail thresholds in `guardrails.py`.

---

## 📊 Data Source

All scheme data is sourced live from the **official Tamil Nadu Agrisnet portal**:
🔗 https://www.tnagrisnet.tn.gov.in/people_app/goScheme/

Currently indexed: **55+ government farmer schemes**, each preserving `department`, `scheme_name`, `eligibility`, `documents`, `guidelines_url`, `source_url`, `retrieved_date`, and `scheme_list_year`.

Ingestion produces two outputs from the same data:
- **JSON** → feeds the structured RAG pipeline
- **PDF** → a human-readable source document for transparency and audit

---

## 🌱 Why This Matters

This isn't a general-purpose chatbot wrapped around an LLM — it's a **narrow, grounded, guardrailed** system built for a real accessibility gap. Every answer traces back to an actual government scheme, in the language the farmer is most comfortable asking in.

---

## 🔮 Future Improvements

- [ ] Voice input for low-literacy users
- [ ] SMS/WhatsApp bot interface for wider rural reach
- [ ] Auto-refresh ingestion pipeline to catch newly published schemes
- [ ] District-level eligibility filtering
- [ ] Application deadline reminders

---

## 👤 Author

Built by [Seema777111-git](https://github.com/Seema777111-git)
