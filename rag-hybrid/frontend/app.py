"""Streamlit front end.  Run from the project root:  streamlit run frontend/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `frontend.*` imports work no matter where Streamlit was launched from
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend.api_client import ApiClient  # noqa: E402
from frontend.components import ask, documents, evaluation, sidebar, theme  # noqa: E402
from frontend.config import backend_url  # noqa: E402

st.set_page_config(page_title="Hybrid RAG workbench", layout="wide")
theme.inject()

client = ApiClient(backend_url())
stats = sidebar.render(client)

st.title("Hybrid RAG workbench")
st.caption("Ask questions about your PDFs and see how vector search and the knowledge graph each contributed to the answer.")

tab_ask, tab_docs, tab_eval = st.tabs(["Ask", "Documents", "Evaluation"])
with tab_ask:
    ask.render(client, stats)
with tab_docs:
    documents.render(client)
with tab_eval:
    evaluation.render(client, stats)
