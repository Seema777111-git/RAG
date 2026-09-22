"""Sidebar: backend status, index size, active models and feature flags."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import ApiClient, ApiError
from frontend.components.theme import chip


def render(client: ApiClient) -> dict:
    """Render the sidebar and return the backend stats. Stops the page if the API is unreachable."""
    try:
        stats = client.stats()
    except ApiError as exc:
        with st.sidebar:
            st.markdown("### Hybrid RAG")
            st.error("Backend offline")
        st.error(str(exc))
        st.markdown("Start the API in another terminal, then refresh this page:")
        st.code("make api          # or: scripts/run_api.sh   (Windows: scripts\\run_api.bat)", language="bash")
        st.stop()

    cfg = stats["config"]
    with st.sidebar:
        st.markdown("### Hybrid RAG")
        st.caption(f"Connected to {client.base_url}")
        a, b = st.columns(2)
        a.metric("Documents", stats["documents"])
        b.metric("Chunks", stats["chunks"])
        c, d = st.columns(2)
        c.metric("Graph nodes", stats["graph_nodes"])
        d.metric("Graph edges", stats["graph_edges"])

        st.markdown(
            chip("guardrails on" if cfg["guardrails_enabled"] else "guardrails off", "guard" if cfg["guardrails_enabled"] else "muted")
            + chip(f"LangSmith: {cfg['langsmith_project']}" if cfg["langsmith_tracing"] else "LangSmith off", "graph" if cfg["langsmith_tracing"] else "muted"),
            unsafe_allow_html=True,
        )
        if not cfg["llm_configured"]:
            st.warning(f"{cfg['api_key_env']} is not set. Add it to `.env` and restart the API.")

        if st.button("Test LLM connection"):
            with st.spinner("Calling the model..."):
                try:
                    check = client.check_llm()
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"{check['provider']} / {check['model']} answered in {check['latency_ms']:.0f} ms")

        with st.expander("Models"):
            st.markdown(
                f"**Provider:** {cfg['llm_provider']}  \n"
                f"**Answers:** `{cfg['llm_model']}`  \n"
                f"**Graph extraction:** `{cfg['extraction_model']}`  \n"
                f"**Evaluation judge:** `{cfg['judge_model']}`  \n"
                f"**Embeddings:** `{cfg['embedding_model']}`"
            )
    return stats
