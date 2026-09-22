"""Documents tab: upload PDFs, watch the two ingestion pipelines run, manage the index."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import ApiClient, ApiError
from frontend.components.jobs import wait_for_job
from frontend.components.theme import chip


def render(client: ApiClient) -> None:
    st.markdown(
        "Each PDF is parsed and chunked once, then indexed twice: embeddings go into **FAISS** for semantic search, "
        "and extracted entities and relations go into a **NetworkX** knowledge graph."
    )
    nonce = st.session_state.setdefault("upload_nonce", 0)
    files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, key=f"uploader_{nonce}")

    if files and st.button("Ingest", type="primary"):
        results = []
        for f in files:
            try:
                started = client.upload_document(f.name, f.getvalue())
            except ApiError as exc:
                results.append({"filename": f.name, "error": str(exc)})
                continue
            job = wait_for_job(client, started["job_id"], f"Ingesting {f.name}")
            results.append({"filename": f.name, "job": job})
        st.session_state["ingest_results"] = results
        st.session_state["upload_nonce"] = nonce + 1  # resets the uploader
        st.rerun()

    for item in st.session_state.get("ingest_results", []):
        _render_ingest_result(item)

    st.markdown("#### Indexed documents")
    try:
        docs = client.list_documents()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not docs:
        st.caption("Nothing indexed yet.")
        return

    head = st.columns([5, 1, 1, 1, 2, 1])
    for col, title in zip(head, ["Document", "Pages", "Chunks", "Facts", "Ingested", ""]):
        col.caption(title)
    for d in docs:
        row = st.columns([5, 1, 1, 1, 2, 1])
        row[0].markdown(f"**{d['filename']}**  \n`{d['doc_id']}`")
        row[1].write(d["pages"])
        row[2].write(d["chunks"])
        row[3].write(d["triples"])
        row[4].write(str(d["ingested_at"])[:16].replace("T", " "))
        if row[5].button("Delete", key=f"delete_{d['doc_id']}"):
            try:
                client.delete_document(d["doc_id"])
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop("ingest_results", None)
                st.rerun()


def _render_ingest_result(item: dict) -> None:
    name = item["filename"]
    if item.get("error"):
        st.error(f"{name}: {item['error']}")
        return
    job = item["job"]
    if job["status"] != "succeeded":
        st.error(f"{name}: {job.get('error') or 'ingestion failed'}")
        return
    res = job["result"]
    doc = res["document"]
    with st.container(border=True):
        st.markdown(f"**{name}**" + ("  (replaced the previous version)" if res.get("replaced_existing") else ""))
        a, b, c = st.columns(3)
        a.metric("Pages", doc["pages"])
        b.metric("Chunks embedded", doc["chunks"])
        c.metric("Graph facts", doc["triples"])
        vec, graph = res["vector"], res["graph"]
        st.markdown(
            chip(f"vector: {vec['status']} · {vec['seconds']}s", "vector" if vec["status"] == "ok" else "bad")
            + chip(f"graph: {graph['status']} · {graph['seconds']}s", "graph" if graph["status"] == "ok" else "muted" if graph["status"] == "skipped" else "bad"),
            unsafe_allow_html=True,
        )
        for warning in res.get("warnings", []):
            st.warning(warning)
