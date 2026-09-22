"""Ask tab: question form and full visibility into how the answer was built."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import ApiClient, ApiError
from frontend.components.graph import to_dot
from frontend.components.theme import chip, panel_head

MODES = {"Hybrid (vector + graph)": "hybrid", "Vector only": "vector", "Graph only": "graph"}


def render(client: ApiClient, stats: dict) -> None:
    cfg = stats["config"]
    if stats["documents"] == 0:
        st.info("No documents are indexed yet. Upload a PDF in the **Documents** tab first.")

    with st.form("ask_form"):
        question = st.text_area("Your question", height=90, placeholder="e.g. Who acquired Widget Inc, and when?")
        mode_label = st.radio("Retrieval mode", list(MODES), horizontal=True)
        c1, c2, c3 = st.columns(3)
        top_k = c1.number_input("Passages (top-k)", min_value=1, max_value=50, value=min(int(cfg["vector_top_k"]), 50))
        hops = c2.number_input("Graph hops", min_value=1, max_value=3, value=min(int(cfg["kg_hops"]), 3))
        max_triples = c3.number_input("Max graph facts", min_value=1, max_value=100, value=min(int(cfg["kg_max_triples"]), 100))
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            payload = {
                "question": question.strip(),
                "mode": MODES[mode_label],
                "top_k": int(top_k),
                "kg_hops": int(hops),
                "kg_max_triples": int(max_triples),
            }
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    st.session_state["last_answer"] = client.query(payload)
                except ApiError as exc:
                    st.session_state.pop("last_answer", None)
                    st.error(str(exc))

    result = st.session_state.get("last_answer")
    if result:
        _render_result(result)


# --------------------------------------------------------------------------- result
def _render_result(r: dict) -> None:
    ctx_chunks = {c["chunk"]["chunk_id"]: c for c in r["context"]["chunks"]}
    triple_refs = {(t["subject"], t["predicate"], t["object"]): t.get("ref") for t in r["context"]["triples"]}

    _answer_card(r)
    left, right = st.columns(2, gap="large")
    with left:
        _vector_panel(r, ctx_chunks)
    with right:
        _graph_panel(r, ctx_chunks, triple_refs)
    _pipeline_details(r)


def _answer_card(r: dict) -> None:
    with st.container(border=True):
        st.markdown("**Answer**")
        if r["blocked"]:
            st.error(f"Blocked by guardrail: {r['blocked_by']}")
        st.markdown(r["answer"])

        badges = [chip(label, "graph" if label.startswith("K") else "vector") for label in r["citations"]]
        for stage in ("input", "output"):
            g = r["guardrails"].get(stage)
            if g:
                badges.append(chip(f"{stage} rail: {g['status']}", "bad" if g["status"] in ("blocked", "error") else "guard"))
        badges.append(chip(f"{r['timings_ms'].get('total_ms', 0):,.0f} ms", "muted"))
        st.markdown("".join(badges), unsafe_allow_html=True)

        cited = []
        for label in r["citations"]:
            if label.startswith("C"):
                item = next((c for c in r["context"]["chunks"] if c["ref"] == label), None)
                if item:
                    cited.append(f"**{label}** {item['chunk']['source']}, page {item['chunk']['page']}")
            else:
                t = next((t for t in r["context"]["triples"] if t.get("ref") == label), None)
                if t:
                    cited.append(f"**{label}** {t['subject']} {t['predicate']} {t['object']}")
        if cited:
            st.caption("Cited: " + "  ·  ".join(cited))


def _vector_panel(r: dict, ctx_chunks: dict) -> None:
    panel_head("Vector retrieval · FAISS", "vector")
    if r["mode"] == "graph":
        st.caption("Skipped: this question used graph-only mode.")
        return
    hits = r["vector_hits"]
    if not hits:
        st.info("No passages were retrieved.")
        return
    st.caption(f"{len(hits)} passages ranked by cosine similarity")
    for h in hits:
        c = h["chunk"]
        item = ctx_chunks.get(c["chunk_id"])
        in_prompt = bool(item and item["in_prompt"])
        with st.expander(f"{h['rank']}. {c['source']} · page {c['page']} · similarity {h['score']:.3f}", expanded=h["rank"] == 1):
            tags = [chip(item["ref"] if in_prompt else "not in prompt", "vector" if in_prompt else "muted")]
            if item and item["origin"] == "both":
                tags.append(chip("also cited by a graph fact", "graph"))
            st.markdown("".join(tags), unsafe_allow_html=True)
            st.write(c["text"])


def _graph_panel(r: dict, ctx_chunks: dict, triple_refs: dict) -> None:
    panel_head("Knowledge graph · NetworkX", "graph")
    if r["mode"] == "vector":
        st.caption("Skipped: this question used vector-only mode.")
        return
    kg = r["kg"]
    if not kg["entities"]:
        st.info(f"No entity in the question matched the graph ({kg['graph_nodes_total']} nodes, {kg['graph_edges_total']} edges).")
        return
    st.caption(
        f"{len(kg['entities'])} entities linked · {len(kg['triples'])} facts · "
        f"graph has {kg['graph_nodes_total']} nodes and {kg['graph_edges_total']} edges"
    )

    st.markdown("**Entities found in the question**")
    st.dataframe(
        pd.DataFrame([{"Entity": e["label"], "Match": e["method"], "Score": round(e["score"], 3)} for e in kg["entities"]]),
        hide_index=True,
    )

    if kg["triples"]:
        st.markdown("**Facts retrieved**")
        rows = []
        for t in kg["triples"]:
            source = ctx_chunks.get(t["chunk_id"])
            rows.append(
                {
                    "Ref": triple_refs.get((t["subject"], t["predicate"], t["object"]), ""),
                    "Subject": t["subject"],
                    "Relation": t["predicate"],
                    "Object": t["object"],
                    "Hop": t["hop"],
                    "Score": round(t["score"], 3),
                    "Source": source["ref"] if source and source["in_prompt"] else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if kg["nodes"]:
            st.graphviz_chart(to_dot(kg["nodes"], kg["edges"]))
            st.caption("Dark nodes were linked from the question; light nodes are their neighbours.")

    graph_only = [c for c in r["context"]["chunks"] if c["origin"] == "graph" and c["in_prompt"]]
    if graph_only:
        st.markdown("**Passages pulled in because a graph fact cites them**")
        for c in graph_only:
            with st.expander(f"{c['ref']} · {c['chunk']['source']} · page {c['chunk']['page']}"):
                st.write(c["chunk"]["text"])


def _pipeline_details(r: dict) -> None:
    st.markdown("#### How this answer was built")
    with st.expander("Guardrails (NVIDIA NeMo)"):
        rows = [
            {"Stage": stage, "Status": g["status"], "Rail": g.get("rail") or "", "Latency (ms)": g["latency_ms"], "Message": g.get("message") or ""}
            for stage, g in r["guardrails"].items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No guardrail checks ran.")

    with st.expander("Timings"):
        timings = {k: v for k, v in r["timings_ms"].items() if k != "total_ms"}
        st.metric("Total", f"{r['timings_ms'].get('total_ms', 0):,.0f} ms")
        if timings:
            st.bar_chart(pd.DataFrame({"ms": timings}))

    with st.expander("Context sent to the model"):
        if r["context"]["truncated"]:
            st.warning("The context hit the size limit, so lower-ranked evidence was left out of the prompt.")
        st.code(r["context"]["text"] or "(no context)", language=None)

    with st.expander("Models and trace"):
        st.json(r["models"])
        trace = r.get("trace")
        if trace:
            st.markdown(f"LangSmith run `{trace['run_id']}` in project `{trace['project']}`")
        else:
            st.caption("LangSmith tracing is off. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in `.env` to enable it.")
