"""Evaluation tab: run DeepEval metrics over your questions and browse saved reports."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend.api_client import ApiClient, ApiError
from frontend.components.jobs import wait_for_job

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "contextual_relevancy"]
NEEDS_EXPECTED = {"contextual_precision", "contextual_recall"}
METRIC_HELP = {
    "faithfulness": "Is every claim in the answer supported by the retrieved context?",
    "answer_relevancy": "Does the answer actually address the question?",
    "contextual_relevancy": "Is the retrieved context relevant to the question?",
    "contextual_precision": "Are the most relevant passages ranked first? (needs an expected answer)",
    "contextual_recall": "Does the retrieved context cover the expected answer? (needs an expected answer)",
}


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({"question": [""], "expected_output": [""]})


def render(client: ApiClient, stats: dict) -> None:
    cfg = stats["config"]
    st.markdown(
        "Questions run through the full pipeline (guardrails, hybrid retrieval, generation), then **DeepEval** scores each answer "
        "with your configured LLM as the judge. Scores are also attached to the matching LangSmith trace when tracing is on."
    )
    if stats["documents"] == 0:
        st.info("Ingest a document first: evaluation needs something to retrieve from.")

    st.session_state.setdefault("eval_df", _empty_frame())
    st.session_state.setdefault("eval_version", 0)

    st.markdown("#### 1. Test questions")
    gen_col, _ = st.columns([2, 3])
    with gen_col:
        n = st.number_input("Questions to generate", min_value=1, max_value=30, value=5)
        if st.button("Generate from my documents", disabled=stats["documents"] == 0):
            with st.spinner("Writing questions from random passages..."):
                try:
                    cases = client.generate_cases(int(n))
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["eval_df"] = pd.DataFrame(
                        [{"question": c["question"], "expected_output": c.get("expected_output") or ""} for c in cases]
                    )
                    st.session_state["eval_version"] += 1
                    st.rerun()

    edited = st.data_editor(
        st.session_state["eval_df"],
        num_rows="dynamic",
        hide_index=True,
        key=f"eval_editor_{st.session_state['eval_version']}",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "expected_output": st.column_config.TextColumn("Expected answer (optional)", width="large"),
        },
    )

    st.markdown("#### 2. Metrics")
    try:
        available = client.evaluation_metrics()
    except ApiError:
        available = list(METRIC_HELP)
    metrics = st.multiselect("DeepEval metrics", available, default=[m for m in DEFAULT_METRICS if m in available])
    for m in metrics:
        st.caption(f"**{m}**: {METRIC_HELP.get(m, '')}")
    c1, c2 = st.columns(2)
    threshold = c1.slider("Pass threshold", 0.0, 1.0, float(cfg.get("eval_threshold", 0.5)), 0.05)
    mode = c2.selectbox("Retrieval mode under test", ["hybrid", "vector", "graph"])

    cases = [
        {"question": str(q).strip(), "expected_output": (str(e).strip() or None) if isinstance(e, str) else None}
        for q, e in zip(edited["question"], edited["expected_output"])
        if isinstance(q, str) and q.strip()
    ]
    if NEEDS_EXPECTED & set(metrics) and any(not c["expected_output"] for c in cases):
        st.warning("Contextual precision and recall are skipped for questions without an expected answer.")

    st.markdown("#### 3. Run")
    if st.button("Run evaluation", type="primary", disabled=not (cases and metrics)):
        try:
            started = client.start_evaluation({"cases": cases, "metrics": metrics, "threshold": threshold, "mode": mode})
        except ApiError as exc:
            st.error(str(exc))
        else:
            job = wait_for_job(client, started["job_id"], f"Evaluating {len(cases)} question(s)", poll_seconds=1.5)
            if job["status"] == "succeeded":
                st.session_state["eval_report"] = job["result"]
            else:
                st.error(job.get("error") or "Evaluation failed.")

    st.markdown("#### Results")
    _saved_reports(client)
    report = st.session_state.get("eval_report")
    if report:
        render_report(report)
    else:
        st.caption("Run an evaluation, or load a saved report above.")


def _saved_reports(client: ApiClient) -> None:
    try:
        reports = client.list_reports()
    except ApiError:
        return
    if not reports:
        return
    names = [r["name"] for r in reports]
    left, right = st.columns([3, 1])
    choice = left.selectbox("Saved reports", names, label_visibility="collapsed")
    if right.button("Load report"):
        try:
            st.session_state["eval_report"] = client.get_report(choice)
        except ApiError as exc:
            st.error(str(exc))


def render_report(report: dict) -> None:
    st.markdown(f"**{report['name']}** · judge `{report['judge_model']}` · threshold {report['threshold']} · mode {report['mode']}")

    summary = report["summary"]
    for col, s in zip(st.columns(max(len(summary), 1)), summary):
        col.metric(
            s["metric"].replace("_", " ").title(),
            "n/a" if s["mean"] is None else f"{s['mean']:.2f}",
            None if s["pass_rate"] is None else f"{s['pass_rate']:.0%} pass",
            delta_color="off",
        )

    rows = []
    for case in report["cases"]:
        row = {"Question": case["question"][:90], "Blocked": "yes" if case["blocked"] else ""}
        for s in case["scores"]:
            row[s["metric"]] = s["score"] if s["score"] is not None else None
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    for case in report["cases"]:
        with st.expander(case["question"]):
            st.markdown(f"**Answer**  \n{case['answer'] or '(none)'}")
            if case.get("expected_output"):
                st.markdown(f"**Expected**  \n{case['expected_output']}")
            for s in case["scores"]:
                if s["error"]:
                    st.markdown(f"- `{s['metric']}`: not scored ({s['error']})")
                else:
                    verdict = "pass" if s["passed"] else "fail"
                    st.markdown(f"- `{s['metric']}`: **{s['score']:.2f}** ({verdict}). {s['reason'] or ''}")
            if case.get("trace_run_id"):
                st.caption(f"LangSmith run: {case['trace_run_id']}")

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"{report['name']}.json",
        mime="application/json",
    )
