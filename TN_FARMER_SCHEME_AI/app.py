import streamlit as st
from rag import load_rag_components, run_rag
##app.py--basic first

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TN Farmer Scheme AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f5f8f5;
    }

    /* Header */
    .gov-header {
        background-color: #176b3a;
        padding: 22px 30px;
        border-radius: 0 0 12px 12px;
        margin-bottom: 25px;
    }

    .gov-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .gov-subtitle {
        color: #e4f3e8;
        font-size: 16px;
    }

    /* Section titles */
    .section-title {
        color: #176b3a;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Cards */
    .answer-card,
    .scheme-card {
        background-color: white;
        border: 1px solid #dce8df;
        border-radius: 12px;
        padding: 22px;
        min-height: 280px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .answer-text {
        color: #26352b;
        font-size: 17px;
        line-height: 1.7;
    }

     /* Scheme Details - readable text */
     .scheme-details-label {
        color: #176b3a !important;
        font-size: 13px;
        font-weight: 700;
        margin-top: 14px;
    }

     .scheme-details-value {
        color: #111111 !important;
        font-size: 16px;
        font-weight: 500;
        line-height: 1.5;
    }
    /* Source */
    .source-card {
        background-color: #eef7f0;
        border-left: 5px solid #176b3a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 15px;
        color: #26352b;
    }
    .source-card a {
        color: #176b3a;
        font-weight: 600;
    }

    /* Ask button */
    .stButton > button {
        background-color: #176b3a;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: 600;
        padding: 10px 25px;
    }

    .stButton > button:hover {
        background-color: #12572f;
        color: white;
    }

    /* RAG process */
    .rag-process {
        background-color: white;
        border: 1px solid #dce8df;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
    }

    .rag-step {
        display: inline-block;
        background-color: #eef7f0;
        border: 1px solid #c8dfcd;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 5px;
        color: #176b3a;
        font-weight: 600;
    }

    .rag-arrow {
        color: #176b3a;
        font-size: 20px;
        font-weight: bold;
    }

    /* RAG Dashboard */
    .dashboard-card {
        background-color: white;
        border: 1px solid #dce8df;
        border-radius: 12px;
        padding: 25px 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .dashboard-metrics {
        display: flex;
        justify-content: space-between;
        text-align: center;
        margin-bottom: 20px;
    }

    .dashboard-metric-value {
        color: #176b3a;
        font-size: 30px;
        font-weight: 800;
    }

    .dashboard-metric-label {
        color: #4a5a4e;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    .dashboard-divider {
        border: none;
        border-top: 1px solid #dce8df;
        margin: 18px 0;
    }

    .dashboard-components {
        display: flex;
        justify-content: space-between;
        text-align: center;
    }

    .dashboard-component-name {
        color: #176b3a;
        font-size: 15px;
        font-weight: 700;
    }

    .dashboard-component-caption {
        color: #68756c;
        font-size: 12px;
        margin-top: 2px;
    }

    .dashboard-status-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }

    .dashboard-status-item {
        color: #26352b;
        font-size: 14px;
        font-weight: 600;
    }

    .dashboard-status-pass {
        color: #176b3a;
    }

    .dashboard-status-blocked {
        color: #b3401e;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #68756c;
        font-size: 13px;
        padding: 25px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="gov-header">
        <div class="gov-title">🌾 TN FARMER SCHEME AI ASSISTANT</div>
        <div class="gov-subtitle">
            Government of Tamil Nadu
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def initialize_rag():
    return load_rag_components()


try:
    components = initialize_rag()
except Exception as e:
    st.error("Unable to load the AI system.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "show_rag" not in st.session_state:
    st.session_state.show_rag = False


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask About Government Schemes</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask about agricultural schemes, eligibility, benefits, "
    "documents or application-related information."
)

question = st.text_area(
    "Question",
    placeholder="Ask in English, தமிழ் or Tanglish...",
    height=100,
    label_visibility="collapsed",
    key="question_input"
)


# ============================================================
# ASK BUTTON
# ============================================================

col1, col2, col3 = st.columns([5, 1, 5])

with col2:
    ask_clicked = st.button(
        "ASK",
        use_container_width=True
    )


# ============================================================
# RUN RAG
# ============================================================

if ask_clicked:

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("Finding relevant government scheme information..."):

            try:
                result = run_rag(
                    query=question.strip(),
                    components=components
                )

                st.session_state.result = result

            except Exception as e:
                st.error("An error occurred while processing your question.")
                st.exception(e)


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.result


if result:

    st.markdown("---")

    # --------------------------------------------------------
    # ANSWER + SCHEME DETAILS
    # --------------------------------------------------------

    answer_col, details_col = st.columns([1.35, 1])

    # ========================================================
    # ANSWER
    # ========================================================

    with answer_col:

        st.markdown(
            '<div class="section-title">💬 Answer</div>',
            unsafe_allow_html=True
        )

        answer = result.get(
            "answer",
            "No answer was generated."
        )

        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-text">
                    {answer}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # ========================================================
    # SCHEME DETAILS
    # ========================================================

    with details_col:

        st.markdown(
            '<div class="section-title">📋 Scheme Details</div>',
            unsafe_allow_html=True
        )

        evidence = result.get("evidence", [])

        if evidence:

            scheme = evidence[0]

            scheme_name = scheme.get("scheme_name", "Not available")
            department = scheme.get("department", "Not available")
            eligibility = scheme.get("eligibility", "Not available")
            documents = scheme.get("documents", "Not available")

            st.markdown(
                f"""
                <div class="scheme-card">
                <div class="scheme-details-label">SCHEME NAME</div>
                <div class="scheme-details-value">{scheme_name}</div>
                <div class="scheme-details-label">DEPARTMENT</div>
                <div class="scheme-details-value">{department}</div>
                <div class="scheme-details-label">ELIGIBILITY</div>
                <div class="scheme-details-value">{eligibility}</div>
                <div class="scheme-details-label">DOCUMENTS</div>
                <div class="scheme-details-value">{documents}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="scheme-card">No specific scheme details were identified.</div>',
                unsafe_allow_html=True
            )
    # ========================================================
    # OFFICIAL SOURCE
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔗 Official Source</div>',
        unsafe_allow_html=True
    )

    source_url = (
        "https://www.tnagrisnet.tn.gov.in/people_app/goScheme/"
    )

    st.markdown(
        f"""
        <div class="source-card">
            <strong>Tamil Nadu Government / TN Agrisnet</strong>
            <br><br>
            Official Government Scheme Information
            <br>
            <a href="{source_url}" target="_blank">
                View Official Source
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ASK ANOTHER QUESTION
    # ========================================================

    st.markdown("---")

    def reset_question():
        st.session_state.result = None
        st.session_state.show_rag = False
        st.session_state.question_input = ""

    st.button("↻ ASK ANOTHER QUESTION", on_click=reset_question)


# ============================================================
# RAG PROCESS BUTTON
# ============================================================

st.markdown("---")

if st.button(
    "🔍 VIEW RAG PROCESS",
    use_container_width=False
):

    st.session_state.show_rag = (
        not st.session_state.show_rag
    )


# ============================================================
# RAG DASHBOARD
# ============================================================

if st.session_state.show_rag:

    st.markdown(
        '<div class="section-title">🔍 RAG Dashboard</div>',
        unsafe_allow_html=True
    )

    if not isinstance(result, dict):

        st.markdown(
            '<div class="dashboard-card">Ask a question to view the RAG dashboard.</div>',
            unsafe_allow_html=True
        )

    else:

        # ----------------------------------------------------
        # GET RAG RESULTS
        # ----------------------------------------------------

        retrieved_results = result.get(
            "retrieved_results", []
        )

        reranked_results = result.get(
            "reranked_results", []
        )

        evidence_results = result.get(
            "evidence", []
        )

        question_status = str(
            result.get(
                "question_status",
                "UNKNOWN"
            )
        ).upper()

        generated = result.get(
            "generated",
            False
        )

        answer_text = str(
            result.get(
                "answer",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # DASHBOARD VALUES
        # ----------------------------------------------------

        retrieved_count = len(
            retrieved_results
        )

        reranked_count = len(
            reranked_results
        )

        evidence_count = len(
            evidence_results
        )

        answer_status = (
            "✓"
            if answer_text
            else "—"
        )

        guardrail_passed = question_status in {
            "VALID",
            "ALLOWED",
            "PASSED"
        }

        guardrail_status_text = (
            "✓ PASSED"
            if guardrail_passed
            else "⚠ BLOCKED"
        )

        guardrail_status_class = (
            "dashboard-status-pass"
            if guardrail_passed
            else "dashboard-status-blocked"
        )

        generation_status = (
            "✓"
            if generated
            else "—"
        )

        # ----------------------------------------------------
        # DASHBOARD CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-metrics">
                    <div>
                        <div class="dashboard-metric-value">{retrieved_count}</div>
                        <div class="dashboard-metric-label">RETRIEVED</div>
                    </div>
                    <div>
                        <div class="dashboard-metric-value">{reranked_count}</div>
                        <div class="dashboard-metric-label">RERANKED</div>
                    </div>
                    <div>
                        <div class="dashboard-metric-value">{evidence_count}</div>
                        <div class="dashboard-metric-label">EVIDENCE</div>
                    </div>
                    <div>
                        <div class="dashboard-metric-value">{answer_status}</div>
                        <div class="dashboard-metric-label">ANSWER</div>
                    </div>
                </div>
                <hr class="dashboard-divider">
                <div class="dashboard-components">
                    <div>
                        <div class="dashboard-component-name">🧠 BGE</div>
                        <div class="dashboard-component-caption">Embedding</div>
                    </div>
                    <div>
                        <div class="dashboard-component-name">🔎 FAISS</div>
                        <div class="dashboard-component-caption">Retrieval</div>
                    </div>
                    <div>
                        <div class="dashboard-component-name">🎯 Reranker</div>
                        <div class="dashboard-component-caption">CrossEncoder</div>
                    </div>
                    <div>
                        <div class="dashboard-component-name">🤖 Qwen 2.5</div>
                        <div class="dashboard-component-caption">Generation</div>
                    </div>
                </div>
                <hr class="dashboard-divider">
                <div class="dashboard-status-row">
                    <div class="dashboard-status-item">
                        🛡️ Guardrails: <span class="{guardrail_status_class}">{guardrail_status_text}</span>
                    </div>
                    <div class="dashboard-status-item">
                        💬 Question: {question_status}
                    </div>
                    <div class="dashboard-status-item">
                        🤖 Generation: {generation_status}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TN Farmer Scheme AI Assistant |
        Powered by Retrieval-Augmented Generation |
        Based on official Tamil Nadu Government scheme information
    </div>
    """,
    unsafe_allow_html=True
)