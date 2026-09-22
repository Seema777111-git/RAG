# app.py - Streamlit Interface for Tamil Nadu Government Agriculture Department KG-RAG Pipeline

import streamlit as st
import json
from query_rag import HybridRAGPipeline

st.set_page_config(
    page_title="TN Agriculture Department Welfare Schemes Assistant",
    page_icon="🏛️",
    layout="wide"
)

# Formal TN Government Portal CSS (Navy/Gold palette, clean light background, zero black UI)
st.markdown("""
<style>
    .main {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #212529;
    }
    
    h1 {
        color: #0b3c5d;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #c59b27;
        padding-bottom: 10px;
    }
    
    h2, h3, h4 {
        color: #1d2d44;
    }
    
    .output-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #d1d5db;
        border-left: 6px solid #0b3c5d;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #212529;
    }
    
    .stButton>button {
        background-color: #0b3c5d;
        color: #ffffff;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: 1px solid #07253a;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1d2d44;
        color: #ffffff;
        border-color: #0b3c5d;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Tamil Nadu Government - Agriculture Department Welfare Schemes")

selected_dept = "Agriculture Department"
st.info(f"**Active Department Scope:** {selected_dept}")

user_query = st.text_input("Enter your question regarding agricultural welfare schemes:")

if st.button("Submit Query"):
    if not user_query.strip():
        st.warning("Please enter a valid query before submitting.")
    else:
        with st.spinner("Processing hybrid retrieval and generating official response..."):
            pipeline = HybridRAGPipeline()
            try:
                raw_json_output = pipeline.run_pipeline(user_query, selected_department=selected_dept)
                data = json.loads(raw_json_output)
                
                if data.get("status") == "blocked":
                    st.error(data.get("message"))
                else:
                    st.markdown('<div class="output-card">', unsafe_allow_html=True)
                    st.subheader("Assistant Response")
                    st.write(data.get("response"))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Evaluation Status:** {data.get('evaluation_status')}")
                    with col2:
                        st.info(f"**Selected Scope:** {data.get('selected_department', selected_dept)}")
                        
                    with st.expander("View Retrieved Schemes & Structured JSON Payload"):
                        st.json(data)
            except Exception as e:
                st.error(f"An error occurred while executing the pipeline: {e}")
            finally:
                pipeline.close()